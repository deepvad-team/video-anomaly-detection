import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from sklearn.metrics import auc, roc_curve, precision_recall_curve
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
from dataset import  Dataset_Con_all_feedback_UCF, Dataset_Con_all_feedback_XD
from torch.utils.data import DataLoader
import option
from tqdm import tqdm
import time
import os
from model import Model, Model_V2
from adapter import CopyPlusExtraAdapter
# from datasets.dataset import 
import copy

#UCF
def test(dataloader, model, args, device):
    with torch.no_grad():
        model.eval()
        pred = torch.zeros(0, device=device)

        for i, input in enumerate(dataloader):
            input = input.to(device)
            # print(input.size())
            startime = time.time()
            logits = model(inputs=input)
            # print("done in {0}.".format(time.time() - startime))
            pred = torch.cat((pred, logits))
            
        gt = np.load(args.gt)
        # print(gt.shape)

        pred_seg = list(pred.cpu().detach().numpy())
        np.save("pred_raw_seg_UCF.npy", pred_seg)

        pred_frame = np.repeat(np.array(pred_seg), 16)
        np.save("pred_raw_frame_UCF.npy", pred_frame)

        # gt = gt[:len(pred)] 
        pred = pred_frame

        #ROC CURVE
        fpr, tpr, threshold = roc_curve(list(gt), pred)
        rec_auc = auc(fpr, tpr)
        print('auc: ' + str(rec_auc))

        np.save('threshold_UCF.npy', threshold)
        np.save('fpr_UCF.npy', fpr)
        np.save('tpr_UCF.npy', tpr)
   
        #PR CURVE
        precision, recall, th = precision_recall_curve(list(gt), pred)
        pr_auc = auc(recall, precision)
        np.save('precision_UCF.npy', precision)
        np.save('recall_UCF.npy', recall)

        # np.save('UCF_pred/'+'{}-pred_UCFV1_i3d.npy'.format(epoch), pred)
        return rec_auc, pr_auc
    

#XD
def test_2(dataloader, model, adapter, args, device):
    with torch.no_grad():  
        model.eval()
        adapter.eval()

        pred = torch.zeros(0, device=device)

        for i, input in enumerate(dataloader):

            if i == 0:
                print("raw input:", input.shape)
            input = input.to(device).float() #(B, 1024)

            #변경 부분 (adapter 적용)
            input = adapter(input) #(B, 2048)
            if i == 0:
                print("after adapter:", input.shape)

            logits = model(inputs=input)
                        
            pred = torch.cat((pred, logits))
            
        gt = np.load(args.gt)

        pred_seg = list(pred.cpu().detach().numpy())
        np.save("pred_raw_seg_XD.npy", pred_seg)
        pred_frame = np.repeat(np.array(pred_seg), 16)
        np.save("pred_raw_frame_XD.npy", pred_frame)

        pred = pred_frame

        #ROC CURVE
        fpr, tpr, threshold = roc_curve(list(gt), pred)
        np.save('threshod_XD.npy',threshold)
        np.save('fpr_XD.npy', fpr)
        np.save('tpr_XD.npy', tpr)
        rec_auc = auc(fpr, tpr)
        print('auc: ' + str(rec_auc))

        #PR CURVE
        precision, recall, th = precision_recall_curve(list(gt), pred)
        pr_auc = auc(recall, precision)
        print("AP = ",pr_auc)
        np.save('precision_XD.npy', precision)
        np.save('recall_XD.npy', recall)
        
        return rec_auc, pr_auc



# ---------------------------
# Video diagnostics
# ---------------------------

def run_one_video(X_flat, nalist, vid_idx, adapter, model, device):
    s, e = nalist[vid_idx]
    x_video = X_flat[s:e]   # (T_i, 1024)

    x_video = torch.from_numpy(x_video).float().to(device)

    with torch.no_grad():
        x_2048 = adapter(x_video)                      # (T_i, 2048)
        prob, logit = model(x_2048, return_logits=True)  # (T_i,1), (T_i,1)

    return {
        "start": s,
        "end": e,
        "T": e - s,
        "prob": prob.detach().cpu().numpy(),
        "logit": logit.detach().cpu().numpy(),
    } 


def get_video_scores_and_mask(X_flat, nalist, vid_idx, adapter, model, device, q=0.5):
    """
    q: 하위 q 비율을 normal 후보로 선택
       예) q=0.5 -> 하위 50%
    """
    s, e = nalist[vid_idx]
    x_video = X_flat[s:e]   # (T_i, 1024)
    x_video = torch.from_numpy(x_video).float().to(device)

    model.eval()
    adapter.eval()

    with torch.no_grad():
        x_2048 = adapter(x_video)                       # (T_i,2048)
        prob, logit = model(x_2048, return_logits=True)  # (T_i,1), (T_i,1)

    prob_1d = prob.squeeze(-1)     # (T_i,)
    logit_1d = logit.squeeze(-1)   # (T_i,)

    thresh = torch.quantile(prob_1d, q)
    mask = prob_1d <= thresh       # normal 후보: anomaly score 낮은 것들

    return {
        "start": s,
        "end": e,
        "T": e - s,
        "x_video": x_video,             # (T_i,1024)
        "prob": prob_1d,                # (T_i,)
        "logit": logit_1d,              # (T_i,)
        "threshold": thresh.item(),
        "mask": mask,                   # (T_i,) bool
        "num_selected": int(mask.sum().item())
    }


def inspect_video_energy_terms(
    X_flat,
    nalist,
    vid_idx,
    adapter,
    model,
    device,
    q=0.2,              # 하위 20%만 normal 후보
    min_keep=8,         # 너무 적으면 skip
    sgld_steps=10,
    sgld_lr=0.05,
    sgld_noise=0.01,
):
    """
    비디오 1개에 대해:
    1) baseline prob/logit 계산
    2) low-score normal 후보 선택
    3) E_real 계산
    4) SGLD로 x_tilde 생성
    5) E_fake 계산
    """

    s, e = nalist[vid_idx]
    x_video = X_flat[s:e]  # (T_i, 1024)
    x_video = torch.from_numpy(x_video).float().to(device)

    model.eval()
    adapter.eval()

    # 1) baseline score 계산
    with torch.no_grad():
        x_2048 = adapter(x_video)                         # (T_i, 2048)
        prob, logit = model(x_2048, return_logits=True)  # (T_i,1), (T_i,1)

    prob = prob.squeeze(-1)     # (T_i,)
    logit = logit.squeeze(-1)   # (T_i,)

    # 2) normal 후보 선택: anomaly score 낮은 것들
    thresh = torch.quantile(prob, q)
    mask = prob <= thresh
    num_selected = int(mask.sum().item())

    print(f"[video {vid_idx}] start={s}, end={e}, T={e-s}")
    print(f"threshold={thresh.item():.6f}, selected={num_selected}")

    if num_selected < min_keep:
        print(f"Too few selected segments (< {min_keep}). Skip.")
        return None

    x_sel = x_video[mask].detach()     # (N,1024)
    prob_sel = prob[mask].detach()
    logit_sel = logit[mask].detach()

    # 3) real energy
    #    energy = softplus(logit) : normal이면 작고 anomaly면 큼
    E_real = F.softplus(logit_sel).mean()

    print(f"E_real = {E_real.item():.6f}")
    print(f"selected prob mean = {prob_sel.mean().item():.6f}")
    print(f"selected logit mean = {logit_sel.mean().item():.6f}")

    # 4) SGLD로 x_tilde 만들기
    #    x_sel 주변에서 시작
    x_tilde = (x_sel + sgld_noise * torch.randn_like(x_sel)).detach()

    for step in range(sgld_steps):
        x_tilde.requires_grad_(True)

        x_tilde_2048 = adapter(x_tilde)
        _, logit_tilde = model(x_tilde_2048, return_logits=True)
        logit_tilde = logit_tilde.squeeze(-1)

        E_tilde_now = F.softplus(logit_tilde).mean()

        grad = torch.autograd.grad(E_tilde_now, x_tilde, create_graph=False)[0]

        with torch.no_grad():
            x_tilde = x_tilde - (sgld_lr / 2.0) * grad + sgld_noise * torch.randn_like(x_tilde)

        x_tilde = x_tilde.detach()

    # 5) 최종 fake energy
    with torch.no_grad():
        x_tilde_2048 = adapter(x_tilde)
        prob_fake, logit_fake = model(x_tilde_2048, return_logits=True)

    prob_fake = prob_fake.squeeze(-1)
    logit_fake = logit_fake.squeeze(-1)

    E_fake = F.softplus(logit_fake).mean()

    print(f"E_fake = {E_fake.item():.6f}")
    print(f"fake prob mean = {prob_fake.mean().item():.6f}")
    print(f"fake logit mean = {logit_fake.mean().item():.6f}")

    return {
        "vid_idx": vid_idx,
        "start": int(s),
        "end": int(e),
        "T": int(e - s),
        "threshold": float(thresh.item()),
        "num_selected": num_selected,
        "E_real": float(E_real.item()),
        "E_fake": float(E_fake.item()),
        "prob_mean": float(prob.mean().item()),
        "prob_sel_mean": float(prob_sel.mean().item()),
        "prob_fake_mean": float(prob_fake.mean().item()),
    }


# ---------------------------
# TEA
# ---------------------------

def tea_one_video_one_step(
    X_flat,
    nalist,
    vid_idx,
    adapter,
    model,
    device,
    q=0.2,
    min_keep=8,
    sgld_steps=10,
    sgld_lr=0.05,
    sgld_noise=0.01,
    tea_lr=1e-3,
):
    """
    비디오 1개에 대해:
    - normal 후보 선택
    - E_real / E_fake 계산
    - adapter.ln 만 1 step update
    - update 전/후 score 평균 비교
    """

    # detector는 freeze
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # adapter 중 LN만 update
    adapter.train()
    for p in adapter.parameters():
        p.requires_grad_(False)

    if not hasattr(adapter, "ln"):
        raise ValueError("adapter has no .ln ; use LayerNorm version adapter")

    adapter.ln.weight.requires_grad_(True)
    adapter.ln.bias.requires_grad_(True)

    optimizer = torch.optim.Adam([adapter.ln.weight, adapter.ln.bias], lr=tea_lr)

    s, e = nalist[vid_idx]
    x_video = torch.from_numpy(X_flat[s:e]).float().to(device)

    # update 전 baseline
    with torch.no_grad():
        x_2048_before = adapter(x_video)
        prob_before, logit_before = model(x_2048_before, return_logits=True)

    prob_before = prob_before.squeeze(-1)
    logit_before = logit_before.squeeze(-1)

    thresh = torch.quantile(prob_before, q)
    mask = prob_before <= thresh

    if int(mask.sum().item()) < min_keep:
        print(f"[video {vid_idx}] too few selected segments. skip.")
        return None

    x_sel = x_video[mask].detach()

    # -------- E_real --------
    x_sel_2048 = adapter(x_sel)
    _, logit_real = model(x_sel_2048, return_logits=True)
    logit_real = logit_real.squeeze(-1)
    E_real = F.softplus(logit_real).mean()

    # -------- E_fake via SGLD --------
    x_tilde = (x_sel + sgld_noise * torch.randn_like(x_sel)).detach()

    # SGLD에서는 input grad 필요, adapter/model 파라미터 grad는 optimizer 대상만
    for _ in range(sgld_steps):
        x_tilde.requires_grad_(True)

        x_tilde_2048 = adapter(x_tilde)
        _, logit_tilde = model(x_tilde_2048, return_logits=True)
        logit_tilde = logit_tilde.squeeze(-1)
        E_tilde_now = F.softplus(logit_tilde).mean()

        grad = torch.autograd.grad(E_tilde_now, x_tilde, create_graph=False)[0]

        with torch.no_grad():
            x_tilde = x_tilde - (sgld_lr / 2.0) * grad + sgld_noise * torch.randn_like(x_tilde)
        x_tilde = x_tilde.detach()

    x_tilde_2048 = adapter(x_tilde)
    _, logit_fake = model(x_tilde_2048, return_logits=True)
    logit_fake = logit_fake.squeeze(-1)
    E_fake = F.softplus(logit_fake).mean()

    # -------- TEA loss --------
    loss = E_real - E_fake

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    # update 후 baseline 다시 확인
    adapter.eval()
    with torch.no_grad():
        x_2048_after = adapter(x_video)
        prob_after, logit_after = model(x_2048_after, return_logits=True)

    prob_after = prob_after.squeeze(-1)
    logit_after = logit_after.squeeze(-1)

    print(f"[video {vid_idx}]")
    print(f" selected = {int(mask.sum().item())}")
    print(f" E_real   = {E_real.item():.6f}")
    print(f" E_fake   = {E_fake.item():.6f}")
    print(f" loss     = {loss.item():.6f}")
    print(f" prob mean before = {prob_before.mean().item():.6f}")
    print(f" prob mean after  = {prob_after.mean().item():.6f}")

    return {
        "vid_idx": vid_idx,
        "selected": int(mask.sum().item()),
        "E_real": float(E_real.item()),
        "E_fake": float(E_fake.item()),
        "loss": float(loss.item()),
        "prob_mean_before": float(prob_before.mean().item()),
        "prob_mean_after": float(prob_after.mean().item()),
    }


# ---------------------------
# TEA helper (episodic)
# ---------------------------

def _segment_gt_from_gt(gt, total_T, frame_repeat=16):
    gt = np.asarray(gt).astype(np.int64).reshape(-1)

    if len(gt) == total_T:
        seg_gt = gt
        gt_mode = "segment"
    elif len(gt) == total_T * frame_repeat:
        seg_gt = gt.reshape(total_T, frame_repeat).max(axis=1)
        gt_mode = "frame"
    else:
        raise ValueError(
            f"GT length mismatch: len(gt)={len(gt)}, total_T={total_T}, "
            f"expected {total_T} or {total_T * frame_repeat}"
        )
    return seg_gt, gt_mode


def _tea_update_one_video(
    x_video,          # torch tensor, (T_i, 1024)
    adapter_episode,
    model,
    q=0.2,
    min_keep=8,
    sgld_steps=10,
    sgld_lr=0.05,
    sgld_noise=0.01,
    tea_lr=1e-3,
    tea_steps_per_video=1,
):
    """
    비디오 하나에 대해 adapter_episode.ln만 업데이트.
    return: adapter_episode (updated), debug_info
    """

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    if not hasattr(adapter_episode, "ln"):
        raise ValueError("adapter_episode must have .ln for LN-only TEA")

    # adapter 전체 freeze, LN만 update
    adapter_episode.train()
    for p in adapter_episode.parameters():
        p.requires_grad_(False)
    adapter_episode.ln.weight.requires_grad_(True)
    adapter_episode.ln.bias.requires_grad_(True)

    optimizer = torch.optim.Adam(
        [adapter_episode.ln.weight, adapter_episode.ln.bias],
        lr=tea_lr
    )

    debug = []

    for step_idx in range(tea_steps_per_video):
        # 1) 현재 비디오 baseline score
        with torch.no_grad():
            x_2048 = adapter_episode(x_video)
            prob, logit = model(x_2048, return_logits=True)

        prob = prob.squeeze(-1)    # (T_i,)
        logit = logit.squeeze(-1)  # (T_i,)

        thresh = torch.quantile(prob, q)
        mask = prob <= thresh
        num_selected = int(mask.sum().item())

        if num_selected < min_keep:
            debug.append({
                "step": step_idx,
                "skipped": True,
                "num_selected": num_selected,
                "threshold": float(thresh.item()),
            })
            break

        x_sel = x_video[mask].detach()      # (N,1024)
        prob_sel = prob[mask].detach()
        logit_sel = logit[mask].detach()

        # 2) real energy
        x_sel_2048 = adapter_episode(x_sel)
        _, logit_real = model(x_sel_2048, return_logits=True)
        logit_real = logit_real.squeeze(-1)
        E_real = F.softplus(logit_real).mean()

        # 3) SGLD fake 생성
        x_tilde = (x_sel + sgld_noise * torch.randn_like(x_sel)).detach()

        for _ in range(sgld_steps):
            x_tilde.requires_grad_(True)

            x_tilde_2048 = adapter_episode(x_tilde)
            _, logit_tilde = model(x_tilde_2048, return_logits=True)
            logit_tilde = logit_tilde.squeeze(-1)

            E_tilde_now = F.softplus(logit_tilde).mean()

            grad = torch.autograd.grad(E_tilde_now, x_tilde, create_graph=False)[0]

            with torch.no_grad():
                x_tilde = x_tilde - (sgld_lr / 2.0) * grad + sgld_noise * torch.randn_like(x_tilde)

            x_tilde = x_tilde.detach()

        x_tilde_2048 = adapter_episode(x_tilde)
        _, logit_fake = model(x_tilde_2048, return_logits=True)
        logit_fake = logit_fake.squeeze(-1)
        E_fake = F.softplus(logit_fake).mean()

        # 4) TEA loss = E_real - E_fake
        loss = E_real - E_fake

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        debug.append({
            "step": step_idx,
            "skipped": False,
            "num_selected": num_selected,
            "threshold": float(thresh.item()),
            "E_real": float(E_real.item()),
            "E_fake": float(E_fake.item()),
            "loss": float(loss.item()),
            "prob_mean": float(prob.mean().item()),
            "prob_sel_mean": float(prob_sel.mean().item()),
            "logit_sel_mean": float(logit_sel.mean().item()),
        })

    return adapter_episode, debug

# ---------------------------
# 전체 xd evaluation용 함수
# ---------------------------

def eval_xd_with_episodic_tea(
    X_flat,              # numpy, (total_T, 1024)
    nalist,              # numpy, (num_videos, 2)
    gt,                  # numpy, (total_T,) or (total_T*16,)
    adapter,
    model,
    device,
    frame_repeat=16,

    use_tea=True,
    q=0.2,
    min_keep=8,
    sgld_steps=10,
    sgld_lr=0.05,
    sgld_noise=0.01,
    tea_lr=1e-3,
    tea_steps_per_video=1,

    verbose_every=100,
):
    """
    XD 전체 test셋:
    - 비디오별 adapter episodic reset
    - optional TEA
    - 최종 segment score 저장
    - frame-level GT와 맞춰 AUC/AP 계산
    """

    total_T = X_flat.shape[0]
    seg_gt, gt_mode = _segment_gt_from_gt(gt, total_T, frame_repeat=frame_repeat)

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    seg_scores_all = np.zeros(total_T, dtype=np.float32)
    tea_logs = []

    for vid_idx in range(len(nalist)):
        s, e = nalist[vid_idx]
        x_video_np = X_flat[s:e]                            # (T_i,1024)
        x_video = torch.from_numpy(x_video_np).float().to(device)

        # 비디오마다 adapter 리셋
        adapter_episode = copy.deepcopy(adapter).to(device)
        adapter_episode.eval()

        # 1) optional TEA
        if use_tea:
            adapter_episode, debug = _tea_update_one_video(
                x_video=x_video,
                adapter_episode=adapter_episode,
                model=model,
                q=q,
                min_keep=min_keep,
                sgld_steps=sgld_steps,
                sgld_lr=sgld_lr,
                sgld_noise=sgld_noise,
                tea_lr=tea_lr,
                tea_steps_per_video=tea_steps_per_video,
            )
        else:
            debug = None

        # 2) adaptation 후 최종 inference
        adapter_episode.eval()
        with torch.no_grad():
            x_2048 = adapter_episode(x_video)
            prob, _ = model(x_2048, return_logits=True)

        prob = prob.squeeze(-1).detach().cpu().numpy()     # (T_i,)
        seg_scores_all[s:e] = prob

        if debug is not None:
            tea_logs.append({
                "vid_idx": vid_idx,
                "start": int(s),
                "end": int(e),
                "T": int(e - s),
                "debug": debug,
            })

        if (vid_idx % verbose_every == 0) or (vid_idx == len(nalist) - 1):
            print(f"[{vid_idx+1}/{len(nalist)}] done, T={e-s}")

    # 3) metric 계산
    if gt_mode == "segment":
        y_true = seg_gt
        y_score = seg_scores_all
    else:
        # segment score를 16배 반복해서 frame-level score로 맞춤
        y_true = np.asarray(gt).reshape(-1)
        y_score = np.repeat(seg_scores_all, frame_repeat)

    auc = roc_auc_score(y_true, y_score)
    ap = average_precision_score(y_true, y_score)

    return {
        "auc": float(auc),
        "ap": float(ap),
        "seg_scores_all": seg_scores_all,
        "tea_logs": tea_logs,
    }

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

def bootstrap_video_ci(
    seg_scores_base,     # (total_T,)
    seg_scores_tea,      # (total_T,)
    nalist,              # (num_videos, 2)
    gt,                  # (total_T,) or (total_T*16,)
    frame_repeat=16,
    n_boot=1000,
    seed=42,
):
    rng = np.random.default_rng(seed)

    total_T = len(seg_scores_base)
    gt = np.asarray(gt).reshape(-1)

    # GT 모드 정리
    if len(gt) == total_T:
        gt_mode = "segment"
        seg_gt = gt.astype(np.int64)
    elif len(gt) == total_T * frame_repeat:
        gt_mode = "frame"
        seg_gt = gt.reshape(total_T, frame_repeat).max(axis=1).astype(np.int64)
    else:
        raise ValueError(
            f"GT length mismatch: len(gt)={len(gt)}, total_T={total_T}, "
            f"expected {total_T} or {total_T*frame_repeat}"
        )

    num_videos = len(nalist)

    delta_auc_list = []
    delta_ap_list = []

    for _ in range(n_boot):
        # 비디오 인덱스 복원추출
        boot_vids = rng.integers(0, num_videos, size=num_videos)

        y_true_parts = []
        y_base_parts = []
        y_tea_parts = []

        for vid_idx in boot_vids:
            s, e = nalist[vid_idx]

            if gt_mode == "segment":
                y_true_parts.append(seg_gt[s:e])
                y_base_parts.append(seg_scores_base[s:e])
                y_tea_parts.append(seg_scores_tea[s:e])
            else:
                # frame-level metric과 맞추기 위해 segment score를 16번 반복
                y_true_parts.append(gt[s*frame_repeat:e*frame_repeat])
                y_base_parts.append(np.repeat(seg_scores_base[s:e], frame_repeat))
                y_tea_parts.append(np.repeat(seg_scores_tea[s:e], frame_repeat))

        y_true = np.concatenate(y_true_parts)
        y_base = np.concatenate(y_base_parts)
        y_tea = np.concatenate(y_tea_parts)

        auc_base = roc_auc_score(y_true, y_base)
        ap_base = average_precision_score(y_true, y_base)

        auc_tea = roc_auc_score(y_true, y_tea)
        ap_tea = average_precision_score(y_true, y_tea)

        delta_auc_list.append(auc_tea - auc_base)
        delta_ap_list.append(ap_tea - ap_base)

    delta_auc = np.array(delta_auc_list)
    delta_ap = np.array(delta_ap_list)

    def ci95(x):
        return np.percentile(x, [2.5, 50, 97.5])

    auc_ci = ci95(delta_auc)
    ap_ci = ci95(delta_ap)

    print("[Bootstrap Δ = TEA - Baseline]")
    print(f"ΔAUC median={auc_ci[1]:.6f}, 95% CI=({auc_ci[0]:.6f}, {auc_ci[2]:.6f})")
    print(f"ΔAP  median={ap_ci[1]:.6f}, 95% CI=({ap_ci[0]:.6f}, {ap_ci[2]:.6f})")

    return {
        "delta_auc": delta_auc,
        "delta_ap": delta_ap,
        "auc_ci95": auc_ci,
        "ap_ci95": ap_ci,
    }

# ---------------------------
# Main test loop
# ---------------------------

if __name__ == '__main__':
    args = option.parser.parse_args()
    device = torch.device("cuda")

    # 1. GT / dataset
    gt = np.load(args.gt)
        #변경 (추가) 부분: (145649, 1024) 데이터 그대로 일단 받아오기
    xd_dataset = Dataset_Con_all_feedback_XD(args, test_mode=True)
    X_flat = xd_dataset.con_all
    nalist = np.load("list/nalist_XD_test.npy")
    print("X_flat shape:", X_flat.shape)
    print("nalist shape:", nalist.shape)

    '''
    test_loader = DataLoader(Dataset_Con_all_feedback_UCF(args, test_mode=True), 
                            batch_size=args.batch_size, shuffle=False, 
                            num_workers=args.workers, pin_memory=False, drop_last=False)
    
    '''
    '''
    test_loader = DataLoader(xd_dataset,
                             batch_size=args.batch_size, shuffle=False, 
                            num_workers=args.workers, pin_memory=False, drop_last=False)
    
    '''
   
    # 2. adapter
    #변경(추가) 부분. 1024 차원으로 들어오는 TEST 데이터 -> 2048 
    adapter = CopyPlusExtraAdapter(d=1024, use_ln=True).to(device)
    #torch.save(adapter.state_dict(), "adapter_init.pt") #baseline adapter 고정(저장) - 초기 1번만
    adapter.load_state_dict(torch.load("adapter_init.pt",  map_location=device))
    adapter.eval()

    # 3. model
    model = Model_V2(args.feature_size).to(device)
    #model_dict = model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load('../../C2FPL/ckpt/UCFfinal(git).pkl').items()})
    model_dict = model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load('../../minjeong/unsupervised_ckpt/UCF_final_20260301_031008_pgn3aode.pkl').items()})
    model.eval()
    

    '''
    for i in range(3):
        s, e = nalist[i]
        x_video = X_flat[s:e]
        print(f"video {i}: start={s}, end={e}, T={e-s}, x_video.shape={x_video.shape}")
    
    
    #Ereal과 Efake가 값이 나오기는 하는지 1차 확인.
    res0 = inspect_video_energy_terms(
        X_flat=X_flat,
        nalist=nalist,
        vid_idx=0,
        adapter=adapter,
        model=model,
        device=device,
        q=0.2,
        min_keep=8,
        sgld_steps=10,
        sgld_lr=0.05,
        sgld_noise=0.01,
    )    print(res0)
    
    #TEA 1-step만 적용해보기
    adapter_test = copy.deepcopy(adapter).to(device) 
    res = tea_one_video_one_step(
        X_flat=X_flat,
        nalist=nalist,
        vid_idx=0,
        adapter=adapter_test,
        model=model,
        device=device,
        q=0.2,
        min_keep=8,
        sgld_steps=10,
        sgld_lr=0.05,
        sgld_noise=0.01,
        tea_lr=1e-3,
    )    print(res)
    '''
    
    #4. baseline (adapter만, TEA 없음)
    res_base = eval_xd_with_episodic_tea(
        X_flat=X_flat,
        nalist=nalist,
        gt=gt,
        adapter=adapter,
        model=model,
        device=device,
        frame_repeat=16,
        use_tea=False,
        verbose_every=100,
    )
    print("\n[BASELINE]")
    print("AUC:", res_base["auc"])
    print("AP :", res_base["ap"])

    # 5. episodic TEA
    res_tea = eval_xd_with_episodic_tea(
        X_flat=X_flat,
        nalist=nalist,
        gt=gt,
        adapter=adapter,
        model=model,
        device=device,
        frame_repeat=16,
        use_tea=True,
        q=0.2,
        min_keep=8,
        sgld_steps=10,
        sgld_lr=0.05,
        sgld_noise=0.01,
        tea_lr=1e-3,
        tea_steps_per_video=3,
        verbose_every=100,
    )
    print("\n[EPISODIC TEA]")
    print("AUC:", res_tea["auc"])
    print("AP :", res_tea["ap"])

    #부트스트랩으로 확인
    boot_res = bootstrap_video_ci(
        seg_scores_base=res_base["seg_scores_all"],
        seg_scores_tea=res_tea["seg_scores_all"],
        nalist=nalist,
        gt=gt,
        frame_repeat=16,
        n_boot=1000,
        seed=42,
    )
    #scores = test(test_loader, model, args, device) #UCF
    #scores = test_2(test_loader, model, adapter, args, device) #XD

    #변경 (추가) 부분: video-wise
    #scores = test_xd_videowise(X_flat, nalist, model, adapter, args, device)
    