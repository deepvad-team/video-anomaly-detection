import matplotlib.pyplot as plt
import torch
from sklearn.metrics import auc, roc_curve, precision_recall_curve
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
import numpy as np
import torch

def diagnose_xd_baseline(
    X_flat,          # numpy array, shape: (total_T, 1024)
    nalist,          # numpy array, shape: (num_videos, 2), [start, end)
    gt,              # numpy array, shape: (total_T,) or (total_T*16,)
    adapter,
    model,
    device,
    frame_repeat=16,
    topk=5
):
    """
    baseline sanity check:
    - segment gt or frame gt를 자동 처리
    - 비디오별 mean/max/topk_mean score 계산
    - normal / abnormal 분리 정도 출력
    """

    total_T = X_flat.shape[0]

    # 1) GT를 segment-level로 맞추기
    gt = np.asarray(gt).astype(np.int64).reshape(-1)

    if len(gt) == total_T:
        seg_gt = gt
        print(f"[GT] using segment-level GT directly: {seg_gt.shape}")
    elif len(gt) == total_T * frame_repeat:
        seg_gt = gt.reshape(total_T, frame_repeat).max(axis=1)
        print(f"[GT] converted frame-level GT -> segment-level GT: {seg_gt.shape}")
    else:
        raise ValueError(
            f"GT length mismatch: len(gt)={len(gt)}, total_T={total_T}, "
            f"expected either {total_T} or {total_T * frame_repeat}"
        )

    adapter.eval()
    model.eval()

    stats = []

    for vid_idx in range(len(nalist)):
        s, e = nalist[vid_idx]
        x_video = torch.from_numpy(X_flat[s:e]).float().to(device)   # (T_i, 1024)

        with torch.no_grad():
            x_2048 = adapter(x_video)                                # (T_i, 2048)
            prob, logit = model(x_2048, return_logits=True)          # (T_i,1), (T_i,1)

        prob = prob.squeeze(-1).detach().cpu().numpy()               # (T_i,)
        logit = logit.squeeze(-1).detach().cpu().numpy()             # (T_i,)

        label = int(seg_gt[s:e].max() > 0)  # 비디오 내 하나라도 이상이면 abnormal video

        k = min(topk, len(prob))
        topk_vals = np.sort(prob)[-k:]

        stats.append({
            "vid_idx": vid_idx,
            "label": label,
            "T": int(e - s),
            "mean_prob": float(np.mean(prob)),
            "max_prob": float(np.max(prob)),
            "topk_mean": float(np.mean(topk_vals)),
            "mean_logit": float(np.mean(logit)),
            "max_logit": float(np.max(logit)),
        })

    # 2) 요약 출력
    normal = [d for d in stats if d["label"] == 0]
    abnormal = [d for d in stats if d["label"] == 1]

    def summarize(group, name):
        if len(group) == 0:
            print(f"\n[{name}] empty")
            return
        mean_probs = np.array([d["mean_prob"] for d in group])
        max_probs = np.array([d["max_prob"] for d in group])
        topk_means = np.array([d["topk_mean"] for d in group])

        print(f"\n[{name}] n={len(group)}")
        print(f" mean_prob : {mean_probs.mean():.4f} ± {mean_probs.std():.4f}")
        print(f" max_prob  : {max_probs.mean():.4f} ± {max_probs.std():.4f}")
        print(f" topk_mean : {topk_means.mean():.4f} ± {topk_means.std():.4f}")

    summarize(normal, "NORMAL")
    summarize(abnormal, "ABNORMAL")

    # 3) 샘플 몇 개 보기
    print("\n[ABNORMAL examples: high topk_mean]")
    abnormal_sorted = sorted(abnormal, key=lambda x: x["topk_mean"], reverse=True)
    for d in abnormal_sorted[:5]:
        print(
            f" vid={d['vid_idx']:4d} | T={d['T']:4d} | "
            f"mean={d['mean_prob']:.4f} | max={d['max_prob']:.4f} | topk_mean={d['topk_mean']:.4f}"
        )

    print("\n[NORMAL examples: high topk_mean]  <-- false alarm 많은 애들")
    normal_sorted = sorted(normal, key=lambda x: x["topk_mean"], reverse=True)
    for d in normal_sorted[:5]:
        print(
            f" vid={d['vid_idx']:4d} | T={d['T']:4d} | "
            f"mean={d['mean_prob']:.4f} | max={d['max_prob']:.4f} | topk_mean={d['topk_mean']:.4f}"
        )

    return stats

if __name__ == '__main__':
    args = option.parser.parse_args()
    gt = np.load(args.gt)
    #con_all = np.load('{}.npy'.format(args.conall))
    device = torch.device("cuda")


    #변경(추가) 부분. 1024 차원으로 들어오는 TEST 데이터 -> 2048 
    #1단계는 LN 없이
    adapter = CopyPlusExtraAdapter(d=1024, use_ln=False).to(device)

    model = Model_V2(args.feature_size).to(device)
    '''
    test_loader = DataLoader(Dataset_Con_all_feedback_UCF(args, test_mode=True), 
                            batch_size=args.batch_size, shuffle=False, 
                            num_workers=args.workers, pin_memory=False, drop_last=False)
    
                            '''
    #변경 (추가) 부분: (145649, 1024) 데이터 그대로 일단 받아오기
    xd_dataset = Dataset_Con_all_feedback_XD(args, test_mode=True)

    test_loader = DataLoader(xd_dataset,
                             batch_size=args.batch_size, shuffle=False, 
                            num_workers=args.workers, pin_memory=False, drop_last=False)
    
    X_flat = xd_dataset.con_all
    nalist = np.load("list/nalist_XD_test.npy")
    print("X_flat shape:", X_flat.shape)
    print("nalist shape:", nalist.shape)

    ''' (videowise test 준비단계 확인용)
    res = run_one_video(X_flat, nalist, 0, adapter, model, device
    )
    print("video T:", res["T"])
    print("prob shape:", res["prob"].shape)
    print("logit shape:", res["logit"].shape)
    print("prob[:5]:", res["prob"][:5].reshape(-1))
    print("logit[:5]:", res["logit"][:5].reshape(-1))
    '''
    '''
    res = get_video_scores_and_mask(
    X_flat, nalist, vid_idx=780,
    adapter=adapter, model=model, device=device, q=0.5
    )
    print("T =", res["T"])
    print("threshold =", res["threshold"])
    print("selected =", res["num_selected"])
    print("prob[:10] =", res["prob"][:10])
    print("mask[:10] =", res["mask"][:10])

    for i in range(3):
        s, e = nalist[i]
        x_video = X_flat[s:e]
        print(f"video {i}: start={s}, end={e}, T={e-s}, x_video.shape={x_video.shape}")
'''
    stats = diagnose_xd_baseline(
        X_flat=X_flat,
        nalist=nalist,
        gt=gt,              # 네가 쓰는 XD GT numpy
        adapter=adapter,
        model=model,
        device=device,
        frame_repeat=16,
        topk=5
    )
        
    #model_dict = model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load('../../C2FPL/ckpt/UCFfinal(git).pkl').items()})
    model_dict = model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load('unsupervised_ckpt/UCF_best_20260218_165841_tgkaplua.pkl').items()})
    
    #scores = test(test_loader, model, args, device) #UCF
    scores = test_2(test_loader, model, adapter, args, device) #XD

    #변경 (추가) 부분: video-wise
    #scores = test_xd_videowise(X_flat, nalist, model, adapter, args, device)
    