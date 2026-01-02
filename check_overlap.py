import os, glob
import numpy as np
import torch
from sklearn.metrics import auc, roc_curve
from dataset import UCFTestVideoDataset
from model import Model_V2
import option

FRAMES_PER_SEG = 16
BETA = 0.2  # 논문 beta (윈도우 길이 비율)

def runs_of_ones(x):
    runs=[]
    i=0
    n=len(x)
    while i<n:
        if x[i]==1:
            j=i
            while j<n and x[j]==1:
                j+=1
            runs.append((i,j))
            i=j
        else:
            i+=1
    return runs

@torch.no_grad()
def main():
    args = option.parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ===== paths =====
    nalist_path = "list/nalist_test_i3d.npy"
    gt_path     = args.gt  # 보통 list/gt-ucf-RTFM.npy
    conall_path = "Concat_test_10.npy"

    nalist = np.load(nalist_path)
    gt_all = np.load(gt_path, allow_pickle=True).astype(np.float32)

    # sanity: test에서 sum(T*16) == len(gt) 이어야 함
    Tsum = int(np.sum((nalist[:,1]-nalist[:,0]) * FRAMES_PER_SEG))
    print("sum(T*16) =", Tsum, "len(gt) =", len(gt_all))
    assert Tsum == len(gt_all), "nalist_test와 gt 길이가 안 맞음"

    # ===== model & ckpt =====
    model = Model_V2(args.feature_size).to(device)
    pattern = os.path.join("unsupervised_ckpt", f"{args.datasetname}_best_*.pkl")
    ckpts = glob.glob(pattern)
    assert ckpts, f"no ckpt: {pattern}"
    ckpt_path = max(ckpts, key=os.path.getmtime)
    print("Loading:", ckpt_path)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict({k.replace('module.', ''): v for k, v in state.items()})
    model.eval()

    # ===== dataloader (batch_size=1 필수) =====
    ds = UCFTestVideoDataset(conall_path=conall_path, nalist_path=nalist_path)

    ptr = 0
    stats = []

    for vid in range(len(ds)):
        x = ds[vid]  # 보통 (T,2048)
        if isinstance(x, (tuple, list)):
            x = x[0]
        x = torch.as_tensor(x, dtype=torch.float32, device=device)
        if x.dim() == 2:
            x = x.unsqueeze(0)  # (1,T,2048)

        logits = model(inputs=x)              # (1,T,1)
        seg = logits.squeeze(0).squeeze(-1)   # (T,)
        seg = seg.detach().cpu().numpy().astype(np.float32)

        T = seg.shape[0]
        need = T * FRAMES_PER_SEG
        gt_i = gt_all[ptr:ptr+need]
        ptr += need

        gt_frame = (gt_i > 0.5).astype(np.int32)
        gt_seg = gt_frame.reshape(T, FRAMES_PER_SEG).max(axis=1)  # (T,)

        # ===== window 선택: 논문 FPL 스타일로 "연속 구간" 가정 =====
        w = int(np.ceil(BETA * T))
        w = max(1, min(w, T))

        # detector 점수는 "클수록 이상"이니까 평균이 가장 큰 window 선택
        csum = np.cumsum(np.concatenate(([0.0], seg)))
        win_mean = (csum[w:] - csum[:-w]) / w  # (T-w+1,)
        start = int(np.argmax(win_mean))
        ywin = np.zeros(T, dtype=np.int32)
        ywin[start:start+w] = 1

        # segment overlap
        inter_seg = int(((ywin==1) & (gt_seg==1)).sum())
        union_seg = int(((ywin==1) | (gt_seg==1)).sum())
        pred_pos = int(ywin.sum())
        gt_pos = int(gt_seg.sum())
        seg_prec = inter_seg / (pred_pos + 1e-6)
        seg_rec  = inter_seg / (gt_pos + 1e-6)
        seg_iou  = inter_seg / (union_seg + 1e-6)

        # frame overlap (윈도우를 frame으로 확장)
        pred_frame = np.repeat(ywin, FRAMES_PER_SEG)
        inter_fr = int(((pred_frame==1) & (gt_frame==1)).sum())
        union_fr = int(((pred_frame==1) | (gt_frame==1)).sum())
        pred_fr_pos = int(pred_frame.sum())
        gt_fr_pos = int(gt_frame.sum())
        fr_prec = inter_fr / (pred_fr_pos + 1e-6)
        fr_rec  = inter_fr / (gt_fr_pos + 1e-6)
        fr_iou  = inter_fr / (union_fr + 1e-6)

        stats.append((vid, T, w, start, seg_iou, seg_prec, seg_rec, fr_iou, fr_prec, fr_rec))

    assert ptr == len(gt_all)
    print("ptr ok:", ptr)

    # ===== summary =====
    arr = np.array(stats, dtype=object)
    mean_fr_iou = float(np.mean(arr[:,7].astype(np.float32)))
    mean_fr_rec = float(np.mean(arr[:,9].astype(np.float32)))
    print("\nMEAN frame IoU:", mean_fr_iou)
    print("MEAN frame Recall:", mean_fr_rec)

    # top / bottom
    stats_sorted = sorted(stats, key=lambda x: x[7], reverse=True)  # frame_iou 기준
    print("\n[Top 10] frame IoU")
    for s in stats_sorted[:10]:
        vid,T,w,start,seg_iou,seg_p,seg_r,fr_iou,fr_p,fr_r = s
        print(f"vid={vid:3d} T={T:4d} w={w:3d} start={start:4d}  fr_iou={fr_iou:.3f} frP={fr_p:.3f} frR={fr_r:.3f}")

    print("\n[Bottom 10] frame IoU")
    for s in stats_sorted[-10:]:
        vid,T,w,start,seg_iou,seg_p,seg_r,fr_iou,fr_p,fr_r = s
        print(f"vid={vid:3d} T={T:4d} w={w:3d} start={start:4d}  fr_iou={fr_iou:.3f} frP={fr_p:.3f} frR={fr_r:.3f}")

if __name__ == "__main__":
    main()
