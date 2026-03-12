import numpy as np
from pathlib import Path
import argparse

import torch
from model import Model_V2


def load_segment_feats(npy_path):
    x = np.load(npy_path).astype(np.float32)
    if x.ndim == 3:
        x = x.mean(axis=1)  # (T, D)
    return x  # (T, D)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def roc_auc(y_true, y_score):
    # y_true: {0,1}
    order = np.argsort(y_score)[::-1]
    y = y_true[order]
    P = y.sum()
    N = len(y) - P
    if P == 0 or N == 0:
        return float("nan")
    tps = np.cumsum(y)
    fps = np.cumsum(1 - y)
    tpr = tps / P
    fpr = fps / N
    # trapezoid
    return float(np.trapz(tpr, fpr))


def pr_auc(y_true, y_score):
    order = np.argsort(y_score)[::-1]
    y = y_true[order]
    P = y.sum()
    if P == 0:
        return float("nan")
    tps = np.cumsum(y)
    fps = np.cumsum(1 - y)
    precision = tps / (tps + fps + 1e-12)
    recall = tps / P
    return float(np.trapz(precision, recall))


def pool_scores(seg_scores, mode="topk", k=5):
    T = len(seg_scores)
    if mode == "mean":
        return float(seg_scores.mean())
    if mode == "max":
        return float(seg_scores.max())
    if mode == "topk":
        k = min(k, T)
        idx = np.argsort(seg_scores)[-k:]
        return float(seg_scores[idx].mean())
    raise ValueError(mode)


@torch.no_grad()
def video_score_from_npy(model, device, npy_path, score_mode="topk", k=5, use_prob=True, batch=256):
    seg = load_segment_feats(npy_path)  # (T,D)
    x = torch.from_numpy(seg).float().to(device)

    # chunk inference
    outs = []
    for i in range(0, len(x), batch):
        prob, logit = model(x[i:i+batch], return_logits=True)
        if use_prob:
            outs.append(prob.squeeze(-1).detach().cpu().numpy())
        else:
            outs.append(logit.squeeze(-1).detach().cpu().numpy())
    seg_scores = np.concatenate(outs, axis=0)
    return pool_scores(seg_scores, mode=score_mode, k=k)


def collect_scores(dir_path, label, model, device, score_mode, k, use_prob):
    files = sorted(Path(dir_path).glob("*.npy"))
    ys = []
    ss = []
    for f in files:
        s = video_score_from_npy(model, device, f, score_mode=score_mode, k=k, use_prob=use_prob)
        ys.append(label)
        ss.append(s)
    return np.array(ys, dtype=np.int64), np.array(ss, dtype=np.float32), [p.name for p in files]


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[device]", device)


    model = Model_V2(2048).to(device) # <- 생성자 인자 필요하면 넣기
    ckpt = torch.load(args.ckpt, map_location=device)

    model.load_state_dict({k.replace('module.', ''): v for k, v in ckpt.items()})

    model.to(device).eval()

    y0, s0, _ = collect_scores(args.normal_dir, 0, model, device, args.pool, args.k, args.use_prob)
    y1, s1, _ = collect_scores(args.abnormal_dir, 1, model, device, args.pool, args.k, args.use_prob)

    y = np.concatenate([y0, y1])
    s = np.concatenate([s0, s1])

    auc = roc_auc(y, s)
    ap = pr_auc(y, s)

    print("=" * 60)
    print(f"[score] use={'prob' if args.use_prob else 'logit'}  pool={args.pool} k={args.k}")
    print(f"[AUROC] {auc:.4f}")
    print(f"[AP]    {ap:.4f}")
    print(f"[normal] mean={s0.mean():.4f} std={s0.std():.4f} max={s0.max():.4f}")
    print(f"[abn]    mean={s1.mean():.4f} std={s1.std():.4f} max={s1.max():.4f}")
    print("=" * 60)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--normal_dir", required=True)
    ap.add_argument("--abnormal_dir", required=True)
    ap.add_argument("--ckpt", required=True, help="detector checkpoint path")
    ap.add_argument("--pool", default="topk", choices=["mean", "max", "topk"])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--use_prob", action="store_true", help="use probability instead of logits")
    args = ap.parse_args()
    main(args)