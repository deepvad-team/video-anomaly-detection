import numpy as np
from pathlib import Path
import argparse

def load_seg_feats(npy_path):
    x = np.load(npy_path).astype(np.float32)
    # (T, 10, D) -> (T, D)
    if x.ndim == 3:
        x = x.mean(axis=1)
    return x  # (T, D)

def video_mu_sigma(npy_path):
    seg = load_seg_feats(npy_path)  # (T, D)
    norms = np.linalg.norm(seg, axis=1)  # (T,)
    return float(norms.mean()), float(norms.std())

def roc_auc(y_true, y_score):
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

def collect_mu_sigma(dir_path, label):
    files = sorted(Path(dir_path).glob("*.npy"))
    mus, sigmas, ys, names = [], [], [], []
    for f in files:
        mu, sigma = video_mu_sigma(f)
        mus.append(mu); sigmas.append(sigma); ys.append(label); names.append(f.name)
    return np.array(ys, np.int64), np.array(mus, np.float32), np.array(sigmas, np.float32), names

def summarize(name, arr):
    print(f"[{name}] mean={arr.mean():.6f}, std={arr.std():.6f}, min={arr.min():.6f}, max={arr.max():.6f}")

def main(args):
    y0, mu0, sg0, _ = collect_mu_sigma(args.normal_dir, 0)
    y1, mu1, sg1, _ = collect_mu_sigma(args.abnormal_dir, 1)

    y = np.concatenate([y0, y1])
    mu = np.concatenate([mu0, mu1])
    sg = np.concatenate([sg0, sg1])

    # 간단 점수 3종: mu, sigma, mu+lambda*sigma
    lam = args.lam
    score_mu = mu
    score_sg = sg
    score_mix = mu + lam * sg

    print("="*60)
    print(f"[CPL stats] normal={len(y0)} abnormal={len(y1)}  (dir={args.tag})")
    summarize("mu(normal)", mu0); summarize("mu(abn)", mu1)
    summarize("sigma(normal)", sg0); summarize("sigma(abn)", sg1)

    print("-"*60)
    print(f"[AUROC] mu     = {roc_auc(y, score_mu):.4f}   | [AP] {pr_auc(y, score_mu):.4f}")
    print(f"[AUROC] sigma  = {roc_auc(y, score_sg):.4f}   | [AP] {pr_auc(y, score_sg):.4f}")
    print(f"[AUROC] mu+{lam}*sigma = {roc_auc(y, score_mix):.4f}   | [AP] {pr_auc(y, score_mix):.4f}")
    print("="*60)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--normal_dir", required=True)
    ap.add_argument("--abnormal_dir", required=True)
    ap.add_argument("--lam", type=float, default=1.0, help="mix weight for mu + lam*sigma")
    ap.add_argument("--tag", type=str, default="run")
    args = ap.parse_args()
    main(args)