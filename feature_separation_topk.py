import numpy as np
from pathlib import Path
import argparse


def load_segment_feats(npy_path):
    """
    return seg_feats: (T, D)
      npy: (T, 10, D) or (T, D)
    """
    x = np.load(npy_path).astype(np.float32)
    if x.ndim == 3:
        x = x.mean(axis=1)  # (T, D)
    return x  # (T, D)


def pool_feature(seg_feats, mode="mean", k=5):
    """
    seg_feats: (T, D)
    mode:
      - mean
      - norm_topk: segment L2 norm 상위 k개 평균
      - max_norm: norm 제일 큰 segment 1개
    """
    if mode == "mean":
        return seg_feats.mean(axis=0)

    norms = np.linalg.norm(seg_feats, axis=1)
    if mode == "max_norm":
        idx = int(np.argmax(norms))
        return seg_feats[idx]

    if mode == "norm_topk":
        k = min(k, len(norms))
        topk_idx = np.argsort(norms)[-k:]
        return seg_feats[topk_idx].mean(axis=0)

    raise ValueError(f"unknown mode: {mode}")


def cosine_sim(a, b, eps=1e-8):
    a = a / (np.linalg.norm(a) + eps)
    b = b / (np.linalg.norm(b) + eps)
    return float(np.dot(a, b))


def euclidean_dist(a, b):
    return float(np.linalg.norm(a - b))


def mean_dist_to_centroid(feats, centroid, metric="euclidean"):
    dists = []
    for f in feats:
        if metric == "euclidean":
            d = euclidean_dist(f, centroid)
        else:
            d = 1.0 - cosine_sim(f, centroid)
        dists.append(d)
    return float(np.mean(dists)), float(np.std(dists))


def leave_one_out_nearest_centroid(normal_feats, abnormal_feats, metric="euclidean"):
    correct = 0
    total = 0

    # normal
    for i in range(len(normal_feats)):
        x = normal_feats[i]
        n_cent = np.delete(normal_feats, i, axis=0).mean(axis=0)
        a_cent = abnormal_feats.mean(axis=0)
        if metric == "euclidean":
            pred = 0 if euclidean_dist(x, n_cent) < euclidean_dist(x, a_cent) else 1
        else:
            pred = 0 if cosine_sim(x, n_cent) > cosine_sim(x, a_cent) else 1
        correct += int(pred == 0)
        total += 1

    # abnormal
    for i in range(len(abnormal_feats)):
        x = abnormal_feats[i]
        a_cent = np.delete(abnormal_feats, i, axis=0).mean(axis=0)
        n_cent = normal_feats.mean(axis=0)
        if metric == "euclidean":
            pred = 0 if euclidean_dist(x, n_cent) < euclidean_dist(x, a_cent) else 1
        else:
            pred = 0 if cosine_sim(x, n_cent) > cosine_sim(x, a_cent) else 1
        correct += int(pred == 1)
        total += 1

    return correct / total


def collect_video_feats(dir_path, pool_mode, k):
    files = sorted(Path(dir_path).glob("*.npy"))
    feats = []
    for f in files:
        seg = load_segment_feats(f)
        v = pool_feature(seg, mode=pool_mode, k=k)
        feats.append(v)
    return np.stack(feats, axis=0), [p.name for p in files]


def main(args):
    normal_feats, _ = collect_video_feats(args.normal_dir, args.pool, args.k)
    abnormal_feats, _ = collect_video_feats(args.abnormal_dir, args.pool, args.k)

    n_cent = normal_feats.mean(axis=0)
    a_cent = abnormal_feats.mean(axis=0)

    cent_cos = cosine_sim(n_cent, a_cent)
    cent_euc = euclidean_dist(n_cent, a_cent)

    n_within_euc, _ = mean_dist_to_centroid(normal_feats, n_cent, "euclidean")
    a_within_euc, _ = mean_dist_to_centroid(abnormal_feats, a_cent, "euclidean")
    n_within_cos, _ = mean_dist_to_centroid(normal_feats, n_cent, "cosine")
    a_within_cos, _ = mean_dist_to_centroid(abnormal_feats, a_cent, "cosine")

    sep_ratio_euc = cent_euc / (0.5 * (n_within_euc + a_within_euc) + 1e-8)
    sep_ratio_cos = (1.0 - cent_cos) / (0.5 * (n_within_cos + a_within_cos) + 1e-8)

    acc_euc = leave_one_out_nearest_centroid(normal_feats, abnormal_feats, "euclidean")
    acc_cos = leave_one_out_nearest_centroid(normal_feats, abnormal_feats, "cosine")

    print("=" * 60)
    print(f"[pool] mode={args.pool}, k={args.k}")
    print(f"[centroid] cos={cent_cos:.6f}, euc={cent_euc:.6f}")
    print(f"[sep ratio] euc={sep_ratio_euc:.6f}, cos={sep_ratio_cos:.6f}")
    print(f"[LOO acc] euc={acc_euc:.4f}, cos={acc_cos:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--normal_dir", required=True)
    ap.add_argument("--abnormal_dir", required=True)
    ap.add_argument("--pool", default="mean", choices=["mean", "norm_topk", "max_norm"])
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()
    main(args)