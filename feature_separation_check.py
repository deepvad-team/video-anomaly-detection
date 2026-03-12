import numpy as np
from pathlib import Path
import argparse


def load_video_feature(npy_path):
    """
    npy shape:
      (T, 10, 2048) or (T, 2048)

    return:
      pooled video feature: (2048,)
    """
    x = np.load(npy_path).astype(np.float32)

    if x.ndim == 3:
        # (T, 10, 2048) -> (T, 2048)
        x = x.mean(axis=1)

    # (T, 2048) -> (2048,)
    v = x.mean(axis=0)
    return v


def l2_normalize(x, eps=1e-8):
    return x / (np.linalg.norm(x) + eps)


def cosine_sim(a, b, eps=1e-8):
    a = a / (np.linalg.norm(a) + eps)
    b = b / (np.linalg.norm(b) + eps)
    return float(np.dot(a, b))


def euclidean_dist(a, b):
    return float(np.linalg.norm(a - b))


def summarize_group(name, feats):
    norms = np.linalg.norm(feats, axis=1)
    print(f"[{name}] N={len(feats)}")
    print(f"  norm mean={norms.mean():.6f}, std={norms.std():.6f}")
    print(f"  norm min={norms.min():.6f}, max={norms.max():.6f}")


def compute_centroid(feats):
    return feats.mean(axis=0)


def mean_dist_to_centroid(feats, centroid, metric="euclidean"):
    dists = []
    for f in feats:
        if metric == "euclidean":
            d = euclidean_dist(f, centroid)
        elif metric == "cosine":
            d = 1.0 - cosine_sim(f, centroid)
        else:
            raise ValueError(metric)
        dists.append(d)
    return float(np.mean(dists)), float(np.std(dists))


def leave_one_out_nearest_centroid(normal_feats, abnormal_feats, metric="euclidean"):
    """
    Leave-one-out nearest centroid classification.
    label 0 = normal, 1 = abnormal
    """
    correct = 0
    total = 0

    # normal samples
    for i in range(len(normal_feats)):
        x = normal_feats[i]
        n_rest = np.delete(normal_feats, i, axis=0)
        a_rest = abnormal_feats

        n_cent = n_rest.mean(axis=0)
        a_cent = a_rest.mean(axis=0)

        if metric == "euclidean":
            d_n = euclidean_dist(x, n_cent)
            d_a = euclidean_dist(x, a_cent)
            pred = 0 if d_n < d_a else 1
        elif metric == "cosine":
            s_n = cosine_sim(x, n_cent)
            s_a = cosine_sim(x, a_cent)
            pred = 0 if s_n > s_a else 1
        else:
            raise ValueError(metric)

        correct += int(pred == 0)
        total += 1

    # abnormal samples
    for i in range(len(abnormal_feats)):
        x = abnormal_feats[i]
        a_rest = np.delete(abnormal_feats, i, axis=0)
        n_rest = normal_feats

        a_cent = a_rest.mean(axis=0)
        n_cent = n_rest.mean(axis=0)

        if metric == "euclidean":
            d_n = euclidean_dist(x, n_cent)
            d_a = euclidean_dist(x, a_cent)
            pred = 0 if d_n < d_a else 1
        elif metric == "cosine":
            s_n = cosine_sim(x, n_cent)
            s_a = cosine_sim(x, a_cent)
            pred = 0 if s_n > s_a else 1
        else:
            raise ValueError(metric)

        correct += int(pred == 1)
        total += 1

    return correct / total


def collect_features_from_dir(root_dir):
    root = Path(root_dir)
    files = sorted(root.glob("*.npy"))
    feats = []
    names = []

    for f in files:
        try:
            v = load_video_feature(f)
            feats.append(v)
            names.append(f.name)
        except Exception as e:
            print(f"[skip] {f}: {e}")

    if len(feats) == 0:
        raise RuntimeError(f"No valid npy files found in {root_dir}")

    return np.stack(feats, axis=0), names


def main(args):
    normal_feats, normal_names = collect_features_from_dir(args.normal_dir)
    abnormal_feats, abnormal_names = collect_features_from_dir(args.abnormal_dir)

    if args.l2norm:
        normal_feats = np.stack([l2_normalize(x) for x in normal_feats], axis=0)
        abnormal_feats = np.stack([l2_normalize(x) for x in abnormal_feats], axis=0)
        print("[info] applied L2 normalization to video-level features")

    print("=" * 60)
    summarize_group("normal", normal_feats)
    summarize_group("abnormal", abnormal_feats)

    n_cent = compute_centroid(normal_feats)
    a_cent = compute_centroid(abnormal_feats)

    cent_cos = cosine_sim(n_cent, a_cent)
    cent_euc = euclidean_dist(n_cent, a_cent)

    print("=" * 60)
    print("[class centroid relation]")
    print(f"  centroid cosine similarity = {cent_cos:.6f}")
    print(f"  centroid euclidean distance = {cent_euc:.6f}")

    n_within_euc_mean, n_within_euc_std = mean_dist_to_centroid(normal_feats, n_cent, metric="euclidean")
    a_within_euc_mean, a_within_euc_std = mean_dist_to_centroid(abnormal_feats, a_cent, metric="euclidean")

    n_within_cos_mean, n_within_cos_std = mean_dist_to_centroid(normal_feats, n_cent, metric="cosine")
    a_within_cos_mean, a_within_cos_std = mean_dist_to_centroid(abnormal_feats, a_cent, metric="cosine")

    print("=" * 60)
    print("[within-class compactness]")
    print(f"  normal -> centroid (euclidean): mean={n_within_euc_mean:.6f}, std={n_within_euc_std:.6f}")
    print(f"  abnormal -> centroid (euclidean): mean={a_within_euc_mean:.6f}, std={a_within_euc_std:.6f}")
    print(f"  normal -> centroid (1-cosine): mean={n_within_cos_mean:.6f}, std={n_within_cos_std:.6f}")
    print(f"  abnormal -> centroid (1-cosine): mean={a_within_cos_mean:.6f}, std={a_within_cos_std:.6f}")

    print("=" * 60)
    print("[separation heuristic]")
    sep_ratio_euc = cent_euc / (0.5 * (n_within_euc_mean + a_within_euc_mean) + 1e-8)
    sep_ratio_cos = (1.0 - cent_cos) / (0.5 * (n_within_cos_mean + a_within_cos_mean) + 1e-8)
    print(f"  separation ratio (euclidean) = {sep_ratio_euc:.6f}")
    print(f"  separation ratio (cosine)    = {sep_ratio_cos:.6f}")

    acc_euc = leave_one_out_nearest_centroid(normal_feats, abnormal_feats, metric="euclidean")
    acc_cos = leave_one_out_nearest_centroid(normal_feats, abnormal_feats, metric="cosine")

    print("=" * 60)
    print("[leave-one-out nearest centroid accuracy]")
    print(f"  euclidean accuracy = {acc_euc:.4f}")
    print(f"  cosine accuracy    = {acc_cos:.4f}")
    print("=" * 60)

    print("[quick interpretation]")
    if max(acc_euc, acc_cos) >= 0.8:
        print("  -> feature space separation looks fairly strong.")
    elif max(acc_euc, acc_cos) >= 0.65:
        print("  -> feature space shows some class separation.")
    else:
        print("  -> class separation is weak or heavily overlapped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--normal_dir", type=str, required=True, help="directory of normal npy files")
    parser.add_argument("--abnormal_dir", type=str, required=True, help="directory of abnormal npy files")
    parser.add_argument("--l2norm", action="store_true", help="L2 normalize video-level features before analysis")
    args = parser.parse_args()
    main(args)