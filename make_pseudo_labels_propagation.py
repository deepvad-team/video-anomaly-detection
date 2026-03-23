# 0323 추가 --------------------------------------

import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse as sp

def normalize_video_feat(video_feat: np.ndarray, mode: str = "standard") -> np.ndarray:
    #video_feat: (T, D)

    if mode == "none":
        return video_feat

    if mode == "standard":
        mean = video_feat.mean(axis=0, keepdims=True)
        std = video_feat.std(axis=0, keepdims=True) + 1e-8
        return (video_feat - mean) / std

    if mode == "l2":
        norm = np.linalg.norm(video_feat, axis=1, keepdims=True) + 1e-8
        return video_feat / norm

    raise ValueError(f"Unknown normalization mode: {mode}")


def build_feature_similarity(video_feat: np.ndarray, temperature: float = 0.2) -> np.ndarray:
    # video_feat: (T, D)
    # return: Wf in [0, 1], shape (T, T)
    
    sim = cosine_similarity(video_feat)
    sim = np.clip(sim, -1.0, 1.0)

    # map to [0, 1]
    sim = (sim + 1.0) / 2.0

    # optional sharpening
    sim = np.exp((sim - 1.0) / max(temperature, 1e-8))

    # self-loop keep
    np.fill_diagonal(sim, 1.0)
    return sim


def build_temporal_similarity(T: int, sigma_t: float = 5.0) -> np.ndarray:
    """
    return: Wt shape (T, T)
    """
    idx = np.arange(T)
    dist = np.abs(idx[:, None] - idx[None, :]).astype(np.float32)
    Wt = np.exp(-(dist ** 2) / (2 * sigma_t * sigma_t))
    np.fill_diagonal(Wt, 1.0)
    return Wt


def normalize_affinity(W: np.ndarray) -> np.ndarray:
    """
    S = D^{-1/2} W D^{-1/2}
    """
    d = W.sum(axis=1) + 1e-8
    d_inv_sqrt = 1.0 / np.sqrt(d)
    S = d_inv_sqrt[:, None] * W * d_inv_sqrt[None, :]
    return S


def propagate_normality(
    S: np.ndarray,
    seed_idx: np.ndarray,
    alpha: float = 0.9,
    num_iters: int = 50,
) -> np.ndarray:
    """
    S: normalized affinity (T, T)
    seed_idx: indices of benign prefix segments
    return: z in [0, 1], shape (T,)
    """
    T = S.shape[0]

    y = np.zeros(T, dtype=np.float32)
    y[seed_idx] = 1.0

    z = y.copy()

    for _ in range(num_iters):
        z = alpha * (S @ z) + (1.0 - alpha) * y
        # clamp seeds to 1
        z[seed_idx] = 1.0

    # normalize to [0, 1]
    z = z - z.min()
    if z.max() > 0:
        z = z / z.max()

    return z

'''
def compute_global_normal_score(video_feat: np.ndarray, seed_feat: np.ndarray) -> np.ndarray:
    """
    논문의 global normal list 아이디어를 단순화한 버전.
    seed_feat: (K, D) benign prefix features
    return: d in [0, 1], higher => more abnormal
    """
    
    # cosine sim between each snippet and each seed
    sim = cosine_similarity(video_feat, seed_feat)  # (T, K)
    sim = np.clip(sim, -1.0, 1.0)

    # best matching seed 기준
    best_sim = sim.max(axis=1)

    # abnormal score
    d = 1.0 - ((best_sim + 1.0) / 2.0)
    d = np.clip(d, 0.0, 1.0)
    return d
'''

from sklearn.metrics.pairwise import cosine_similarity

def compute_global_abnormal_score(video_feat: np.ndarray, normal_list: np.ndarray) -> np.ndarray:
    """
    video_feat: (T, D)
    normal_list: (K, D)  # selected high-confidence normal video mean features

    return:
      d in [0,1], shape (T,)
    """
    sim = cosine_similarity(video_feat, normal_list)   # (T, K)
    sim = np.clip(sim, -1.0, 1.0)

    # best matching global normal prototype
    best_sim = sim.max(axis=1)   # (T,)

    d = 1.0 - best_sim           # 논문식 정의와 최대한 맞춤
    d = np.clip(d, 0.0, 2.0)

    # weight와 pseudo-label 비교를 위해 [0,1]로 압축
    d = d / 2.0
    return d.astype(np.float32)   


def make_hard_label_and_weight(
    z: np.ndarray,
    d: np.ndarray,
    seed_idx: np.ndarray,
    normal_q: float = 0.20,
    abnormal_q: float = 0.05,
):
    """
    z: propagated normality score in [0,1], higher => more normal
    d: global abnormal score in [0,1], higher => more abnormal

    return:
      label: {-1,0,1}  (-1 ignore, 0 normal, 1 abnormal)
      weight: [0,1]
    """
    T = len(z)
    label = np.full(T, -1, dtype=np.int8)
    weight = np.zeros(T, dtype=np.float32)

    # exact counts
    n_normal = max(1, int(T * normal_q))
    n_abnormal = max(1, int(T * abnormal_q))

    # seeds are always normal
    seed_mask = np.zeros(T, dtype=bool)
    seed_mask[seed_idx] = True

    # normal: top z
    normal_rank = np.argsort(-z)   # descending
    normal_idx = normal_rank[:n_normal]

    # abnormal: low z + high d 를 같이 만족하는 후보 중 bottom selection
    # 단순 z만 쓰지 말고 d까지 같이 쓰는 게 훨씬 안전함
    abnormal_score = (1.0 - z) + d   # 높을수록 abnormal candidate
    abnormal_rank = np.argsort(-abnormal_score)

    abnormal_idx = []
    normal_idx_set = set(normal_idx.tolist())

    for idx in abnormal_rank:
        if seed_mask[idx]:
            continue
        if idx in normal_idx_set:
            continue
        abnormal_idx.append(idx)
        if len(abnormal_idx) >= n_abnormal:
            break
    abnormal_idx = np.array(abnormal_idx, dtype=np.int64)

    # assign labels
    label[normal_idx] = 0
    label[abnormal_idx] = 1
    label[seed_idx] = 0   # seed 강제 normal
    '''
    # reliability weight
    pseudo_abn = 1.0 - z
    weight_all = np.exp(-np.abs(d - pseudo_abn)).astype(np.float32)

    weight[normal_idx] = weight_all[normal_idx]
    weight[abnormal_idx] = weight_all[abnormal_idx]
    weight[seed_idx] = 1.0
    '''
    # 논문식 weight: w = exp(-|d - y_hat|)
    # y_hat은 anomaly pseudo label이므로
    # normal -> 0, abnormal -> 1
    pseudo_y = np.zeros(T, dtype=np.float32)
    pseudo_y[abnormal_idx] = 1.0
    pseudo_y[normal_idx] = 0.0
    pseudo_y[seed_idx] = 0.0

    weight_all = np.exp(-np.abs(d - pseudo_y)).astype(np.float32)

    used_mask = (label != -1)
    weight[used_mask] = weight_all[used_mask]
    weight[seed_idx] = 1.0


    return label, weight, n_normal, n_abnormal




def build_local_affinity_sparse(
    video_feat: np.ndarray,
    sigma_t: float = 5.0,
    window: int = 20,
    topk: int = 10,
) -> sp.csr_matrix:
    """
    video_feat: (T, D)
    return: sparse normalized affinity S, shape (T, T)
    """

    T, D = video_feat.shape

    # cosine similarity용 L2 normalize
    feat = video_feat.astype(np.float32)
    feat = feat / (np.linalg.norm(feat, axis=1, keepdims=True) + 1e-8)

    rows = []
    cols = []
    vals = []

    for i in range(T):
        left = max(0, i - window)
        right = min(T, i + window + 1)

        neigh = feat[left:right]              # (M, D)
        sim = neigh @ feat[i]                 # cosine similarity, shape (M,)
        sim = np.clip(sim, -1.0, 1.0)
        sim = (sim + 1.0) / 2.0               # [0,1]

        # temporal weight
        t_idx = np.arange(left, right)
        dt = (t_idx - i).astype(np.float32)
        temp = np.exp(-(dt ** 2) / (2.0 * sigma_t * sigma_t))

        w = sim * temp

        # 자기 자신은 유지
        self_pos = i - left
        w[self_pos] = max(w[self_pos], 1.0)

        # top-k만 유지
        k = min(topk, len(w))
        keep_idx = np.argpartition(w, -k)[-k:]
        keep_cols = t_idx[keep_idx]
        keep_vals = w[keep_idx]

        rows.extend([i] * len(keep_cols))
        cols.extend(keep_cols.tolist())
        vals.extend(keep_vals.tolist())

    W = sp.csr_matrix((vals, (rows, cols)), shape=(T, T), dtype=np.float32)

    # 대칭화
    W = 0.5 * (W + W.T)

    # self-loop 추가
    W = W + sp.eye(T, dtype=np.float32, format='csr')

    # S = D^{-1/2} W D^{-1/2}
    deg = np.array(W.sum(axis=1)).reshape(-1) + 1e-8
    d_inv_sqrt = 1.0 / np.sqrt(deg)
    D_inv_sqrt = sp.diags(d_inv_sqrt)

    S = D_inv_sqrt @ W @ D_inv_sqrt
    return S.tocsr()


def propagate_normality_sparse(
    S: sp.csr_matrix,
    seed_idx: np.ndarray,
    alpha: float = 0.9,
    num_iters: int = 30,
) -> np.ndarray:
    """
    S: sparse normalized affinity
    seed_idx: benign prefix indices
    return: z in [0,1]
    """
    T = S.shape[0]

    y = np.zeros(T, dtype=np.float32)
    y[seed_idx] = 1.0

    z = y.copy()

    for _ in range(num_iters):
        z = alpha * (S @ z) + (1.0 - alpha) * y
        z[seed_idx] = 1.0

    z = z - z.min()
    z = z / (z.max() + 1e-8)
    return z.astype(np.float32)





def select_highconf_normal_videos(video_scores_list, all_video_feats, ratio=0.15):
    """
    video_scores_list: list of z arrays, each shape (T,)
    all_video_feats: list of video feature arrays, each shape (T, D)

    return:
      normal_video_indices
      normal_list: (K, D) mean features of selected normal videos
    """
    video_std = np.array([np.std(z) for z in video_scores_list], dtype=np.float32)

    K = max(1, int(len(video_std) * ratio))
    normal_video_indices = np.argsort(video_std)[:K]

    normal_list = []
    for vid in normal_video_indices:
        mean_feat = all_video_feats[vid].mean(axis=0)
        normal_list.append(mean_feat.astype(np.float32))

    normal_list = np.stack(normal_list, axis=0)  # (K, D)
    return normal_video_indices, normal_list


def generate_propagation_pseudo_labels(
    train_data,
    nalist,
    prefix_len: int = 5,
    feature_norm: str = "standard",
    sigma_t: float = 5.0,
    alpha: float = 0.9,
    num_iters: int = 30,
    normal_q: float = 0.20,
    abnormal_q: float = 0.05,
    normal_video_ratio: float = 0.15,
):
    all_video_feats = []
    video_scores_list = []

    # ---------------- 1st pass: propagation only ----------------
    for info in tqdm(nalist, desc="Pass 1: propagation"):
        start, end = int(info[0]), int(info[1])
        video_feat = np.mean(train_data[start:end], axis=1).astype(np.float32)

        if len(video_feat) == 0:
            continue

        video_feat = normalize_video_feat(video_feat, mode=feature_norm)
        T = len(video_feat)
        k = min(prefix_len, T)
        seed_idx = np.arange(k)

        S = build_local_affinity_sparse(
            video_feat=video_feat,
            sigma_t=sigma_t,
            window=20,
            topk=10
        )

        z = propagate_normality_sparse(
            S=S,
            seed_idx=seed_idx,
            alpha=alpha,
            num_iters=num_iters
        )

        all_video_feats.append(video_feat)
        video_scores_list.append(z)

    # ---------------- global normal list ----------------
    normal_video_indices, normal_list = select_highconf_normal_videos(
        video_scores_list=video_scores_list,
        all_video_feats=all_video_feats,
        ratio=normal_video_ratio
    )

    print("selected high-conf normal videos:", len(normal_video_indices))
    print("normal list shape:", normal_list.shape)

    # ---------------- 2nd pass: label + weight ----------------
    all_scores = []
    all_labels = []
    all_weights = []
    normal_counts = []
    abnormal_counts = []

    for video_feat, z in tqdm(list(zip(all_video_feats, video_scores_list)), desc="Pass 2: label+weight"):
        T = len(video_feat)
        k = min(prefix_len, T)
        seed_idx = np.arange(k)

        d = compute_global_abnormal_score(video_feat, normal_list)

        label, weight, n_normal, n_abnormal = make_hard_label_and_weight(
            z=z,
            d=d,
            seed_idx=seed_idx,
            normal_q=normal_q,
            abnormal_q=abnormal_q
        )

        all_scores.append(z.astype(np.float32))
        all_labels.append(label.astype(np.int8))
        all_weights.append(weight.astype(np.float32))
        normal_counts.append(n_normal)
        abnormal_counts.append(n_abnormal)

    scores_flat = np.concatenate(all_scores, axis=0)
    labels_flat = np.concatenate(all_labels, axis=0)
    weights_flat = np.concatenate(all_weights, axis=0)

    print("\n==================== Statistics ====================")
    print("Total segments:", len(scores_flat))
    print("Normal (0):", int((labels_flat == 0).sum()), f"({100 * (labels_flat == 0).mean():.2f}%)")
    print("Abnormal (1):", int((labels_flat == 1).sum()), f"({100 * (labels_flat == 1).mean():.2f}%)")
    print("Ignore (-1):", int((labels_flat == -1).sum()), f"({100 * (labels_flat == -1).mean():.2f}%)")
    print("Weight mean (used only):", float(weights_flat[labels_flat != -1].mean()) if (labels_flat != -1).sum() > 0 else 0.0)
    print("Per-video selected normal count mean:", float(np.mean(normal_counts)))
    print("Per-video selected abnormal count mean:", float(np.mean(abnormal_counts)))
    print("==================================================\n")

    return scores_flat, labels_flat, weights_flat, normal_video_indices






def main():
    print("=" * 80)
    print("Propagation-based Pseudo Labeling")
    print("=" * 80)

    train_nalist_path = r".\list\nalist_i3d.npy"
    train_data_path = r"..\..\C2FPL\concat_UCF.npy"

    nalist = np.load(train_nalist_path)
    total_T = int(nalist[-1, 1])
    lengths = []
    for info in nalist:
        start, end = int(info[0]), int(info[1])
        lengths.append(end - start)

    lengths = np.array(lengths)
    print("min:", lengths.min())
    print("max:", lengths.max())
    print("mean:", lengths.mean())
    print("p95:", np.percentile(lengths, 95))
    print("top10:", np.sort(lengths)[-10:])



    train_data = np.memmap(
        train_data_path,
        dtype="float32",
        mode="r",
        shape=(total_T, 10, 2048)
    )

    scores_flat, labels_flat, weights_flat, normal_video_indices = generate_propagation_pseudo_labels(
        train_data=train_data,
        nalist=nalist,
        prefix_len=5,
        feature_norm="standard",
        sigma_t=5.0,
        alpha=0.9,
        num_iters=50,
        normal_q=0.20,
        abnormal_q=0.05,
    )

    np.save("pseudo_prop5_scorez_glist.npy", scores_flat)
    np.save("pseudo_prop5_label_glist.npy", labels_flat)
    np.save("pseudo_prop5_weight_glist.npy", weights_flat)
    np.save("pseudo_prop5_highconf_video_idx.npy", normal_video_indices)
    '''
    np.savez(
        "pseudo_prop5_meta_glist.npz",
        thr_normal=thr_n,
        thr_abnormal=thr_a,
        prefix_len=5,
        feature_norm="standard",
        sigma_t=5.0,
        alpha=0.9,
        num_iters=50,
        normal_q=0.20,
        abnormal_q=0.05,
    )
    
    print("Saved:")
    print("  pseudo_prop5_score.npy")
    print("  pseudo_prop5_label.npy")
    print("  pseudo_prop5_weight.npy")
    print("  pseudo_prop5_meta.npz")

    '''
if __name__ == "__main__":
    main()