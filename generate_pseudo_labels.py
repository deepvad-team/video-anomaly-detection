"""
ETCL Pseudo Label Generator - Standalone Script

Run this FIRST before training!

Usage:
    python generate_pseudo_labels.py --output Unsup_labels/etcl_labels.npz

This generates:
    - labels: segment-level pseudo labels
    - confidence: confidence score for each label (ETCL addition)
"""

import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal
import argparse
import os
import warnings
warnings.filterwarnings("ignore")


def get_matrix(data):
    """Compute L2 norm features (same as C2FPL)"""
    if len(data.shape) == 3:
        l2_norm = np.sum(np.square(data), axis=2)
        l2_mean = np.mean(l2_norm, axis=1)
    else:
        l2_mean = np.sum(np.square(data), axis=1)
    return l2_mean


def estimate_gauss(X):
    """Estimate Gaussian parameters"""
    mu = np.mean(X)
    var = np.var(X) if len(X) > 1 else 1.0
    return mu, var


def generate_etcl_pseudo_labels(
    features_path: str,
    nalist_path: str,
    nu: float = 1.0,
    percentile: float = 3.0,
    window_length: float = 0.2,
    use_gradient: bool = True,
    verbose: bool = True
):
    """
    Generate ETCL pseudo labels with confidence scores
    
    Args:
        features_path: Path to concat_UCF.npy
        nalist_path: Path to nalist.npy
        nu: Target abnormal/normal ratio
        percentile: Bottom percentile for CPL
        window_length: FPL window size (fraction)
        use_gradient: True=gradient-based FPL, False=p-value based
        
    Returns:
        labels: np.ndarray of segment-level labels
        confidence: np.ndarray of confidence scores
    """
    if verbose:
        print("=" * 60)
        print("ETCL Pseudo Label Generation")
        print("=" * 60)
    
    # Load data
    nalist = np.load(r".\list\nalist_i3d.npy")
    total_T = int(nalist[-1, 1])
    train_data = np.memmap(r"..\C2FPL\concat_UCF.npy", dtype="float32", mode="r", shape=(total_T, 10, 2048))
    
    if verbose:
        print(f"\nData shape: {train_data.shape}")
        print(f"Number of videos: {len(nalist)}")
    
    # Split by video
    videos = []
    for fromid, toid in nalist:
        videos.append(train_data[fromid:toid])
    
    # ========== Stage 1: CPL ==========
    if verbose:
        print("\n[Stage 1] Coarse Pseudo Labeling (CPL)")
    
    # Extract video-level features
    params = []
    for video in videos:
        l2_norms = get_matrix(video)
        mu, var = estimate_gauss(l2_norms)
        params.append([mu, var])
    params = np.array(params)
    
    # GMM clustering
    gmm = GaussianMixture(n_components=2, max_iter=150, random_state=42, covariance_type='spherical')
    initial_labels = gmm.fit_predict(params)
    
    # Smaller cluster = abnormal
    cluster_sizes = [np.sum(initial_labels == 0), np.sum(initial_labels == 1)]
    abnormal_cluster = 0 if cluster_sizes[0] < cluster_sizes[1] else 1
    
    # Initialize bags
    abag = [(params[i], i) for i in range(len(params)) if initial_labels[i] == abnormal_cluster]
    nbag = [(params[i], i) for i in range(len(params)) if initial_labels[i] != abnormal_cluster]
    
    if verbose:
        print(f"  Initial: {len(abag)} abnormal, {len(nbag)} normal videos")
    
    # Iterative refinement
    step = 1
    while len(nbag) > 0 and len(abag) / max(len(nbag), 1) < nu:
        temp_params = np.array([x[0] for x in nbag])
        if len(temp_params) < 2:
            break
        
        gmm.fit(temp_params)
        scores = gmm.score_samples(temp_params)
        threshold = np.percentile(scores, percentile)
        is_abnormal = scores < threshold
        
        new_abnormal = [(nbag[i][0], nbag[i][1]) for i in range(len(nbag)) if is_abnormal[i]]
        new_normal = [(nbag[i][0], nbag[i][1]) for i in range(len(nbag)) if not is_abnormal[i]]
        
        abag.extend(new_abnormal)
        nbag = new_normal
        
        if verbose:
            print(f"  Step {step}: added {len(new_abnormal)} abnormal videos")
        step += 1
    
    # Video-level labels
    video_labels = np.zeros(len(videos), dtype=np.float32)
    abnormal_indices = set([x[1] for x in abag])
    for idx in abnormal_indices:
        video_labels[idx] = 1.0
    
    # Video-level confidence
    all_scores = gmm.score_samples(params)
    median_score = np.median(all_scores)
    video_confidence = np.abs(all_scores - median_score)
    video_confidence = (video_confidence - video_confidence.min()) / (video_confidence.max() - video_confidence.min() + 1e-8)
    
    if verbose:
        print(f"  Final: {int(video_labels.sum())} abnormal, {len(videos) - int(video_labels.sum())} normal videos")
    
    # ========== Stage 2: FPL ==========
    if verbose:
        print("\n[Stage 2] Fine Pseudo Labeling (FPL)")
    
    # Build normal distribution
    normal_l2_norms = []
    for i, video in enumerate(videos):
        if video_labels[i] == 0:
            normal_l2_norms.extend(get_matrix(video))
    
    mu_normal, var_normal = estimate_gauss(np.array(normal_l2_norms))
    var_normal = max(var_normal, 1e-6)
    p_normal = multivariate_normal(mu_normal, var_normal)
    
    if verbose:
        print(f"  Normal distribution: mu={mu_normal:.4f}, var={var_normal:.4f}")
    
    # Generate segment-level labels
    all_labels = []
    all_confidence = []
    
    for i, video in enumerate(videos):
        num_segs = len(video)
        
        if video_labels[i] == 0:
            # Normal video
            all_labels.extend([0.0] * num_segs)
            all_confidence.extend([video_confidence[i]] * num_segs)
        else:
            # Abnormal video - localize
            l2_norms = get_matrix(video)
            probs = p_normal.pdf(l2_norms)
            
            seg_labels = [0.0] * num_segs
            seg_confidence = [0.5] * num_segs
            
            window_size = max(3, int(num_segs * window_length))
            window_size = min(window_size, num_segs - 1)
            
            if window_size > 0 and num_segs > window_size:
                if use_gradient:
                    # Gradient-based (C2FPL code style)
                    gradients = []
                    for idx in range(num_segs - window_size + 1):
                        window_probs = probs[idx:idx + window_size]
                        gradient = np.sum(np.abs(np.diff(window_probs)))
                        gradients.append(gradient)
                    
                    if gradients:
                        max_idx = np.argmax(gradients)
                        for j in range(max_idx, min(max_idx + window_size, num_segs)):
                            seg_labels[j] = 1.0
                            seg_confidence[j] = min(1.0, gradients[max_idx] / (np.mean(gradients) + 1e-8))
                else:
                    # P-value based (C2FPL paper style)
                    avg_pvals = []
                    for idx in range(num_segs - window_size + 1):
                        avg_pvals.append(np.mean(probs[idx:idx + window_size]))
                    
                    if avg_pvals:
                        min_idx = np.argmin(avg_pvals)
                        for j in range(min_idx, min(min_idx + window_size, num_segs)):
                            seg_labels[j] = 1.0
                            seg_confidence[j] = max(0.3, (np.mean(avg_pvals) - avg_pvals[min_idx]) / (np.mean(avg_pvals) + 1e-8))
            
            all_labels.extend(seg_labels)
            all_confidence.extend([c * video_confidence[i] for c in seg_confidence])
    
    labels = np.array(all_labels, dtype=np.float32)
    confidence = np.array(all_confidence, dtype=np.float32)
    
    if verbose:
        print("\n" + "=" * 60)
        print("Summary:")
        print(f"  Total segments: {len(labels)}")
        print(f"  Abnormal segments: {int(labels.sum())} ({100*labels.mean():.2f}%)")
        print(f"  Mean confidence: {confidence.mean():.4f}")
        print("=" * 60)
    
    return labels, confidence


def main():
    parser = argparse.ArgumentParser(description='Generate ETCL Pseudo Labels')
    
    parser.add_argument('--features', type=str, default='../C2FPL/concat_UCF.npy',
                        help='Path to concatenated features')
    parser.add_argument('--nalist', type=str, default='list/nalist_i3d.npy',
                        help='Path to nalist.npy')
    parser.add_argument('--output', type=str, default='Unsup_labels/etcl_labels.npz',
                        help='Output path for labels')
    parser.add_argument('--nu', type=float, default=1.0,
                        help='Target abnormal/normal ratio')
    parser.add_argument('--percentile', type=float, default=3.0,
                        help='Bottom percentile for CPL')
    parser.add_argument('--window_length', type=float, default=0.2,
                        help='FPL window size (fraction)')
    parser.add_argument('--use_gradient', action='store_true', default=True,
                        help='Use gradient-based FPL (default: True)')
    parser.add_argument('--use_pvalue', action='store_true',
                        help='Use p-value based FPL instead')
    
    args = parser.parse_args()
    
    use_gradient = not args.use_pvalue
    
    # Generate labels
    labels, confidence = generate_etcl_pseudo_labels(
        features_path=args.features,
        nalist_path=args.nalist,
        nu=args.nu,
        percentile=args.percentile,
        window_length=args.window_length,
        use_gradient=use_gradient
    )
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Save in multiple formats for compatibility
    
    # 1. NPZ format (labels + confidence together) - for ETCL
    np.savez(args.output, labels=labels, confidence=confidence)
    print(f"\nSaved NPZ to: {args.output}")
    
    # 2. NPY format (labels only) - for C2FPL compatibility
    labels_only_path = args.output.replace('.npz', '_labels.npy')
    np.save(labels_only_path, labels)
    print(f"Saved labels to: {labels_only_path}")
    
    # 3. NPY format (confidence only)
    conf_path = args.output.replace('.npz', '_confidence.npy')
    np.save(conf_path, confidence)
    print(f"Saved confidence to: {conf_path}")
    
    print("\nDone! Use these files with main_etcl.py")


if __name__ == '__main__':
    main()
