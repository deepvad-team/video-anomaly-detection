# pseudo_labeler.py -> sh train.sh로 모델 학습을 진행하기 전 이 모듈 먼저 돌려 gt 역할을 할 pseudo label을 생성할 것임
# 20260429 현재 진행 중인 코드 개선 과정에서 본 파일의 결과로 저장된 pseudo_labels_swap_90.npy 파일이 pseudo label로 사용되고 있음.  

"""
Improved Pseudo Labeling with Post-processing

새로운 기능:
1. Temporal Attraction (기존) - Abnormal 주변을 high score로
2. Remove Isolated Abnormal - N-A-N → N-N-N
3. Fill Isolated Normal - A-N-A → A-A-A
4. Prototype Swap - 비디오 대부분이 abnormal이면 반전
"""

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.mixture import GaussianMixture
from sklearn.metrics.pairwise import cosine_similarity
from scipy.ndimage import gaussian_filter1d


def find_optimal_threshold_gmm(scores, method='gmm_2component'):
    """Adaptive threshold using GMM"""
    valid_scores = scores[scores > 0]
    
    if len(valid_scores) < 100:
        return np.percentile(valid_scores, 85)
    
    if method == 'gmm_2component':
        gmm = GaussianMixture(n_components=2, random_state=42)
        gmm.fit(valid_scores.reshape(-1, 1))
        
        means = gmm.means_.flatten()
        stds = np.sqrt(gmm.covariances_.flatten())
        
        threshold = (means[0] + means[1]) / 2
        lower_mean_idx = np.argmin(means)
        threshold += 0.5 * stds[lower_mean_idx]
        
        return threshold
    
    return np.percentile(valid_scores, 90)


def temporal_attraction(video_scores, attraction_strength=0.4, iterations=3):
    """
    Step 3: Temporal Attraction
    
    동작: Abnormal (high score) 주변을 끌어올림
    효과: A-N-A 패턴에서 중간 N이 A로 변함 (부분적으로)
    
    Args:
        video_scores: (T,) distance scores
        attraction_strength: 끌어당기는 힘
        iterations: 반복 횟수
    
    Returns:
        attracted_scores: (T,) attracted scores
    """
    scores = video_scores.copy()
    
    for _ in range(iterations):
        attracted = scores.copy()
        
        for i in range(len(scores)):
            if scores[i] == 0:  # Prototype는 skip
                continue
            
            # Window (i-5 ~ i+5)
            window_start = max(0, i - 5)
            window_end = min(len(scores), i + 6)
            window = scores[window_start:window_end]
            
            # 주변에 더 높은 score 있으면 끌어올림
            if window.max() > scores[i]:
                max_idx = window.argmax() + window_start
                distance = abs(max_idx - i)
                
                # Exponential decay
                force = attraction_strength * window.max() * np.exp(-distance / 2.0)
                attracted[i] += force
        
        # Renormalize
        valid_mask = attracted > 0
        if valid_mask.sum() > 0:
            valid = attracted[valid_mask]
            if valid.max() > valid.min():
                attracted[valid_mask] = (valid - valid.min()) / (valid.max() - valid.min())
        
        scores = attracted
    
    return scores


def remove_isolated_abnormal(binary_labels, min_length=1):
    """
    Post-processing 1: Remove Isolated Abnormal
    
    패턴: N-A-N → N-N-N
    
    이유: Abnormal events는 연속적이어야 함
         1개만 튀는 것은 noise일 가능성
    
    Args:
        binary_labels: (T,) {0, 1}
        min_length: Abnormal이 최소 이 길이는 되어야 유지
    
    Returns:
        cleaned: (T,) {0, 1}
    """
    cleaned = binary_labels.copy()
    T = len(binary_labels)
    
    i = 0
    while i < T:
        if cleaned[i] == 1:  # Abnormal 발견
            # Abnormal run의 길이 측정
            run_start = i
            run_end = i
            
            while run_end < T and cleaned[run_end] == 1:
                run_end += 1
            
            run_length = run_end - run_start
            
            # 너무 짧은 abnormal run → Normal로 변경
            if run_length <= min_length:
                cleaned[run_start:run_end] = 0
            
            i = run_end
        else:
            i += 1
    
    return cleaned


def fill_isolated_normal(binary_labels, max_gap=1):
    """
    Post-processing 2: Fill Isolated Normal
    
    패턴: A-N-A → A-A-A
    
    이유: Abnormal event 중간에 1개만 Normal은 이상함
         연속성 유지
    
    Args:
        binary_labels: (T,) {0, 1}
        max_gap: Normal gap이 최대 이 길이까지만 채우기
    
    Returns:
        filled: (T,) {0, 1}
    """
    filled = binary_labels.copy()
    T = len(binary_labels)
    
    i = 0
    while i < T:
        if filled[i] == 1:  # Abnormal 발견
            # 다음 Abnormal까지의 gap 찾기
            j = i + 1
            
            # Normal gap 건너뛰기
            while j < T and filled[j] == 0:
                j += 1
            
            gap_length = j - i - 1
            
            # Gap이 작고, 다음도 Abnormal이면 채우기
            if j < T and filled[j] == 1 and gap_length <= max_gap:
                filled[i+1:j] = 1
                i = j
            else:
                i += 1
        else:
            i += 1
    
    return filled


def check_prototype_swap(video_binary_labels, abnormal_ratio_threshold=0.8):
    """
    Prototype Swap 필요 여부 체크
    
    문제: 앞 5개로 prototype 만들었는데,
         실제로 비디오의 80-90%가 abnormal이면?
         → 앞 5개가 abnormal일 가능성!
         → Prototype이 abnormal을 가리킴
         → Normal/Abnormal 반전!
    
    해결: 비디오의 abnormal 비율이 너무 높으면
         Normal ↔ Abnormal 반전
    
    Args:
        video_binary_labels: (T,) {0, 1}
        abnormal_ratio_threshold: 이 비율 이상이면 swap
    
    Returns:
        should_swap: bool
        abnormal_ratio: float
    """
    abnormal_ratio = video_binary_labels.mean()
    should_swap = abnormal_ratio >= abnormal_ratio_threshold
    
    return should_swap, abnormal_ratio


def apply_prototype_swap(video_binary_labels):
    """
    Prototype Swap 적용
    
    0 ↔ 1 반전
    
    Args:
        video_binary_labels: (T,) {0, 1}
    
    Returns:
        swapped: (T,) {0, 1}
    """
    return 1 - video_binary_labels


def generate_improved_pseudo_labels(train_data, nalist,
                                    feature_normalization='standard',
                                    threshold_method='gmm_2component',
                                    score_normalization='zscore',
                                    prototype_method='median_based',
                                    prefix_len=5,   # 추가 
                                    use_attraction=True,
                                    attraction_strength=0.4,
                                    attraction_iterations=3,
                                    remove_isolated_abn=True,
                                    isolated_abn_min_length=1,
                                    fill_isolated_norm=True,
                                    isolated_norm_max_gap=2,
                                    use_prototype_swap=True,
                                    swap_threshold=0.8):
    """
    개선된 Pseudo Label 생성
    
    새로운 기능:
    1. Remove isolated abnormal (N-A-N → N-N-N)
    2. Fill isolated normal (A-N-A → A-A-A)
    3. Prototype swap (비디오 대부분 abnormal → 반전)
    """
    print("\n" + "="*80)
    print("Improved Pseudo Labeling with Post-processing")
    print("="*80)
    
    print(f"\nConfiguration:")
    print(f"  Feature normalization: {feature_normalization}")
    print(f"  Threshold method: {threshold_method}")
    print(f"  Score normalization: {score_normalization}")
    print(f"  Prototype method: {prototype_method}")
    print(f"  Use attraction: {use_attraction}")
    print(f"  Remove isolated abnormal: {remove_isolated_abn} (min_length={isolated_abn_min_length})")
    print(f"  Fill isolated normal: {fill_isolated_norm} (max_gap={isolated_norm_max_gap})")
    print(f"  Use prototype swap: {use_prototype_swap} (threshold={swap_threshold})")
    
    total_T = int(nalist[-1, 1])
    
    # Step 0: Feature Normalization
    print(f"\n[Step 0: Feature Normalization]")
    all_features = []
    
    for info in tqdm(nalist, desc="Loading"):
        start, end = int(info[0]), int(info[1])
        #video_feat = train_data[start:end].mean(axis=1)
        video_feat = np.mean(train_data[start:end], axis=1)
        all_features.append(video_feat)
    
    # Normalize per video
    if feature_normalization != 'none':
        print(f"  Normalizing ({feature_normalization})...")
        for i, video_feat in enumerate(tqdm(all_features, desc="Normalizing")):
            if feature_normalization == 'standard':
                mean = video_feat.mean(axis=0)
                std = video_feat.std(axis=0) + 1e-8
                all_features[i] = (video_feat - mean) / std
            elif feature_normalization == 'l2':
                norms = np.linalg.norm(video_feat, axis=1, keepdims=True) + 1e-8
                all_features[i] = video_feat / norms
    
    # Step 1: Compute Distance Scores
    print(f"\n[Step 1: Computing Distance Scores]")
    all_scores = []
    
    for video_feat in tqdm(all_features, desc="Scoring"):


        if len(video_feat) < max(prefix_len + 3, 8):
            all_scores.append(np.zeros(len(video_feat)))
            continue
            
        # Prototype
        if prototype_method == 'median_based':

            median_feat = np.median(video_feat, axis=0)
            distances_to_median = np.linalg.norm(video_feat - median_feat, axis=1)
            top_k = min(5, len(video_feat) // 2)
            proto_indices = np.argsort(distances_to_median)[:top_k]
            prototype = video_feat[proto_indices].mean(axis=0)


        elif prototype_method == 'prefix_median':

            prototype = np.median(video_feat[:prefix_len], axis=0)
            proto_indices = np.arange(prefix_len)


        elif prototype_method == 'prefix_mean':

            prototype = video_feat[:prefix_len].mean(axis=0)
            proto_indices = np.arange(prefix_len)

        else:
            raise ValueError(f"Unknown prototype_method: {prototype_method}")

            '''for _ in range(3):
                dist = np.linalg.norm(video_feat - prototype, axis=1)
                idx = np.argsort(dist)[:5]
                prototype = video_feat[idx].mean(axis=0)'''

            #proto_indices = range(5)
        
        # Distance
        #video_feat = video_feat / np.linalg.norm(video_feat, axis=1, keepdims=True)
        distances = np.linalg.norm(video_feat - prototype, axis=1)
        #distances = np.sum(np.abs(video_feat - prototype), axis=1)
        distances[proto_indices] = 0
        #sim = cosine_similarity(video_feat, prototype.reshape(1,-1)).squeeze()
        #scores = 1 - sim
        #scores[proto_indices] = 0
        
        all_scores.append(distances)
        #all_scores.append(scores)
    
    # Step 2: Score Normalization
    print(f"\n[Step 2: Score Normalization ({score_normalization})]")
    
    if score_normalization != 'none':
        for i, video_scores in enumerate(tqdm(all_scores, desc="Normalizing scores")):
            valid_mask = video_scores > 0
            
            if valid_mask.sum() < 2:
                continue
            
            valid_scores = video_scores[valid_mask]
            
            if score_normalization == 'zscore':
                mean = valid_scores.mean()
                std = valid_scores.std()
                if std > 1e-6:
                    normalized = np.zeros_like(video_scores)
                    normalized[valid_mask] = (valid_scores - mean) / std
                    all_scores[i] = normalized

            
            elif score_normalization == 'median':
                median = np.median(valid_scores)
                if median > 1e-6:
                    normalized = np.zeros_like(video_scores)
                    normalized[valid_mask] = valid_scores / median
                    all_scores[i] = normalized
    
    # Step 3: Temporal Attraction
    if use_attraction:
        print(f"\n[Step 3: Temporal Attraction]")
        for i, video_scores in enumerate(tqdm(all_scores, desc="Attraction")):
            if len(video_scores) >= 3:
                all_scores[i] = temporal_attraction(
                    video_scores,
                    attraction_strength=attraction_strength,
                    iterations=attraction_iterations
                )

            from scipy.ndimage import gaussian_filter1d
            all_scores[i] = gaussian_filter1d(all_scores[i], sigma=2.0)
    
    # Step 4: Adaptive Threshold
    print(f"\n[Step 4: Finding Optimal Threshold ({threshold_method})]")
    
    all_scores_flat = np.concatenate(all_scores)
    threshold = find_optimal_threshold_gmm(all_scores_flat, method=threshold_method)
    
    print(f"  Optimal threshold: {threshold:.4f}")
    
    # Step 5: Binary Labels
    print(f"\n[Step 5: Generating Binary Labels]")
    
    all_binary_labels = []
    
    for video_scores in all_scores:
        binary = (video_scores >= threshold).astype(int)
        all_binary_labels.append(binary)
    
    # Step 6: Post-processing
    print(f"\n[Step 6: Post-processing]")
    
    # 6a: Remove Isolated Abnormal
    if remove_isolated_abn:
        print(f"  Removing isolated abnormal (min_length={isolated_abn_min_length})...")
        for i, binary in enumerate(tqdm(all_binary_labels, desc="Remove isolated abn")):
            all_binary_labels[i] = remove_isolated_abnormal(binary, min_length=isolated_abn_min_length)
    
    # 6b: Fill Isolated Normal
    if fill_isolated_norm:
        print(f"  Filling isolated normal (max_gap={isolated_norm_max_gap})...")
        for i, binary in enumerate(tqdm(all_binary_labels, desc="Fill isolated norm")):
            all_binary_labels[i] = fill_isolated_normal(binary, max_gap=isolated_norm_max_gap)
    
    # 6c: Prototype Swap
    swap_count = 0
    swap_ratios = []
    
    if use_prototype_swap:
        print(f"  Checking prototype swap (threshold={swap_threshold})...")
        for i, binary in enumerate(tqdm(all_binary_labels, desc="Prototype swap")):
            should_swap, abn_ratio = check_prototype_swap(binary, abnormal_ratio_threshold=swap_threshold)
            
            if should_swap:
                all_binary_labels[i] = apply_prototype_swap(binary)
                swap_count += 1
                swap_ratios.append(abn_ratio)
        
        print(f"    Swapped: {swap_count}/{len(all_binary_labels)} videos ({100*swap_count/len(all_binary_labels):.1f}%)")
        if swap_count > 0:
            print(f"    Swap ratios: mean={np.mean(swap_ratios):.3f}, min={np.min(swap_ratios):.3f}, max={np.max(swap_ratios):.3f}")
    
    # Statistics
    print(f"\n{'='*80}")
    print("Statistics")
    print(f"{'='*80}")
    
    all_binary_flat = np.concatenate(all_binary_labels)
    
    print(f"\nBinary Labels:")
    print(f"  Normal (0):   {(all_binary_flat == 0).sum():,} ({100*(all_binary_flat==0).mean():.1f}%)")
    print(f"  Abnormal (1): {(all_binary_flat == 1).sum():,} ({100*(all_binary_flat==1).mean():.1f}%)")
    
    # Per-video analysis
    video_abn_ratios = [binary.mean() for binary in all_binary_labels]
    video_abn_ratios = np.array(video_abn_ratios)
    
    print(f"\nPer-Video Analysis:")
    print(f"  Abnormal ratio: {video_abn_ratios.mean():.3f} ± {video_abn_ratios.std():.3f}")
    print(f"  Min: {video_abn_ratios.min():.3f}, Max: {video_abn_ratios.max():.3f}")
    
    print(f"{'='*80}")
    
    return all_binary_labels


def main():
    """Main"""
    print("="*80)
    print("Improved Pseudo Labeling")
    print("="*80)
    
    # Load data
    print("\n[Loading Data]")
    train_nalist_path = r".\list\nalist_i3d.npy"
    train_data_path = r"..\..\C2FPL\concat_UCF.npy"
    
    nalist = np.load(train_nalist_path)
    total_T = int(nalist[-1, 1])
    
    train_data = np.memmap(
        train_data_path,
        dtype="float32",
        mode="r",
        shape=(total_T, 10, 2048)
    )
    
    print(f"  Segments: {total_T:,}")
    print(f"  Videos: {len(nalist)}")
    
    # Generate
    pseudo_labels_list = generate_improved_pseudo_labels(
        train_data, nalist,
        feature_normalization='standard',
        threshold_method='none',
        score_normalization='none',
        prototype_method='prefix_median',
        prefix_len=5,
        use_attraction=True,
        attraction_strength=0.3,
        attraction_iterations=3,
        remove_isolated_abn=True,  # ⭐ NEW
        isolated_abn_min_length=1,  # N-A-N 제거
        fill_isolated_norm=True,  # ⭐ NEW
        isolated_norm_max_gap=1,  # A-N-N-A도 채우기
        use_prototype_swap=True,  # ⭐ NEW
        swap_threshold=0.8  # 80% 이상이면 swap
    )
    
    # Save
    print(f"\n[Saving Results]")
    
    # Flat array로 저장 (no pickle)
    all_labels_flat = np.concatenate(pseudo_labels_list)
    np.save("pseudo_labels_prefix5_swap.npy", all_labels_flat)
    
    print(f"  Saved: pseudo_labels_prefix5_swap.npy")
    
    print("\n" + "="*80)
    print("Complete! ✨")
    print("="*80)
    
    print(f"\n✨ Improvements:")
    print(f"  ✅ Temporal attraction (A-N-A → A-A-A partially)")
    print(f"  ✅ Remove isolated abnormal (N-A-N → N-N-N)")
    print(f"  ✅ Fill isolated normal (A-N-A → A-A-A fully)")
    print(f"  ✅ Prototype swap (비디오 대부분 abnormal → 반전)")
    
    print(f"\n📊 Usage:")
    print(f"  labels = np.load('pseudo_labels_swap.npy')")
    print(f"  nalist = np.load('nalist_i3d.npy')")
    print(f"  ")
    print(f"  start, end = int(nalist[0, 0]), int(nalist[0, 1])")
    print(f"  video_0_labels = labels[start:end]")


if __name__ == "__main__":
    main()