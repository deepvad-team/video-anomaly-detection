import os
import re
import numpy as np

"""
FOR UCF CRIME

annotation txt format examples:
Shoplifting037_x264|1386|[1161, 1192]
Shoplifting044_x264|14555|[2075, 2531, 8600, 11298]
Normal_Videos_897_x264 876 -1
"""

root_path = 'C:/Users/jplabuser/Downloads/UCF_Test_ten_i3d/UCF_Test_ten_i3d/'

rgb_list_file = 'ucf-i3d_test_fixed_local.list'
annot_file = "UCF-R-annotations.txt"
output_file = "gt-ucf-R.npy"


def parse_annotation_file(txt_path):
    """
    return:
        ann_dict[video_name] = {
            "num_frames": int,
            "segments": [(start1, end1), (start2, end2), ...]
        }
    """
    ann_dict = {}

    with open(txt_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            # 구분자가 | 인 경우와 공백인 경우 둘 다 처리
            if "|" in line:
                parts = line.split("|")
            else:
                parts = line.split(maxsplit=2)

            if len(parts) < 3:
                print(f"[Skip] malformed line: {line}")
                continue

            video_name = parts[0].strip()
            num_frames = int(parts[1].strip())
            event_str = parts[2].strip()

            # 정상 영상
            if event_str == "-1":
                segments = []
            else:
                nums = list(map(int, re.findall(r"\d+", event_str)))

                if len(nums) % 2 != 0:
                    raise ValueError(f"Odd number of indices in line: {line}")

                segments = []
                for i in range(0, len(nums), 2):
                    start_idx = nums[i]
                    end_idx = nums[i + 1]
                    segments.append((start_idx, end_idx))

            ann_dict[video_name] = {
                "num_frames": num_frames,
                "segments": segments
            }

    return ann_dict


def get_video_name_from_feature_path(path):
    """
    feature path에서 annotation과 맞는 video 이름 추출

    예:
      Shoplifting037_x264_i3d.npy -> Shoplifting037_x264
      Normal_Videos_897_x264.npy -> Normal_Videos_897_x264
    """
    base = os.path.basename(path.strip())
    stem = os.path.splitext(base)[0]

    # xxx_x264_i3d, xxx_x264 등에서 xxx_x264까지만 추출
    m = re.match(r"(.+?_x264)", stem)
    if m:
        return m.group(1)

    return stem


def load_feature(file_path):
    """
    feature npy 로드.
    torch tensor가 섞여 있을 가능성까지 고려.
    """
    features = np.load(file_path, allow_pickle=True)

    features = [
        t.cpu().detach().numpy() if hasattr(t, "cpu") else t
        for t in features
    ]

    features = np.array(features, dtype=np.float32)
    return features


# =========================
# Main
# =========================

ann_dict = parse_annotation_file(annot_file)

with open(rgb_list_file, "r", encoding="utf-8") as f:
    file_list = [line.strip() for line in f if line.strip()]

gt = []

total_segments = 0
total_feature_frames = 0
total_ann_frames = 0

num_padding_like_mismatch = 0
num_unusual_mismatch = 0

for file in file_list:
    # list 안의 경로가 절대경로면 그대로 사용
    # 상대경로면 root_path 기준으로 붙여서 사용
    if os.path.isabs(file):
        feature_path = file
    else:
        feature_path = os.path.join(root_path, file)

    features = load_feature(feature_path)

    # feature shape 예:
    # (T, 10, 2048) 또는 (T, 2048)
    T = features.shape[0]

    # feature 1 segment = 16 frames
    feature_num_frame = T * 16

    video_name = get_video_name_from_feature_path(feature_path)

    if video_name not in ann_dict:
        print(f"[Error] Annotation not found for: {video_name}")
        print(f"feature path: {feature_path}")
        exit(1)

    ann_num_frames = ann_dict[video_name]["num_frames"]
    segments = ann_dict[video_name]["segments"]

    # --------------------------------------------------------
    # 중요:
    # 최종 GT 길이는 반드시 feature 기준 길이에 맞춘다.
    # 왜냐하면 모델 prediction도 T개 segment -> T*16 frame으로 확장되기 때문.
    # --------------------------------------------------------
    gt_video = np.zeros(feature_num_frame, dtype=np.float32)

    # --------------------------------------------------------
    # anomaly 구간은 annotation의 실제 frame 번호 기준으로 표시한다.
    # feature 때문에 뒤에 붙은 padding frame은 그대로 0으로 둔다.
    # --------------------------------------------------------
    for start_idx, end_idx in segments:
        # 만약 annotation이 1-based inclusive라면 아래 두 줄을 활성화해야 함.
        # UCF 계열 annotation은 코드/자료마다 차이가 있을 수 있어서
        # 기존에 쓰던 GT와 비교해서 결정하는 게 가장 안전함.
        #
        # start_idx -= 1
        # end_idx -= 1

        # annotation 실제 프레임 범위 안으로 clipping
        start_idx = max(0, start_idx)
        end_idx = min(end_idx, ann_num_frames - 1)

        if start_idx <= end_idx:
            gt_video[start_idx:end_idx + 1] = 1.0

    # 길이 검증
    if len(gt_video) != feature_num_frame:
        print(f"[Error] GT length mismatch in {video_name}")
        print(f"  expected feature_num_frame = {feature_num_frame}")
        print(f"  gt_video length = {len(gt_video)}")
        exit(1)

    # frame mismatch 확인
    diff = feature_num_frame - ann_num_frames

    if diff != 0:
        if 0 < diff < 16:
            # 정상적인 segment padding mismatch로 보면 됨
            num_padding_like_mismatch += 1
            print(f"[Info] feature padding mismatch: {video_name}")
            print(f"  annotation num_frames = {ann_num_frames}")
            print(f"  feature-derived num_frame = {feature_num_frame}")
            print(f"  padded frames = {diff}")
        else:
            # 이 경우는 조금 주의해서 봐야 함
            num_unusual_mismatch += 1
            print(f"[Warning] unusual frame mismatch: {video_name}")
            print(f"  annotation num_frames = {ann_num_frames}")
            print(f"  feature-derived num_frame = {feature_num_frame}")
            print(f"  diff = {diff}")

    gt.extend(gt_video.tolist())

    total_segments += T
    total_feature_frames += feature_num_frame
    total_ann_frames += ann_num_frames


gt = np.array(gt, dtype=np.float32)

# 최종 검증
expected_gt_length = total_segments * 16

print("=" * 60)
print("GT generation summary")
print("=" * 60)
print("Number of videos:", len(file_list))
print("Total segments:", total_segments)
print("Expected GT length:", expected_gt_length)
print("Actual GT length:", len(gt))
print("Total annotation frames:", total_ann_frames)
print("Total feature-derived frames:", total_feature_frames)
print("Padding-like mismatches:", num_padding_like_mismatch)
print("Unusual mismatches:", num_unusual_mismatch)
print("Output file:", output_file)

assert len(gt) == expected_gt_length, "Final GT length does not match feature-derived length!"

np.save(output_file, gt)

print("=" * 60)
print("Saved GT successfully.")
print("GT positive frames:", int(gt.sum()))
print("GT anomaly ratio:", float(gt.mean()))
print("=" * 60)