import numpy as np
import os
import re
import glob
import numpy as np
from scipy.io import loadmat
from os import walk

'''
FOR UCF CRIME

annotation txt format examples:
Shoplifting037_x264|1386|[1161, 1192]
Shoplifting044_x264|14555|[2075, 2531, 8600, 11298]
Normal_Videos_897_x264 876 -1

'''
root_path = 'C:/Users/jplabuser/Downloads/UCF_Test_ten_i3d/UCF_Test_ten_i3d/'
#dirs = os.listdir(root_path)
rgb_list_file ='ucf-i3d_test_fixed_local.list'
#temporal_root = '/list/Matlab_formate/'
annot_file = "C:/Users/jplabuser/Downloads/UCF-R-annotations.txt"
#mat_name_list = os.listdir(temporal_root)
output_file = "gt-ucf.npy"

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
            # 예:
            # Shoplifting037_x264|1386|[1161, 1192]
            # Normal_Videos_897_x264 876 -1
            if "|" in line:
                parts = line.split("|")
            else:
                parts = line.split(maxsplit=2)

            if len(parts) < 3:
                print(f"skip malformed line: {line}")
                continue

            video_name = parts[0].strip()
            num_frames = int(parts[1].strip())
            event_str = parts[2].strip()

            # 정상 영상
            if event_str == "-1":
                segments = []
            else:
                # [1161, 1192] 또는 [2075, 2531, 8600, 11298] 등에서 숫자 추출
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
      /home/.../Shoplifting037_x264_i3d.npy -> Shoplifting037_x264
      /home/.../Normal_Videos_897_x264.npy -> Normal_Videos_897_x264
    """
    base = os.path.basename(path.strip())

    # 확장자 제거
    stem = os.path.splitext(base)[0]

    # 필요하면 뒤쪽 suffix 제거
    # 예: xxx_x264_i3d -> xxx_x264
    m = re.match(r"(.+?_x264)", stem)
    if m:
        return m.group(1)

    return stem



ann_dict = parse_annotation_file(annot_file)

file_list = list(open(rgb_list_file))
gt = []

for file in file_list:
    file = file.strip()
    features = np.load(file, allow_pickle=True)

    # 네 feature 저장 형식 유지
    features = [t.cpu().detach().numpy() if hasattr(t, "cpu") else t for t in features]
    features = np.array(features, dtype=np.float32)

    # segment feature 1개가 16프레임 담당
    num_frame = features.shape[0] * 16

    video_name = get_video_name_from_feature_path(file)

    if video_name not in ann_dict:
        print(f"Annotation not found for: {video_name}")
        exit(1)

    ann_num_frames = ann_dict[video_name]["num_frames"]
    segments = ann_dict[video_name]["segments"]

    # 기본값: 전부 정상
    gt_video = np.zeros(num_frame, dtype=float)

    # 이상 구간 채우기
    for start_idx, end_idx in segments:
        # ----------------------------------------
        # 중요:
        # annotation이 1-based inclusive면 아래처럼 -1 해줘야 함
        # start_idx -= 1
        # end_idx   -= 1
        #
        # annotation이 이미 0-based면 그대로 둬야 함
        # ----------------------------------------

        # 범위 보정
        start_idx = max(0, start_idx)
        end_idx = min(end_idx, num_frame - 1)

        if start_idx <= end_idx:
            gt_video[start_idx:end_idx + 1] = 1.0

    # 길이 체크
    if len(gt_video) != num_frame:
        print(file)
        print("Num of frames is not correct!!")
        print("feature-derived num_frame:", num_frame)
        print("gt_video length:", len(gt_video))
        exit(1)

    # annotation 파일의 num_frames와 feature-derived num_frame가 다를 수 있으니 확인용 출력
    if ann_num_frames != num_frame:
        print(f"[Warning] frame mismatch: {video_name}")
        print(f"  annotation num_frames = {ann_num_frames}")
        print(f"  feature-derived num_frame = {num_frame}")

    gt.extend(gt_video.tolist())

gt = np.array(gt, dtype=float)
np.save(output_file, gt)
print("Total gt length:", len(gt))