import numpy as np
import os

segment_len = 16   # 1 segment = 16 frames

# ===== paths =====
feature_list = 'XD_rgb_test_R50NL.list'          # 비디오 순서 파일
gt_txt = r'C:\Users\jplabuser\Downloads\i3d-features\annotations.txt'
new_nalist_path = 'nalist_XD_test_R50NL.npy'            # 새 extractor 기준 nalist
save_path = 'gt-XD-R50NL.npy'


def normalize_name(x):
    x = x.strip()

    # Windows 경로까지 들어와도 basename 추출되게
    x = x.replace("\\", "/")
    x = x.split("/")[-1]

    # .npy / .mp4 제거
    if x.endswith(".npy"):
        x = x[:-4]
    elif x.endswith(".mp4"):
        x = x[:-4]

    # annotation 쪽에 붙어 있는 v= 제거
    if x.startswith("v="):
        x = x[2:]

    return x


def load_video_names(list_path):
    names = []
    with open(list_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            name = normalize_name(line)
            names.append(name)

    return names

def parse_annotations(annotation_path):
    """
    annotations.txt 형식:
    video_name start1 end1 start2 end2 ...
    """
    ann = {}

    with open(annotation_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            vid = normalize_name(parts[0])

            nums = list(map(int, parts[1:]))
            assert len(nums) % 2 == 0, f"Odd number of frame indices: {line}"

            intervals = []
            for i in range(0, len(nums), 2):
                s = nums[i]
                e = nums[i + 1]
                if e < s:
                    s, e = e, s
                intervals.append((s, e))

            ann[vid] = intervals

    return ann


def has_overlap(seg_start, seg_end, intervals):
    for a, b in intervals:
        if not (seg_end < a or seg_start > b):
            return True
    return False


def build_segment_gt(video_names, ann, new_nalist, segment_len=16):
    gt_seg_all = []
    missing = []

    assert len(video_names) == len(new_nalist), \
        f"video_names({len(video_names)}) != new_nalist({len(new_nalist)})"

    for i, vid in enumerate(video_names):
        s, e = new_nalist[i]
        T = int(e - s)

        # annotation이 없으면 normal video로 간주
        intervals = ann.get(vid, [])
        if vid not in ann:
            missing.append(vid)

        gt_vec_seg = np.zeros(T, dtype=np.float32)

        for seg_idx in range(T):
            seg_start = seg_idx * segment_len
            seg_end = (seg_idx + 1) * segment_len - 1

            if has_overlap(seg_start, seg_end, intervals):
                gt_vec_seg[seg_idx] = 1.0

        gt_seg_all.extend(gt_vec_seg.tolist())

    gt_seg_all = np.array(gt_seg_all, dtype=np.float32)

    return gt_seg_all, missing


# ===== main =====
video_names = load_video_names(feature_list)
new_nalist = np.load(new_nalist_path)
ann = parse_annotations(gt_txt)

print('[INFO] num videos in list:', len(video_names))
print('[INFO] num videos in nalist:', len(new_nalist))
print('[INFO] num annotated videos:', len(ann))

gt_seg, missing = build_segment_gt(
    video_names=video_names,
    ann=ann,
    new_nalist=new_nalist,
    segment_len=segment_len
)

# frame-level GT 생성 (1 segment = 16 frames)
gt_frame = np.repeat(gt_seg, segment_len).astype(np.float32)
np.save('gt-XD-R50NL.npy', gt_frame)

print('\n[SAVED]')
print(save_path)
print('gt_seg shape:', gt_seg.shape)
print('gt_seg shape:', gt_frame.shape)

print('num anomalous segments:', int(gt_seg.sum()))
print('num anomalous frames:', int(gt_frame.sum()))

print('num missing annotation names:', len(missing))
if len(missing) > 0:
    print('example missing names:', missing[:10])

total_T = int(new_nalist[-1, 1])
print('total_T from nalist:', total_T)
print('len(gt_seg):', len(gt_seg))
print('match:', total_T == len(gt_seg))