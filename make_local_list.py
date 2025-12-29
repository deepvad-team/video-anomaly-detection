#Download 디렉토리에 있는 훈련 피처 1610개 중 nalist 개수 1609개에 맞는 1개 누락 파일을 제거하여 list 디렉토리 아래에 ucf-i3d_train_fixed_local.list로 저장

import os
import numpy as np

# 로컬 train feature 폴더
FEATURE_DIR = r"C:\\Users\\jplabuser\\Downloads\\UCF_train_feature\\UCF_Train_ten_crop_i3d"

# 저자 제공 list
LIST_PATH = r"list\\ucf-i3d.list"

# nalist (훈련 비디오 경계) - 1609개여야 정상
NALIST_PATH = r"list\\nalist.npy"

#로컬 경로에 맞게 저장될 최종 list
OUT_FIXED_LIST = r"list\\ucf-i3d_train_fixed_local.list"

# 1) 피처 폴더 내 파일 목록(파일명만)
folder_files = sorted([f for f in os.listdir(FEATURE_DIR) if f.endswith(".npy")])
folder_set = set(folder_files)

# 2) ucf-i3d.list에서 파일명만 뽑기
list_files = []
with open(LIST_PATH, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        p = line.split()[0]  # "path label"이면 첫 토큰이 경로
        base = os.path.basename(p)
        # 어떤 list는 확장자가 없을 수도 있어서 보정
        if not base.endswith(".npy"):
            base = base + ".npy"
        list_files.append(base)

print("folder npy:", len(folder_files))
print("list entries:", len(list_files))

# 3) list -> local 파일 매칭 (1개 누락된 것 찾기)
missing_in_folder = [x for x in list_files if x not in folder_set]
extra_in_folder = [x for x in folder_files if x not in set(list_files)]

print("missing_in_folder:", len(missing_in_folder))
print("extra_in_folder:", len(extra_in_folder))
if missing_in_folder:
    print("example missing:", missing_in_folder[:10])
if extra_in_folder:
    print("example extra:", extra_in_folder[:10])

# 4) nalist 길이 확인 (list와 맞춰야 함)
nalist = np.load(NALIST_PATH)
print("nalist videos:", nalist.shape[0])

# 5) “재현용 fixed list” 만들기: nalist(1609)와 길이를 맞추는 게 중요
# - list_files가 1609개가 아니면, 여기서부터는 list 파일 자체가 train split과 다를 수 있음
if len(list_files) != nalist.shape[0]:
    print("WARNING: list 길이와 nalist 비디오 수가 다릅니다.")
    print("list:", len(list_files), "nalist:", nalist.shape[0])
    # 그래도 우선은 'nalist 수만큼 앞에서부터' 쓰는 방식은 위험할 수 있음.
    # 차라리 list/ 폴더에 train split용 다른 list가 있는지 찾아보는 게 안전.
else:
    with open(OUT_FIXED_LIST, "w", encoding="utf-8") as f:
        for base in list_files:
            local_path = os.path.join(FEATURE_DIR, base)
            f.write(local_path + "\n")
    print("Wrote fixed local list:", OUT_FIXED_LIST)
