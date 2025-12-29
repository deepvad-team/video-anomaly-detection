import os

# 저자 제공 test list (저자 로컬 경로일 것)
LIST_PATH = r"list\ucf-i3d-test.list"

# 네 로컬 test feature 폴더
FEATURE_DIR = r"C:\Users\jplabuser\Downloads\UCF_Test_ten_i3d\UCF_Test_ten_i3d"

# 출력: 내 로컬 절대경로로 바꾼 test list
OUT_FIXED_LIST = r"list\ucf-i3d_test_fixed_local.list"

# 1) 폴더 파일명 세트
folder_files = sorted([f for f in os.listdir(FEATURE_DIR) if f.endswith(".npy")])
folder_set = set(folder_files)

# 2) list에서 파일명만 추출 후 local 경로로 변환
fixed_paths = []
missing = []
with open(LIST_PATH, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        p = line.split()[0]
        base = os.path.basename(p)
        if not base.endswith(".npy"):
            base = base + ".npy"

        if base not in folder_set:
            missing.append(base)
        fixed_paths.append(os.path.join(FEATURE_DIR, base))

print("folder npy:", len(folder_files))
print("list entries:", len(fixed_paths))
print("missing_in_folder:", len(missing))
if missing:
    print("example missing:", missing[:10])

# 3) fixed list 저장
with open(OUT_FIXED_LIST, "w", encoding="utf-8") as f:
    for p in fixed_paths:
        f.write(p + "\n")

print("Wrote:", OUT_FIXED_LIST)
