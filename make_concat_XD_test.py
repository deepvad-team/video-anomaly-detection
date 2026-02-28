import os
import numpy as np

# ====== 너 환경에 맞게 경로만 확인 ======
XD_LIST_PATH = r"list\XD_rgb_test.list"   # 논문 make_list가 만든 list (0~4 포함)
XD_FEAT_DIR  = r"C:\Users\jplabuser\Downloads\i3d-features\RGBTest"

OUT_MEMMAP = r"concat_XD_test.npy"
OUT_NALIST = r"list\nalist_XD_test.npy"
OUT_TLEN   = r"list\tlen_XD_test.npy"

NUM_CROPS = 5
FEAT_DIM  = 1024

def read_center_paths(list_path):
    """
    list에 crop 0~4가 다 들어있어도,
    _0.npy만 대표로 골라서 비디오당 1개로 만든다.
    """
    paths = []
    with open(list_path, "r", encoding="utf-8") as f:
        for line in f:
            p = line.strip().split()[0]
            if not p:
                continue
            # center crop only
            if p.endswith("_0.npy"):
                paths.append(p)
    return paths

def load_5crop_mean(path0):
    """
    path0: ..._0.npy (center)
    -> base를 만들고 ..._0.npy ~ ..._4.npy를 로드해 평균 -> (T,1024)
    """
    base = path0[:-6]  # remove "_0.npy" (6 chars)
    feats = []
    for c in range(NUM_CROPS):
        p = f"{base}_{c}.npy"
        if not os.path.isabs(p):
            # list에 상대경로가 들어있을 수도 있으니 feat_dir로 보정
            p = os.path.join(XD_FEAT_DIR, os.path.basename(p))
        if not os.path.exists(p):
            raise FileNotFoundError(p)

        x = np.load(p, mmap_mode="r")
        # (T,1024) 또는 (T,1,1024) 같은 경우 대응
        if x.ndim == 3:
            x = x.mean(axis=1)
        if x.ndim != 2 or x.shape[1] != FEAT_DIM:
            raise ValueError((p, x.shape))
        feats.append(np.array(x, dtype=np.float32))

    # (T,5,1024) -> mean over crops -> (T,1024)
    return np.stack(feats, axis=1).mean(axis=1)

def main():
    center_paths = read_center_paths(XD_LIST_PATH)
    n = len(center_paths)
    print("videos (center only):", n)
    if n == 0:
        raise RuntimeError("No *_0.npy found in list. Check XD_LIST_PATH.")

    # 1) tlen scan
    tlen = np.zeros(n, dtype=np.int32)
    for i, p0 in enumerate(center_paths):
        feat = load_5crop_mean(p0)
        tlen[i] = feat.shape[0]
        if i % 200 == 0:
            print(f"scan {i}/{n} T={tlen[i]} file={os.path.basename(p0)}")

    # 2) nalist
    nalist = np.zeros((n, 2), dtype=np.int64)
    start = 0
    for i in range(n):
        nalist[i, 0] = start
        nalist[i, 1] = start + int(tlen[i])
        start = int(nalist[i, 1])
    total_T = int(nalist[-1, 1])
    print("sum(T):", total_T)

    os.makedirs(os.path.dirname(OUT_NALIST), exist_ok=True)
    np.save(OUT_NALIST, nalist)
    np.save(OUT_TLEN, tlen)
    print("saved:", OUT_NALIST, OUT_TLEN)

    # 3) write memmap (sumT,1024)
    fp = np.memmap(OUT_MEMMAP, dtype="float32", mode="w+", shape=(total_T, FEAT_DIM))
    for i, p0 in enumerate(center_paths):
        a, b = map(int, nalist[i])
        feat = load_5crop_mean(p0)  # (T,1024)
        fp[a:b, :] = feat
        if i % 200 == 0:
            print(f"write {i}/{n} [{a}:{b}] file={os.path.basename(p0)}")

    fp.flush()
    print("DONE:", OUT_MEMMAP, "shape=", fp.shape)

if __name__ == "__main__":
    main()