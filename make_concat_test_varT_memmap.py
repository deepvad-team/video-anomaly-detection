import os
import numpy as np

TEST_LOCAL_LIST = r"list\\ucf-i3d_test_fixed_local.list"

OUT_NALIST = r"list\\nalist_test_i3d.npy"
OUT_TLEN   = r"list\\tlen_test_i3d.npy"
OUT_MEMMAP = r"Concat_test_10.npy"   # test용 var-T concat

NUM_CROPS = 10
FEAT_DIM = 2048

def read_list(list_path):
    with open(list_path, "r", encoding="utf-8") as f:
        return [line.strip().split()[0] for line in f if line.strip()]

def main():
    paths = read_list(TEST_LOCAL_LIST)
    n = len(paths)
    print("test videos:", n)

    # 1) tlen 스캔
    tlen = np.zeros(n, dtype=np.int32)
    for i, p in enumerate(paths):
        if not os.path.exists(p):
            raise FileNotFoundError(p)
        feat = np.load(p, mmap_mode="r")
        if feat.ndim != 3 or feat.shape[1] != NUM_CROPS or feat.shape[2] != FEAT_DIM:
            raise ValueError((p, feat.shape))
        tlen[i] = feat.shape[0]
        if i % 50 == 0:
            print(f"scan {i}/{n} T={tlen[i]} file={os.path.basename(p)}")

    # 2) nalist 생성 [from,to)
    nalist = np.zeros((n, 2), dtype=np.int64)
    start = 0
    for i in range(n):
        nalist[i,0] = start
        nalist[i,1] = start + int(tlen[i])
        start = int(nalist[i,1])

    total_T = int(nalist[-1,1])
    print("sum(T_test):", total_T)

    np.save(OUT_NALIST, nalist)
    np.save(OUT_TLEN, tlen)
    print("saved:", OUT_NALIST, OUT_TLEN)

    # 3) memmap write (sumT,10,2048)
    fp = np.memmap(OUT_MEMMAP, dtype="float32", mode="w+", shape=(total_T, NUM_CROPS, FEAT_DIM))
    for i, p in enumerate(paths):
        a, b = map(int, nalist[i])
        feat = np.load(p).astype(np.float32)
        fp[a:b, :, :] = feat
        if i % 50 == 0:
            print(f"write {i}/{n} [{a}:{b}] file={os.path.basename(p)}")

    fp.flush()
    print("DONE:", OUT_MEMMAP, "shape=", fp.shape)

if __name__ == "__main__":
    main()
