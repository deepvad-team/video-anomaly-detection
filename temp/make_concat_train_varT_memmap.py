import os
import numpy as np

# 네가 방금 만든 "로컬 절대경로 list" (1609개, 순서=저자 split)
TRAIN_LOCAL_LIST = r"list\\ucf-i3d_train_fixed_local.list"

# 저자가 준 nalist (1609,2)  sum(T)=779951 -> 1개가 빠져있음.
NALIST_PATH = r"list\\nalist.npy"

# 출력 파일명 유지 (하지만 shape는 var-T concat 형태로 바뀜!)
OUT_MEMMAP = r"concat_UCF.npy"   # (sumT,10,2048)

NUM_CROPS = 10
FEAT_DIM = 2048

def read_list(list_path):
    with open(list_path, "r", encoding="utf-8") as f:
        return [line.strip().split()[0] for line in f if line.strip()]

def main():
    paths = read_list(TRAIN_LOCAL_LIST)
    nalist = np.load(NALIST_PATH)

    assert nalist.ndim == 2 and nalist.shape[1] == 2, nalist.shape
    assert len(paths) == nalist.shape[0], (len(paths), nalist.shape[0])

    total_T = int(nalist[-1, 1])
    print("videos:", len(paths))
    print("expected sum(T) from nalist:", total_T)

    # memmap 생성: (sumT,10,2048)
    fp = np.memmap(OUT_MEMMAP, dtype="float32", mode="w+", shape=(total_T, NUM_CROPS, FEAT_DIM))

    # 안전 체크: 각 비디오 길이가 nalist와 일치해야 함 (순서 검증)
    for i, p in enumerate(paths):
        if not os.path.exists(p):
            raise FileNotFoundError(p)

        a, b = map(int, nalist[i])
        T_expected = b - a

        feat = np.load(p, mmap_mode="r")   # (T,10,2048)
        if feat.ndim != 3 or feat.shape[1] != NUM_CROPS or feat.shape[2] != FEAT_DIM:
            raise ValueError(f"Unexpected shape {feat.shape} for {p}")
        T = feat.shape[0]

        if T != T_expected:
            raise RuntimeError(
                f"ORDER/LENGTH MISMATCH at idx={i}: "
                f"file={os.path.basename(p)}  T={T}  expected={T_expected} (from nalist)"
            )

        # 실제 write (float32)
        fp[a:b, :, :] = np.asarray(feat, dtype=np.float32)

        if i % 200 == 0:
            print(f"write {i}/{len(paths)}  [{a}:{b}]  T={T}  {os.path.basename(p)}")

    fp.flush()
    print("DONE:", OUT_MEMMAP, "shape=", fp.shape)

if __name__ == "__main__":
    main()

'''출력 결과
videos: 1609
expected sum(T) from nalist: 779951
Traceback (most recent call last):
  File "c:\Users\jplabuser\C2FPL\make_concat_train_varT_memmap.py", line 63, in <module>
    main()
    ~~~~^^
  File "c:\Users\jplabuser\C2FPL\make_concat_train_varT_memmap.py", line 48, in main
    raise RuntimeError(
    ...<2 lines>...
    )
RuntimeError: ORDER/LENGTH MISMATCH at idx=0: file=Abuse001_x264_i3d.npy  T=171  expected=34 (from nalist)
'''