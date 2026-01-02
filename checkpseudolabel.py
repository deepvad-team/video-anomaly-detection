import numpy as np,math

# 1) 경로 설정
pseudofile = "Unsup_labels/UCF_unsup_labels_original_V2.npy"          # 예: "pseudo_labels.npy"
nalist_path = "list/nalist_i3d.npy"   # 너희가 쓰는 경로로

# 2) 로드
y = np.load(pseudofile).astype(np.float32)      # (sumT,)
nalist = np.load(nalist_path).astype(np.int64)  # (N,2) each: [from,to]
print("y unique (sample):", np.unique(y[:100000])[:10], " ...")
print("overall positive ratio:", float((y > 0.5).mean()))

beta = 0.2

def count_runs_of_ones(arr, thr=0.5):
    b = (arr > thr).astype(np.int32)
    if b.sum() == 0:
        return 0, 0
    starts = np.where((b[1:] == 1) & (b[:-1] == 0))[0] + 1
    if b[0] == 1:
        starts = np.concatenate(([0], starts))
    ends = np.where((b[1:] == 0) & (b[:-1] == 1))[0] + 1
    if b[-1] == 1:
        ends = np.concatenate((ends, [len(b)]))
    run_lengths = ends - starts
    return len(run_lengths), int(run_lengths.max())

pos_rows = []
for vid, (frm, to) in enumerate(nalist):
    seg = y[frm:to]
    m = len(seg)
    if m == 0:
        continue
    if (seg > 0.5).sum() == 0:
        continue

    ones_ratio = float((seg > 0.5).mean())
    expected_w = int(math.ceil(beta * m))
    expected_ratio = expected_w / m
    n_runs, max_run = count_runs_of_ones(seg, thr=0.5)

    pos_rows.append((vid, m, ones_ratio, expected_ratio, n_runs, max_run, expected_w))

print("\n#videos with any 1s:", len(pos_rows), "out of", len(nalist))

# (A) 1이 있는 비디오들에서 ratio가 0.2 근처인지
pos_rows_sorted = sorted(pos_rows, key=lambda x: abs(x[2]-x[3]), reverse=True)
print("\n[Top 20] 1이 있는 비디오 중 ones_ratio가 expected(=ceil(0.2m)/m)에서 많이 벗어난 비디오")
for vid, m, r, er, n_runs, max_run, ew in pos_rows_sorted[:20]:
    print(f"vid={vid:4d} m={m:3d} ones_ratio={r:.3f} expected={er:.3f} runs={n_runs} max_run={max_run} expected_w={ew}")

# (B) 연속 한 덩어리인지
bad_runs = [x for x in pos_rows if x[4] != 1]
print("\n[Top 20] runs != 1 (연속 한 덩어리 위반)")
for vid, m, r, er, n_runs, max_run, ew in bad_runs[:20]:
    print(f"vid={vid:4d} m={m:3d} ones_ratio={r:.3f} expected={er:.3f} runs={n_runs} max_run={max_run} expected_w={ew}")