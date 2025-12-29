import numpy as np

nalist = np.load("list/nalist.npy")
lens = (nalist[:,1] - nalist[:,0]).astype(int)

paths = [line.strip() for line in open(r"list\\ucf-i3d_train_fixed_local.list", "r", encoding="utf-8") if line.strip()]
assert len(paths) == len(lens)

ratios = []
for i in range(200):  # 200개면 충분
    feat = np.load(paths[i], mmap_mode="r")  # (T,10,2048)
    T = feat.shape[0]
    L = lens[i]
    ratios.append(T / L)

ratios = np.array(ratios, dtype=float)
print("checked:", len(ratios))
print("ratio stats min/median/mean/max:", ratios.min(), np.median(ratios), ratios.mean(), ratios.max())
print("first 20 ratios:", ratios[:20])
