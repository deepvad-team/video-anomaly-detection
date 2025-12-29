import os, numpy as np

feature_dir = r"C:\\Users\\jplabuser\\Downloads\\UCF_Test_ten_i3d\\UCF_Test_ten_i3d"
video_files = sorted([f for f in os.listdir(feature_dir) if f.endswith(".npy")])

Ts = []
for f in video_files:
    feat = np.load(os.path.join(feature_dir, f), mmap_mode="r")
    Ts.append(feat.shape[0])   # T

gt = np.load("list/gt-ucf-RTFM.npy", allow_pickle=True)

print("num videos:", len(Ts))
print("sum(T):", sum(Ts))
print("sum(T*16):", sum(t*16 for t in Ts))
print("len(gt):", len(gt))