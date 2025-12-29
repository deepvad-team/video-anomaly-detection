import numpy as np

nalist = np.load("list/nalist.npy", allow_pickle=True)
print(type(nalist), nalist.shape, nalist.dtype)
print("first 5:", nalist[:5])
print("last 5:", nalist[-5:])

lens = nalist[:,1] - nalist[:,0]
print("num videos:", len(lens))
print("sum(T):", int(lens.sum()))
print("min/mean/max T:", int(lens.min()), float(lens.mean()), int(lens.max()))
