import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import auc

# npy 불러오기
fpr = np.load("fpr_UCF.npy")
tpr = np.load("tpr_UCF.npy")
#thresholds = np.load("roc_threshold.npy")
roc_auc = auc(fpr, tpr)
print("ROC AUC:", roc_auc)

print("fpr shape:", fpr.shape)
print("tpr shape:", tpr.shape)
#print("thresholds shape:", thresholds.shape)

# 길이 확인
assert len(fpr) == len(tpr) #== len(thresholds), "길이가 서로 다릅니다."

# ROC 곡선 그리기
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.4f})")
plt.plot([0, 1], [0, 1], linestyle="--", label="Random guess")  # 대각선 기준선

plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()