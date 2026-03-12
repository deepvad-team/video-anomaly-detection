import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# 1) 파일 로드
gt = np.load("list/gt-ucf-RTFM.npy").reshape(-1)
pred = np.load("pred_raw_frame_UCF.npy").reshape(-1)

precision = np.load("precision_UCF.npy").reshape(-1)
recall = np.load("recall_UCF.npy").reshape(-1)
thresholds = np.load("pr_threshold_UCF.npy").reshape(-1)

# 2) 기본 확인
print("gt shape:", gt.shape)
print("pred shape:", pred.shape)
print("precision shape:", precision.shape)
print("recall shape:", recall.shape)
print("thresholds shape:", thresholds.shape)

assert len(gt) == len(pred), "gt와 pred 길이가 다릅니다."
assert len(precision) == len(recall), "precision/recal 길이가 다릅니다."
assert len(thresholds) == len(precision) - 1, "threshold 길이는 보통 precision보다 1 작아야 합니다."

print("gt unique:", np.unique(gt))
print("pred range:", pred.min(), pred.max())

# 3) threshold와 길이 맞는 precision/recall로 F1 계산
precision_t = precision[:-1]
recall_t = recall[:-1]

f1_scores = 2 * precision_t * recall_t / (precision_t + recall_t + 1e-12)

best_idx = np.argmax(f1_scores)
best_thr = thresholds[best_idx]
best_f1_from_curve = f1_scores[best_idx]

print("\nBest index:", best_idx)
print("Best threshold by PR-curve F1:", best_thr)
print("Best F1 from saved precision/recall:", best_f1_from_curve)

# 4) 그 threshold 하나만 써서 예측값 생성
y_pred = (pred >= best_thr).astype(int)

# 5) confusion matrix 계산
cm = confusion_matrix(gt, y_pred, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()

# 6) 실제 y_pred 기준 metric 다시 계산
precision_final = precision_score(gt, y_pred, zero_division=0)
recall_final = recall_score(gt, y_pred, zero_division=0)
f1_final = f1_score(gt, y_pred, zero_division=0)

print("\nConfusion matrix:")
print(cm)
print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")

print("\nMetrics at best threshold:")
print("Precision:", precision_final)
print("Recall   :", recall_final)
print("F1       :", f1_final)