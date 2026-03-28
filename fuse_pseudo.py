import numpy as np
# 실험 1
# 1) 원래 쓰던 pseudo label 파일
old_path = "Unsup_labels/pseudo_labels_swap_90.npy"   

# 2) refined pseudo label / weight
prefix = 5
ref_label_path = f"Unsup_labels/pseudo_prop{prefix}_label_glist.npy"
ref_weight_path = f"Unsup_labels/pseudo_prop{prefix}_weight_glist.npy"   

# refined 쪽 반영 강도 하이퍼파라미터
eta = 0.5
gamma = 3.0

# 3) 저장하여 모델에 넘겨줄 최종 "fused" pseudo label
save_path = f"Unsup_labels/pseudo_fused_prop{prefix}_soft_e{eta}_g{gamma}.npy"

y_old = np.load(old_path).astype(np.float32)
y_ref = np.load(ref_label_path).astype(np.float32)
w_ref = np.load(ref_weight_path).astype(np.float32)

assert len(y_old) == len(y_ref) == len(w_ref), (
    f"length mismatch: old={len(y_old)}, ref={len(y_ref)}, weight={len(w_ref)}"
)

# refined label이 유효한 위치만 사용 (0, 1)
valid = (y_ref != -1)

# alpha = eta * confidence
alpha = np.zeros_like(w_ref, dtype=np.float32)
#alpha[valid] = eta * w_ref[valid] # 0, 1 에 해당하는 가중치만 가져옴 
alpha[valid] = eta * (w_ref[valid] ** gamma)

# valid하지 않은 곳은 기존(swap) label 유지, valid 부분만 y_ref로 교체
y_ref_clean = y_old.copy()
y_ref_clean[valid] = y_ref[valid]

# soft fusion
y_final = (1.0 - alpha) * y_old + alpha * y_ref_clean
y_final = np.clip(y_final, 0.0, 1.0).astype(np.float32)

print("old mean(기존 pl의 평균):", float(y_old.mean())) #(기존 pl의 평균. 즉 1의 비율)
print("ref valid ratio:", float(valid.mean())) #(refine pl에서 0과 1의 비율)
print("final mean(fusion 후 pl 평균):", float(y_final.mean())) #-> 학습에 들어가는 anomaly supervision 강도)
print("final min/max:", float(y_final.min()), float(y_final.max()))

np.save(save_path, y_final)
print("saved:", save_path)




# 실험 2
"""old와 refined가 같은 방향이면
→ refined를 더 강하게 반영
old와 refined가 충돌하면
→ refined를 약하게만 반영하거나, 중립 쪽으로 둠
refined가 -1이면
→ old 유지"""
'''
# 1) 원래 쓰던 pseudo label 파일
old_path = "Unsup_labels/pseudo_labels_swap_90.npy"

# 2) refined pseudo label / weight
prefix = 5
ref_label_path = f"Unsup_labels/pseudo_prop{prefix}_label_glist.npy"
ref_weight_path = f"Unsup_labels/pseudo_prop{prefix}_weight_glist.npy"

# 3) fusion 강도
eta_agree = 0.8
eta_conflict = 0.2

# 4) 저장할 최종 fused pseudo label
save_path = f"Unsup_labels/pseudo_fused_prop{prefix}_agree_soft_a{eta_agree}_c{eta_conflict}.npy"

y_old = np.load(old_path).astype(np.float32)
y_ref = np.load(ref_label_path).astype(np.float32)
w_ref = np.load(ref_weight_path).astype(np.float32)

assert len(y_old) == len(y_ref) == len(w_ref), (
    f"length mismatch: old={len(y_old)}, ref={len(y_ref)}, weight={len(w_ref)}"
)

# refined label이 유효한 위치만 사용
valid = (y_ref != -1)

# old hard label (혹시 old가 soft여도 방향 판별용으로만 사용)
old_hard = (y_old >= 0.5).astype(np.float32)

agree = valid & (old_hard == y_ref)
conflict = valid & (old_hard != y_ref)

alpha = np.zeros_like(w_ref, dtype=np.float32)
alpha[agree] = eta_agree * w_ref[agree]
alpha[conflict] = eta_conflict * w_ref[conflict]

# valid하지 않은 곳은 기존 label 유지
y_ref_clean = y_old.copy()
y_ref_clean[valid] = y_ref[valid]

# soft fusion
y_final = (1.0 - alpha) * y_old + alpha * y_ref_clean
y_final = np.clip(y_final, 0.0, 1.0).astype(np.float32)

print("old mean:", float(y_old.mean()))
print("ref valid ratio:", float(valid.mean()))
print("agree ratio:", float(agree.mean()))
print("conflict ratio:", float(conflict.mean()))
print("final mean:", float(y_final.mean()))
print("final min/max:", float(y_final.min()), float(y_final.max()))

np.save(save_path, y_final)
print("saved:", save_path)

'''


"""Propagation으로 “확실한 중심”을 찾고,
Prototype으로 “경계와 연속성”을 채우고,
애매한 곳은 버리지 말고 soft하게 학습시키자"""