import numpy as np

# 학습용 데이터 확인
train_data = np.memmap('concat_UCF.npy', mode='r')
print(f"학습용 데이터 Shape: {train_data.shape}")

# 테스트용 데이터 확인
test_data = np.memmap('Concat_test_10.npy', mode='r')
print(f"테스트용 데이터 Shape: {test_data.shape}")