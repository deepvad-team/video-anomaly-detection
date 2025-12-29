import numpy as np
import os
import torch
import torch.nn.functional as F

feature_dir = "C:\\Users\\jplabuser\\Downloads\\UCF_train_feature\\UCF_Train_ten_crop_i3d"
video_files = sorted([f for f in os.listdir(feature_dir) if f.endswith('.npy')])

num_videos = len(video_files)
target_segments = 32
feature_dim = 2048

# 메모리 부족을 방지하기 위해 memmap 사용 (float32로 용량 최적화)
output_file = 'concat_UCF.npy'
fp = np.memmap(output_file, dtype='float32', mode='w+', shape=(num_videos, target_segments, 10, feature_dim))

for i, video_file in enumerate(video_files):
    # 1. 로드 (예: shape (171, 10, 2048) 또는 (171, 2048))
    feat = np.load(os.path.join(feature_dir, video_file)).astype(np.float32)
    
    # 2. Ten-crop(10)인 경우 평균을 내서 (171, 2048)로 만듦
    #if len(feat.shape) == 3: # (T, 10, 2048)
    #    feat = np.mean(feat, axis=1) # (T, 2048)
    
    # 3. 32개 세그먼트로 Linear Interpolation (논문 표준 방식)
    # torch의 interpolate를 쓰면 가장 정확하고 빠릅니다.
    #feat_tensor = torch.from_numpy(feat).unsqueeze(0).permute(0, 2, 1) # (1, 2048, T)
    #resampled_feat = F.interpolate(feat_tensor, size=target_segments, mode='linear', align_corners=False)
    #resampled_feat = resampled_feat.permute(0, 2, 1).squeeze(0).numpy() # (32, 2048)

    feat_tensor = torch.from_numpy(feat).permute(1, 2, 0)
    resampled_feat = F.interpolate(feat_tensor, size=target_segments, mode='linear', align_corners=False)
    resampled_feat = resampled_feat.permute(2, 0, 1).numpy()

    # 4. 저장
    #fp[i, :, :] = resampled_feat
    fp[i, :, :, :] = resampled_feat
    
    if i % 100 == 0:
        print(f"Progress: {i}/{num_videos} 완료")

fp.flush()
print("🎉 concat_UCF.npy 생성 완료!")