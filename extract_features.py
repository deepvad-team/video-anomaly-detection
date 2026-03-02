import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

import numpy as np
import torch
from natsort import natsorted
from PIL import Image
from torch.autograd import Variable


# 프레임 한 장을 읽어서 모델 입력 형식으로 바꾸는 전처리 함수
def load_frame(frame_file):
	data = Image.open(frame_file)
	data = data.resize((340, 256), Image.Resampling.LANCZOS) # 원본 프레임을 가로 340 세로 256 으로 맞춤
	data = np.array(data) #(256, 240, 3)
	data = data.astype(float)
	data = (data * 2 / 255) - 1 # 정규화. 픽셀 범위를 [0, 255]에서 [-1, 1]로 변환. backbone이 기대하는 입력 정규화 방식.
	assert(data.max()<=1.0)
	assert(data.min()>=-1.0)
	return data

# 여러 chunk에 필요한 프레임들을 한꺼번에 로드하는 역할. 
# 예를 들어, batch안에 chunk가 20개 있으면, 각 chunk마다 16프레임씩 필요하니까 (20, 16, 256, 340, 3) 이걸 채우는 구조.
def load_rgb_batch(frames_dir, rgb_files, frame_indices):
	batch_data = np.zeros(frame_indices.shape + (256,340,3))
	for i in range(frame_indices.shape[0]):
		for j in range(frame_indices.shape[1]):
			batch_data[i,j,:,:,:] = load_frame(os.path.join(frames_dir, rgb_files[frame_indices[i][j]]))
	return batch_data


# spatial augmentation/oversampling 담당 
def oversample_data(data): # 입력 shape (B, T, H, W, C)
	data_flip = np.array(data[:,:,:,::-1,:]) #좌우 반전

	#5개 CROP
	data_1 = np.array(data[:, :, :224, :224, :])
	data_2 = np.array(data[:, :, :224, -224:, :])
	data_3 = np.array(data[:, :, 16:240, 58:282, :]) #중앙 crop. resize한 256x340에서 중앙 224x224를 자름.
	data_4 = np.array(data[:, :, -224:, :224, :])
	data_5 = np.array(data[:, :, -224:, -224:, :])

	#좌우반전 본 5개 추가
	data_f_1 = np.array(data_flip[:, :, :224, :224, :])
	data_f_2 = np.array(data_flip[:, :, :224, -224:, :])
	data_f_3 = np.array(data_flip[:, :, 16:240, 58:282, :])
	data_f_4 = np.array(data_flip[:, :, -224:, :224, :])
	data_f_5 = np.array(data_flip[:, :, -224:, -224:, :])

	return [data_1, data_2, data_3, data_4, data_5,
		data_f_1, data_f_2, data_f_3, data_f_4, data_f_5]


def run(i3d, frequency, frames_dir, batch_size, sample_mode):
	assert(sample_mode in ['oversample', 'center_crop'])
	print("batchsize", batch_size)

	# chunk size 고정! 1 chunk(segment, clip)에 16 frames
	chunk_size = 16

	def forward_batch(b_data):
		b_data = b_data.transpose([0, 4, 1, 2, 3])
		b_data = torch.from_numpy(b_data)   # b,c,t,h,w  # 40x3x16x224x224 PyTorch 3D conv 모델이 원하는 입력으로 바꿔줌.
		with torch.no_grad():
			b_data = Variable(b_data.cuda()).float()
			inp = {'frames': b_data}
			features = i3d(inp) # 백본에 통과시킴
		return features.cpu().numpy()

	rgb_files = natsorted([i for i in os.listdir(frames_dir)]) # 중요! 프레임 파일 정렬해주는 역할. temporal oder 보장해주는 애 (1.jpg, 2.jpg ...)
	frame_cnt = len(rgb_files)

	# Cut frames
	assert(frame_cnt > chunk_size) #최소 길이 검사 (16프레임보다 짧은 비디오는 feature 생성 못 함)

	#중요!!!!!!!! 끝 단에서 16프레임이 안되는 애들은 버려짐. 
	clipped_length = frame_cnt - chunk_size
	clipped_length = (clipped_length // frequency) * frequency  # The start of last chunk

	frame_indices = [] # Frames to chunks
	for i in range(clipped_length // frequency + 1):
		frame_indices.append([j for j in range(i * frequency, i * frequency + chunk_size)]) #중요 !!!!!!!!!!! stride = frequency = 16 (즉, 겹치지 않게 뽑음)
	frame_indices = np.array(frame_indices)
	#한 번에 너무 많은 chunk를 못 돌리니까 batch로 나눔.
	chunk_num = frame_indices.shape[0]
	batch_num = int(np.ceil(chunk_num / batch_size))    # Chunks to batches
	frame_indices = np.array_split(frame_indices, batch_num, axis=0)
	
	if sample_mode == 'oversample':
		full_features = [[] for i in range(10)]  #crop 0 전체 chunk feature 들, crop 1 전체 chunk feature 들 ... crop 9 전체 chunk feature 들을 합쳐서 (T, 10, 2048) 로 만듦.
	else: #center_crop이면 이렇게 저장
		full_features = [[]]


	for batch_id in range(batch_num): 
		batch_data = load_rgb_batch(frames_dir, rgb_files, frame_indices[batch_id]) #현재 batch에 해당하는 chunk들의 프레임을 읽어옴.
		if(sample_mode == 'oversample'):
		   batch_data_ten_crop = oversample_data(batch_data)
		   for i in range(10): #같은 temporal batch에 대해 각 crop version을 backbone에 따로 넣고, 각 결과를 crop 별 리스트에 저장
			   assert(batch_data_ten_crop[i].shape[-2]==224)
			   assert(batch_data_ten_crop[i].shape[-3]==224)
			   temp = forward_batch(batch_data_ten_crop[i])
			   full_features[i].append(temp)

		elif(sample_mode == 'center_crop'):
			batch_data = batch_data[:,:,16:240,58:282,:]
			assert(batch_data.shape[-2]==224)
			assert(batch_data.shape[-3]==224)
			temp = forward_batch(batch_data)
			full_features[0].append(temp)
	
	full_features = [np.concatenate(i, axis=0) for i in full_features]
	full_features = [np.expand_dims(i, axis=0) for i in full_features]
	full_features = np.concatenate(full_features, axis=0)
	full_features = full_features[:,:,:,0,0,0]
	full_features = np.array(full_features).transpose([1,0,2])
	return full_features
