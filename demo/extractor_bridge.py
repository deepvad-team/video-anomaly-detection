import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable


def load_frame_from_array(frame_bgr):
    """
    frame_bgr: OpenCV frame (H,W,3), uint8, BGR
    return: (256,340,3) float in [-1,1], RGB
    """
    # OpenCV BGR -> RGB
    frame_rgb = frame_bgr[:, :, ::-1]
    data = Image.fromarray(frame_rgb)
    data = data.resize((340, 256), Image.Resampling.LANCZOS)
    data = np.array(data).astype(np.float32)
    data = (data * 2.0 / 255.0) - 1.0

    assert data.max() <= 1.0 + 1e-5
    assert data.min() >= -1.0 - 1e-5
    return data


def oversample_data(data):
    # data: (B, T, H, W, C)
    data_flip = np.array(data[:, :, :, ::-1, :])

    data_1 = np.array(data[:, :, :224, :224, :])
    data_2 = np.array(data[:, :, :224, -224:, :])
    data_3 = np.array(data[:, :, 16:240, 58:282, :])
    data_4 = np.array(data[:, :, -224:, :224, :])
    data_5 = np.array(data[:, :, -224:, -224:, :])

    data_f_1 = np.array(data_flip[:, :, :224, :224, :])
    data_f_2 = np.array(data_flip[:, :, :224, -224:, :])
    data_f_3 = np.array(data_flip[:, :, 16:240, 58:282, :])
    data_f_4 = np.array(data_flip[:, :, -224:, :224, :])
    data_f_5 = np.array(data_flip[:, :, -224:, -224:, :])

    return [
        data_1, data_2, data_3, data_4, data_5,
        data_f_1, data_f_2, data_f_3, data_f_4, data_f_5
    ]


def _forward_batch(i3d, b_data):
    # b_data: (B,T,H,W,C)
    b_data = b_data.transpose([0, 4, 1, 2, 3])   # -> (B,C,T,H,W)
    b_data = torch.from_numpy(b_data)

    with torch.no_grad():
        b_data = Variable(b_data.cuda()).float()
        inp = {'frames': b_data}
        features = i3d(inp)

    return features.cpu().numpy()


def extract_clip_feature(frames, i3d, sample_mode='oversample'):
    """
    frames: list of 16 webcam frames (OpenCV BGR)
    return:
      oversample -> (1, 10, 2048)
      center_crop -> (1, 1, 2048) or squeezed later
    """
    assert sample_mode in ['oversample', 'center_crop']
    assert len(frames) == 16, f"Expected 16 frames, got {len(frames)}"

    # 16 frames -> one chunk
    chunk = np.stack([load_frame_from_array(f) for f in frames], axis=0)   # (16,256,340,3)
    batch_data = np.expand_dims(chunk, axis=0)                              # (1,16,256,340,3)

    if sample_mode == 'oversample':
        batch_data_ten_crop = oversample_data(batch_data)
        full_features = []

        for i in range(10):
            temp = _forward_batch(i3d, batch_data_ten_crop[i])   # expected (1,2048,1,1,1)
            full_features.append(temp)

        full_features = [np.expand_dims(f, axis=0) for f in full_features]
        full_features = np.concatenate(full_features, axis=0)     # (10,1,2048,1,1,1) maybe
        full_features = full_features[:, :, :, 0, 0, 0]          # (10,1,2048)
        full_features = full_features.transpose([1, 0, 2])       # (1,10,2048)
        return full_features.astype(np.float32)

    else:
        batch_data = batch_data[:, :, 16:240, 58:282, :]
        temp = _forward_batch(i3d, batch_data)                   # (1,2048,1,1,1)
        temp = temp[:, :, 0, 0, 0]                               # (1,2048)
        temp = np.expand_dims(temp, axis=1)                      # (1,1,2048)
        return temp.astype(np.float32)