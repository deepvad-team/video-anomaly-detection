'''
기능
- .npy 로드
- shape check 
- detector input으로 넘기기
'''

# feature_bridge/feature_loader.py
import os
import numpy as np
import torch


def load_feature_npy(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Feature file not found: {path}")
    feat = np.load(path)
    return feat


def normalize_feature_shape(feat: np.ndarray) -> np.ndarray:
    """
    Accept:
      - (T, 10, 2048)
      - (T, 2048)

    Return:
      - same shape if valid
    """
    if feat.ndim == 3:
        # expected: (T, 10, 2048)
        if feat.shape[1] != 10 or feat.shape[2] != 2048:
            raise ValueError(f"Unexpected 3D feature shape: {feat.shape}")
        return feat.astype(np.float32)

    if feat.ndim == 2:
        # expected: (T, 2048)
        if feat.shape[1] != 2048:
            raise ValueError(f"Unexpected 2D feature shape: {feat.shape}")
        return feat.astype(np.float32)

    raise ValueError(f"Unsupported feature shape: {feat.shape}")


def feature_to_tensor(feat: np.ndarray, device: torch.device) -> torch.Tensor:
    feat = normalize_feature_shape(feat)
    return torch.from_numpy(feat).float().to(device)


def summarize_feature(feat: np.ndarray, name: str = "feature") -> None:
    x = normalize_feature_shape(feat)
    x_mean = x.mean(axis=1) if x.ndim == 3 else x
    norms = np.linalg.norm(x_mean, axis=-1)

    print(f"[{name}] shape={x.shape}")
    print(f"  mean={x_mean.mean():.6f}, std={x_mean.std():.6f}")
    print(f"  min={x_mean.min():.6f}, max={x_mean.max():.6f}")
    print(f"  L2 norm mean={norms.mean():.6f}, std={norms.std():.6f}")
