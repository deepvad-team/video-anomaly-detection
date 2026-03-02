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


def ensure_feature_shape(feat: np.ndarray) -> np.ndarray:
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
        if feat.shape[1] != (10, 2048):
            raise ValueError(f"Unexpected 2D feature shape: {feat.shape}")
        return feat.astype(np.float32)

    raise ValueError(f"Unsupported feature shape: {feat.shape}")


def run_detector_with_adapter(feat_np, adapter, model, device):
    """
    feat_np: (1,10,2048) or (10,2048)

    returns:
      prob_np: (1,)
      logit_np: (1,)
    """
    feat_np = ensure_feature_shape(feat_np)
    x = torch.from_numpy(feat_np).float().to(device)

    adapter.eval()
    model.eval()

    with torch.no_grad():
        x_adapt = adapter(x)
        prob, logit = model(x_adapt, return_logits=True)

    prob = prob.squeeze(-1).detach().cpu().numpy()
    logit = logit.squeeze(-1).detach().cpu().numpy()

    return prob, logit

def run_detector_with_adapter(feat_np, adapter, model, device):
    """
    feat_np: (1,10,2048) or (10,2048)

    returns:
      prob_np: (1,)
      logit_np: (1,)
    """
    feat_np = ensure_feature_shape(feat_np)
    x = torch.from_numpy(feat_np).float().to(device)

    adapter.eval()
    model.eval()

    with torch.no_grad():
        x_adapt = adapter(x)
        prob, logit = model(x_adapt, return_logits=True)

    prob = prob.squeeze(-1).detach().cpu().numpy()
    logit = logit.squeeze(-1).detach().cpu().numpy()

    return prob, logit

class EMASmoother:
    def __init__(self, alpha=0.2):
        self.alpha = alpha
        self.value = None

    def update(self, x):
        x = float(x)
        if self.value is None:
            self.value = x
        else:
            self.value = self.alpha * x + (1.0 - self.alpha) * self.value
        return self.value


class HysteresisAlert:
    def __init__(self, enter_thr=2.0, exit_thr=1.2):
        self.enter_thr = enter_thr
        self.exit_thr = exit_thr
        self.alert = False

    def update(self, score):
        if (not self.alert) and score >= self.enter_thr:
            self.alert = True
        elif self.alert and score <= self.exit_thr:
            self.alert = False
        return self.alert



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
