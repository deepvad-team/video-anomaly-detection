import torch
import torch.nn as nn
import numpy as np


class PrefixGateNet(nn.Module):
    """
    prefix 통계 -> adaptation을 할지 말지(logit) 예측
    """
    def __init__(self, in_dim=7, hidden_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        # x: (B, in_dim)
        return self.net(x).squeeze(-1)   # raw logit


def normalize_video_feature_shape(x_video_np):
    """
    입력 feature를 (T, D)로 맞춤
    """
    x = np.asarray(x_video_np)

    if x.ndim == 2:
        return x

    if x.ndim == 3:
        return x.mean(axis=1)

    if x.ndim == 4:
        x = np.squeeze(x)
        if x.ndim == 2:
            return x
        elif x.ndim == 3:
            return x.mean(axis=1)

    raise ValueError(f"Unsupported feature shape: {x.shape}")


def compute_prefix_stats(
    x_video_np,
    adapter,
    model,
    device,
    warmup_segments=5,
):
    """
    prefix 5개만 보고 gate 입력용 통계 벡터 생성
    출력: torch tensor, shape (7,)
    """
    x_video_np = normalize_video_feature_shape(x_video_np)
    x_video = torch.from_numpy(x_video_np).float().to(device)

    T = x_video.shape[0]
    prefix_len = min(warmup_segments, T)
    x_prefix = x_video[:prefix_len]

    if prefix_len == 0:
        return torch.zeros(7, device=device)

    x_prefix_in = x_prefix.unsqueeze(0)   # (1, K, D)

    adapter.eval()
    model.eval()

    with torch.no_grad():
        x_2048 = adapter(x_prefix_in)
        prob, logit = model(x_2048, return_logits=True)

    prob = prob[0, :, 0]    # (K,)
    logit = logit[0, :, 0]  # (K,)

    # prefix 내부 raw feature consistency
    proto = x_prefix.mean(dim=0, keepdim=True)
    dists = torch.norm(x_prefix - proto, dim=1)

    stats = torch.stack([
        prob.mean(),
        prob.std(unbiased=False),
        prob.max(),
        prob.min(),
        logit.mean(),
        logit.std(unbiased=False),
        dists.mean(),
    ], dim=0)

    return stats