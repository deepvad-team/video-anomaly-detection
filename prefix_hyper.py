import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from adapter import ResidualAdapter2048


def normalize_video_feature_shape(x_video_np):
    """
    입력 feature를 (T, D)로 맞춤
    가능한 입력:
      (T, D)
      (T, C, D)  -> crop 평균
      (T, 1, C, D) 등 -> squeeze 후 crop 평균
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


def build_fixed_default_adapter(device, init_ckpt="adapter_init.pt"):
    """
    항상 같은 baseline adapter 초기값을 쓰기 위한 helper
    """
    adapter = ResidualAdapter2048(d=2048, use_ln=True).to(device)

    if os.path.exists(init_ckpt):
        adapter.load_state_dict(torch.load(init_ckpt, map_location=device))
        print(f"[Adapter Init] loaded fixed init from {init_ckpt}")
    else:
        torch.save(adapter.state_dict(), init_ckpt)
        print(f"[Adapter Init] saved new fixed init to {init_ckpt}")

    adapter.eval()
    for p in adapter.parameters():
        p.requires_grad = False

    return adapter


def compute_prefix_stats(
    x_video_np,
    adapter,
    model,
    device,
    warmup_segments=5,
):
    """
    prefix 5개에서 간단한 scene statistics 추출
    shape: (7,)
    """
    x_video_np = normalize_video_feature_shape(x_video_np)
    x_video = torch.from_numpy(x_video_np).float().to(device)

    T = x_video.shape[0]
    prefix_len = min(warmup_segments, T)
    x_prefix = x_video[:prefix_len]

    if prefix_len == 0:
        return torch.zeros(7, device=device)

    x_prefix_in = x_prefix.unsqueeze(0)  # (1, K, D)

    adapter.eval()
    model.eval()

    with torch.no_grad():
        x_2048 = adapter(x_prefix_in)
        prob, logit = model(x_2048, return_logits=True)

    prob = prob[0, :, 0]
    logit = logit[0, :, 0]

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


def apply_adapter_with_generated_ln(x, base_adapter, gamma, beta):
    """
    base adapter의 구조는 유지하되,
    LN gamma/beta만 prefix-conditioned 값으로 대체
    """
    squeeze_back = False
    if x.dim() == 2:
        x = x.unsqueeze(0)
        squeeze_back = True

    h = x

    # residual delta branch가 있으면 그대로 반영
    if hasattr(base_adapter, "delta") and base_adapter.delta is not None:
        h = x + base_adapter.delta(x)

    eps = 1e-5
    if hasattr(base_adapter, "ln") and base_adapter.ln is not None:
        eps = base_adapter.ln.eps

    y = F.layer_norm(
        h,
        normalized_shape=(h.shape[-1],),
        weight=gamma,
        bias=beta,
        eps=eps,
    )

    if squeeze_back:
        y = y.squeeze(0)

    return y


class PrefixHyperNet(nn.Module):
    """
    prefix statistics -> scene-specific LN gamma/beta 생성
    + adaptation 강도 gate g도 같이 생성
    """
    def __init__(self, stats_dim=7, feat_dim=2048, hidden_dim=128, delta_scale=0.10):
        super().__init__()
        self.stats_dim = stats_dim
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim
        self.delta_scale = delta_scale

        self.trunk = nn.Sequential(
            nn.Linear(stats_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.gate_head = nn.Linear(hidden_dim, 1)
        self.gamma_head = nn.Linear(hidden_dim, feat_dim)
        self.beta_head = nn.Linear(hidden_dim, feat_dim)

        # 처음엔 base adapter와 거의 동일하게 시작
        nn.init.zeros_(self.gate_head.weight)
        nn.init.zeros_(self.gate_head.bias)
        nn.init.zeros_(self.gamma_head.weight)
        nn.init.zeros_(self.gamma_head.bias)
        nn.init.zeros_(self.beta_head.weight)
        nn.init.zeros_(self.beta_head.bias)

    def forward(self, stats, gamma0, beta0):
        """
        stats:  (B, stats_dim)
        gamma0: (feat_dim,) or (B, feat_dim)
        beta0 : (feat_dim,) or (B, feat_dim)
        """
        if gamma0.dim() == 1:
            gamma0 = gamma0.unsqueeze(0).expand(stats.shape[0], -1)
        if beta0.dim() == 1:
            beta0 = beta0.unsqueeze(0).expand(stats.shape[0], -1)

        h = self.trunk(stats)

        g = torch.sigmoid(self.gate_head(h))               # (B,1)
        dgamma = self.delta_scale * torch.tanh(self.gamma_head(h))  # (B,D)
        dbeta = self.delta_scale * torch.tanh(self.beta_head(h))    # (B,D)

        gamma = gamma0 + g * dgamma
        beta = beta0 + g * dbeta

        return gamma, beta, g.squeeze(-1), dgamma, dbeta