import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from prefix_hyper import (
    normalize_video_feature_shape,
    build_fixed_default_adapter,
    apply_adapter_with_generated_ln,
)


class SafeMetaPolicyNet(nn.Module):
    """
    prefix 5개를 보고
    - g: adaptation permission
    - alpha: per-video inner-step size
    - w: prefix segment weights
    를 예측하는 policy net
    """
    def __init__(self, warmup_segments=5, stats_dim=7, hidden_dim=64, lr_max=1e-2):
        super().__init__()
        self.warmup_segments = warmup_segments
        self.stats_dim = stats_dim
        self.input_dim = stats_dim + 2 * warmup_segments + 1
        self.hidden_dim = hidden_dim
        self.lr_max = lr_max

        self.trunk = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.gate_head = nn.Linear(hidden_dim, 1)
        self.alpha_head = nn.Linear(hidden_dim, 1)
        self.weight_head = nn.Linear(hidden_dim, warmup_segments)

        # 처음엔 보수적으로 시작
        nn.init.zeros_(self.gate_head.weight)
        nn.init.constant_(self.gate_head.bias, -0.5)   # g ~ 0.38 근처

        nn.init.zeros_(self.alpha_head.weight)
        nn.init.constant_(self.alpha_head.bias, -2.0)  # alpha는 작게 시작

        nn.init.zeros_(self.weight_head.weight)
        nn.init.zeros_(self.weight_head.bias)          # 처음엔 거의 uniform

    def forward(self, feat, prefix_len):
        """
        feat: (B, input_dim)
        prefix_len: int
        """
        h = self.trunk(feat)

        g = torch.sigmoid(self.gate_head(h)).squeeze(-1)                 # (B,)
        alpha = self.lr_max * torch.sigmoid(self.alpha_head(h)).squeeze(-1)  # (B,)

        w_logits = self.weight_head(h)  # (B, K)

        if prefix_len < self.warmup_segments:
            mask = torch.full_like(w_logits, -1e9)
            mask[:, :prefix_len] = 0.0
            w = torch.softmax(w_logits + mask, dim=-1)
        else:
            w = torch.softmax(w_logits, dim=-1)

        return g, alpha, w


def extract_policy_features(
    x_video_np,
    base_adapter,
    model,
    device,
    warmup_segments=5,
):
    """
    prefix 5개에서 policy 입력 feature를 만든다.

    feature 구성:
    [global stats 7개, prefix prob K개, prefix logit K개, prefix_ratio 1개]
    총 dim = 7 + 2K + 1
    """
    x_video_np = normalize_video_feature_shape(x_video_np)
    x_video = torch.from_numpy(x_video_np).float().to(device)

    T = x_video.shape[0]
    prefix_len = min(warmup_segments, T)
    x_prefix = x_video[:prefix_len]

    if prefix_len == 0:
        feat = torch.zeros(7 + 2 * warmup_segments + 1, device=device)
        return feat, x_video, prefix_len

    x_prefix_in = x_prefix.unsqueeze(0)  # (1, K, D)

    base_adapter.eval()
    model.eval()

    with torch.no_grad():
        x_2048 = base_adapter(x_prefix_in)
        prob, logit = model(x_2048, return_logits=True)

    prob = prob[0, :, 0]    # (K,)
    logit = logit[0, :, 0]  # (K,)

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

    prob_pad = torch.zeros(warmup_segments, device=device)
    logit_pad = torch.zeros(warmup_segments, device=device)
    prob_pad[:prefix_len] = prob
    logit_pad[:prefix_len] = logit

    prefix_ratio = torch.tensor([prefix_len / float(warmup_segments)], device=device)

    feat = torch.cat([stats, prob_pad, logit_pad, prefix_ratio], dim=0)
    return feat, x_video, prefix_len


def policy_inner_update(
    x_video_np,
    base_adapter,
    model,
    policy_net,
    device,
    warmup_segments=5,
    inner_steps=3,
    create_graph=False,
    gate_threshold=None,
):
    """
    policy가 g, alpha, w를 예측하고
    그걸 이용해 LN gamma/beta를 실제로 inner update 한다.
    """
    feat, x_video, prefix_len = extract_policy_features(
        x_video_np=x_video_np,
        base_adapter=base_adapter,
        model=model,
        device=device,
        warmup_segments=warmup_segments,
    )

    feat = feat.unsqueeze(0)  # (1, input_dim)

    g, alpha, w = policy_net(feat, prefix_len)
    g = g[0]
    alpha = alpha[0]
    w = w[0, :prefix_len]     # valid prefix만 사용

    gamma0 = base_adapter.ln.weight.detach().clone().to(device)
    beta0 = base_adapter.ln.bias.detach().clone().to(device)

    gamma = gamma0.clone().requires_grad_(True)
    beta = beta0.clone().requires_grad_(True)

    if prefix_len == 0:
        info = {
            "g_tensor": g,
            "alpha_tensor": alpha,
            "w_tensor": w,
            
            "prefix_len": 0,
            "gate": float(g.item()),
            "alpha": float(alpha.item()),
            "weights": [],
            "skipped": True,
            "reason": "empty_prefix",
        }
        return gamma0, beta0, info

    if (not create_graph) and gate_threshold is not None and g.item() < gate_threshold:
        info = {
            "g_tensor": g,
            "alpha_tensor": alpha,
            "w_tensor": w,

            "prefix_len": int(prefix_len),
            "gate": float(g.item()),
            "alpha": float(alpha.item()),
            "weights": w.detach().cpu().tolist(),
            "skipped": True,
            "reason": "gate_below_threshold",
        }
        return gamma0, beta0, info

    x_prefix = x_video[:prefix_len].unsqueeze(0)  # (1, K, D)

    for _ in range(inner_steps):
        x_prefix_adapted = apply_adapter_with_generated_ln(
            x_prefix, base_adapter, gamma, beta
        )
        _, logit_prefix = model(x_prefix_adapted, return_logits=True)
        logit_prefix = logit_prefix[0, :prefix_len, 0]  # (K,)

        prefix_energy = F.softplus(logit_prefix)        # (K,)
        inner_loss = g * (w * prefix_energy).sum()

        dgamma, dbeta = torch.autograd.grad(
            inner_loss,
            [gamma, beta],
            create_graph=create_graph,
            retain_graph=create_graph,
        )

        gamma = gamma - alpha * dgamma
        beta = beta - alpha * dbeta

    info = {
        # train에서 gradient 써야 하는 텐서
        "g_tensor": g,
        "alpha_tensor": alpha,
        "w_tensor": w,

        # debug용
        "prefix_len": int(prefix_len),
        "gate": float(g.item()),
        "alpha": float(alpha.item()),
        "weights": w.detach().cpu().tolist(),
        "skipped": False,
    }

    return gamma, beta, info


def build_trusted_suffix_masks(
    base_prob,
    y_suffix,
    normal_q=0.40,          # normal bucket: 하위 40%
    anom_q=0.10,            # anomaly bucket: 상위 10%
    tail_gap_score=0.03,    # score tail이 충분히 벌어졌는지
    tail_gap_pseudo=0.02,   # pseudo tail이 충분히 벌어졌는지
    min_keep_normal=4,
    min_keep_anom=2,
):
    """
    suffix에서 신뢰할 normal/anomaly bucket을 만든다.

    핵심:
    - normal bucket은 거의 항상 만듦
    - anomaly bucket은 '이 비디오에 이상 tail이 있다'고 판단될 때만 만듦
    - 0.5 같은 절대 threshold는 쓰지 않음
    """

    T = len(base_prob)
    device = base_prob.device

    # -------------------------
    # 1) trusted normal bucket
    #    score와 pseudo 둘 다 낮은 쪽의 교집합
    # -------------------------
    score_low_thr = torch.quantile(base_prob.detach(), normal_q)
    pseudo_low_thr = torch.quantile(y_suffix.detach(), normal_q)

    norm_mask = (base_prob <= score_low_thr) & (y_suffix <= pseudo_low_thr)

    if int(norm_mask.sum().item()) < min_keep_normal:
        k = min(max(min_keep_normal, 1), T)
        idx = torch.argsort(base_prob.detach())[:k]
        norm_mask = torch.zeros(T, dtype=torch.bool, device=device)
        norm_mask[idx] = True

    # -------------------------
    # 2) anomaly tail 존재 여부 판단
    # -------------------------
    score_med = torch.quantile(base_prob.detach(), 0.50)
    score_hi = torch.quantile(base_prob.detach(), 1.0 - anom_q)

    pseudo_med = torch.quantile(y_suffix.detach(), 0.50)
    pseudo_hi = torch.quantile(y_suffix.detach(), 1.0 - anom_q)

    has_score_tail = (score_hi - score_med) > tail_gap_score
    has_pseudo_tail = (pseudo_hi - pseudo_med) > tail_gap_pseudo

    has_anomaly_tail = bool((has_score_tail & has_pseudo_tail).item())

    # -------------------------
    # 3) trusted anomaly bucket
    #    tail이 있을 때만 만듦
    # -------------------------
    anom_mask = torch.zeros(T, dtype=torch.bool, device=device)

    if has_anomaly_tail:
        score_hi_mask = base_prob >= score_hi
        pseudo_hi_mask = y_suffix >= pseudo_hi

        # conservative: 둘 다 높은 애들만
        anom_mask = score_hi_mask & pseudo_hi_mask

        # 너무 적으면 score high 쪽만 fallback
        if int(anom_mask.sum().item()) < min_keep_anom:
            anom_mask = score_hi_mask

        # 그래도 너무 적으면 top-k by score
        if int(anom_mask.sum().item()) < min_keep_anom:
            k = min(max(min_keep_anom, 1), T)
            idx = torch.argsort(base_prob.detach(), descending=True)[:k]
            anom_mask = torch.zeros(T, dtype=torch.bool, device=device)
            anom_mask[idx] = True

    debug = {
        "score_low_thr": float(score_low_thr.item()),
        "pseudo_low_thr": float(pseudo_low_thr.item()),
        "score_med": float(score_med.item()),
        "score_hi": float(score_hi.item()),
        "pseudo_med": float(pseudo_med.item()),
        "pseudo_hi": float(pseudo_hi.item()),
        "has_score_tail": bool(has_score_tail.item()),
        "has_pseudo_tail": bool(has_pseudo_tail.item()),
        "has_anomaly_tail": has_anomaly_tail,
        "num_norm_bucket": int(norm_mask.sum().item()),
        "num_anom_bucket": int(anom_mask.sum().item()),
    }

    return norm_mask, anom_mask, has_anomaly_tail, debug


def compute_safe_outer_loss(
    x_video,
    y_video,
    base_adapter,
    model,
    gamma_adapt,
    beta_adapt,
    warmup_segments=5,
    normal_q=0.4,
    anom_q=0.1,
    tail_gap_score=0.03,
    tail_gap_pseudo=0.02,
    preserve_margin=0.02,
    rank_margin=0.05,
    min_keep_normal=4,
    min_keep_anom=2,
):
    """
    adaptation 후 suffix에서
    - trusted normal은 더 낮아지게
    - trusted anomaly는 '있을 때만' 너무 깎이지 않게
    - anomaly tail이 있을 때만 rank margin 유지
    """
    T = x_video.shape[0]
    prefix_len = min(warmup_segments, T)

    if T <= prefix_len:
        return None

    x_suffix = x_video[prefix_len:]
    y_suffix = y_video[prefix_len:]

    x_suffix_in = x_suffix.unsqueeze(0)  # (1, T-K, D)

    gamma0 = base_adapter.ln.weight.detach()
    beta0 = base_adapter.ln.bias.detach()

    # baseline suffix score
    with torch.no_grad():
        x_suffix_base = apply_adapter_with_generated_ln(
            x_suffix_in, base_adapter, gamma0, beta0
        )
        prob_base, _ = model(x_suffix_base, return_logits=True)
        prob_base = prob_base[0, :, 0]   # (T-K,)

    # adapted suffix score
    x_suffix_adapt = apply_adapter_with_generated_ln(
        x_suffix_in, base_adapter, gamma_adapt, beta_adapt
    )
    prob_adapt, _ = model(x_suffix_adapt, return_logits=True)
    prob_adapt = prob_adapt[0, :, 0]     # (T-K,)

    norm_mask, anom_mask, has_anomaly_tail, mask_debug = build_trusted_suffix_masks(
        base_prob=prob_base,
        y_suffix=y_suffix,
        normal_q=normal_q,
        anom_q=anom_q,
        tail_gap_score=tail_gap_score,
        tail_gap_pseudo=tail_gap_pseudo,
        min_keep_normal=min_keep_normal,
        min_keep_anom=min_keep_anom,
    )

    # 1) trusted normal은 낮춰라
    loss_norm = prob_adapt[norm_mask].mean()

    # 2) anomaly bucket은 있을 때만 보존/분리 제약
    if has_anomaly_tail and int(anom_mask.sum().item()) > 0:
        drop = prob_base[anom_mask] - prob_adapt[anom_mask]
        loss_preserve = F.relu(drop - preserve_margin).mean()

        gap = prob_adapt[anom_mask].mean() - prob_adapt[norm_mask].mean()
        loss_rank = F.relu(rank_margin - gap)

        gap_value = float(gap.item())
        base_anom_mean = float(prob_base[anom_mask].mean().item())
        adapt_anom_mean = float(prob_adapt[anom_mask].mean().item())
    else:
        loss_preserve = torch.zeros((), device=x_video.device)
        loss_rank = torch.zeros((), device=x_video.device)

        gap_value = None
        base_anom_mean = None
        adapt_anom_mean = None

    debug = {
        **mask_debug,
        "base_norm_mean": float(prob_base[norm_mask].mean().item()),
        "adapt_norm_mean": float(prob_adapt[norm_mask].mean().item()),
        "base_anom_mean": base_anom_mean,
        "adapt_anom_mean": adapt_anom_mean,
        "gap": gap_value,
    }

    return loss_norm, loss_preserve, loss_rank, debug