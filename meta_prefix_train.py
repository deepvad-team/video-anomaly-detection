import os
import copy
import random
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import Dataset_Con_all_feedback_UCF, Dataset_Con_all_feedback_XD
from model import Model_V2_AllCNN
from adapter import ResidualAdapter2048


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def freeze_module(module):
    module.eval()
    for p in module.parameters():
        p.requires_grad = False


def enable_ln_only(adapter):
    for p in adapter.parameters():
        p.requires_grad = False

    if not hasattr(adapter, "ln") or adapter.ln is None:
        raise ValueError("Adapter must have LayerNorm 'ln'")

    adapter.ln.weight.requires_grad = True
    adapter.ln.bias.requires_grad = True


def build_detector(args, device):
    model = Model_V2_AllCNN(args.feature_size).to(device)

    ckpt = torch.load(args.model_ckpt, map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        ckpt = ckpt["model"]
    ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}

    model.load_state_dict(ckpt, strict=True)
    freeze_module(model)
    return model


def build_meta_adapter(device):
    adapter = ResidualAdapter2048(d=2048, use_ln=True).to(device)
    enable_ln_only(adapter)
    return adapter


def get_valid_video(features, labels, masks, lengths):
    T = int(lengths[0].item())
    x = features[:, :T, :]
    y = labels[:, :T]
    m = masks[:, :T]
    return x, y, m, T
def _normalize_video_feature_shape(x_video_np):
    x = np.asarray(x_video_np)

    if x.ndim == 2:
        return x

    if x.ndim == 3:
        # (T, 10, 2048) -> crop 평균
        return x.mean(axis=1)

    if x.ndim == 4:
        x = np.squeeze(x)
        if x.ndim == 2:
            return x
        elif x.ndim == 3:
            return x.mean(axis=1)

    raise ValueError(f"Unsupported feature shape: {x.shape}")


def load_train_arrays(args):
    nalist = np.load(args.nalist_path_meta)
    total_T = int(nalist[-1, 1])

    # train feature memmap
    X_flat = np.memmap(
        args.conall_path,
        dtype="float32",
        mode="r",
        shape=(total_T, 10, 2048)
    )

    pseudo_all = np.load(args.pseudofile)
    pseudo_all = np.asarray(pseudo_all).reshape(-1).astype(np.float32)

    if len(pseudo_all) != total_T:
        raise ValueError(
            f"Pseudo label length mismatch: len(pseudo_all)={len(pseudo_all)}, total_T={total_T}"
        )

    return X_flat, nalist, pseudo_all


def get_video_episode(X_flat, pseudo_all, nalist, vid_idx, device):
    s, e = map(int, nalist[vid_idx])

    x_video_np = X_flat[s:e]                       # (T, 10, 2048)
    x_video_np = _normalize_video_feature_shape(x_video_np)   # (T, 2048)
    y_video_np = pseudo_all[s:e]                   # (T,)

    x_video = torch.from_numpy(np.asarray(x_video_np)).float().to(device).unsqueeze(0)  # (1,T,2048)
    y_video = torch.from_numpy(np.asarray(y_video_np)).float().to(device).unsqueeze(0)   # (1,T)
    m_video = torch.ones_like(y_video)

    return x_video, y_video, m_video

def split_prefix_suffix(x, y, m, warmup_segments=5):
    T = x.shape[1]
    if T <= warmup_segments:
        return None

    x_prefix = x[:, :warmup_segments, :]
    x_suffix = x[:, warmup_segments:, :]
    y_suffix = y[:, warmup_segments:]
    m_suffix = m[:, warmup_segments:]

    if x_suffix.shape[1] == 0:
        return None

    return x_prefix, x_suffix, y_suffix, m_suffix


def prefix_loss(detector, adapter, x_prefix):
    adapted = adapter(x_prefix)                       # (1, K, 2048)
    prob, logit = detector(adapted, return_logits=True)
    logit = logit[:, :, 0]                            # (1, K)
    loss = F.softplus(logit).mean()
    return loss


def masked_bce_from_logits(logits, targets, mask, label_smoothing=True):
    if label_smoothing:
        targets = targets * 0.9 + 0.05

    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    loss = loss * mask
    denom = mask.sum().clamp_min(1.0)
    return loss.sum() / denom


def copy_fast_grad_to_meta(meta_adapter, fast_adapter):
    if fast_adapter.ln.weight.grad is None or fast_adapter.ln.bias.grad is None:
        raise RuntimeError("Fast adapter grad is None")

    if meta_adapter.ln.weight.grad is None:
        meta_adapter.ln.weight.grad = fast_adapter.ln.weight.grad.detach().clone()
    else:
        meta_adapter.ln.weight.grad.copy_(fast_adapter.ln.weight.grad.detach())

    if meta_adapter.ln.bias.grad is None:
        meta_adapter.ln.bias.grad = fast_adapter.ln.bias.grad.detach().clone()
    else:
        meta_adapter.ln.bias.grad.copy_(fast_adapter.ln.bias.grad.detach())


def inner_adapt_prefix(detector, fast_adapter, x_prefix, inner_lr, inner_steps):
    optimizer = torch.optim.SGD(
        [fast_adapter.ln.weight, fast_adapter.ln.bias],
        lr=inner_lr
    )

    last_loss = None
    for _ in range(inner_steps):
        loss = prefix_loss(detector, fast_adapter, x_prefix)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        last_loss = loss

    return last_loss.item()


def train_one_epoch(meta_adapter, detector, X_flat, nalist, pseudo_all, device, outer_optimizer, args):
    meta_adapter.train()

    total_inner = 0.0
    total_outer = 0.0
    total_used = 0

    vid_indices = np.random.permutation(len(nalist))
    pbar = tqdm(vid_indices, desc="Meta-Train", dynamic_ncols=True)

    for vid_idx in pbar:
        x, y, m = get_video_episode(X_flat, pseudo_all, nalist, vid_idx, device)

        split = split_prefix_suffix(x, y, m, warmup_segments=args.warmup_segments)
        if split is None:
            continue

        x_prefix, x_suffix, y_suffix, m_suffix = split

        fast_adapter = copy.deepcopy(meta_adapter).to(device)
        enable_ln_only(fast_adapter)

        inner_loss_val = inner_adapt_prefix(
            detector=detector,
            fast_adapter=fast_adapter,
            x_prefix=x_prefix,
            inner_lr=args.inner_lr,
            inner_steps=args.inner_steps,
        )

        adapted_suffix = fast_adapter(x_suffix)
        prob_suf, logit_suf = detector(adapted_suffix, return_logits=True)
        logits = logit_suf[:, :, 0]

        outer_loss = masked_bce_from_logits(
            logits=logits,
            targets=y_suffix,
            mask=m_suffix,
            label_smoothing=args.label_smoothing,
        )

        outer_optimizer.zero_grad(set_to_none=True)
        fast_adapter.zero_grad(set_to_none=True)

        outer_loss.backward()
        copy_fast_grad_to_meta(meta_adapter, fast_adapter)

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                [meta_adapter.ln.weight, meta_adapter.ln.bias],
                args.grad_clip
            )

        outer_optimizer.step()

        total_inner += inner_loss_val
        total_outer += outer_loss.item()
        total_used += 1

        pbar.set_postfix({
            "vid": int(vid_idx),
            "inner": f"{inner_loss_val:.4f}",
            "outer": f"{outer_loss.item():.4f}",
            "used": total_used
        })

    if total_used == 0:
        return 0.0, 0.0

    return total_inner / total_used, total_outer / total_used


def main():
    import option
    args = option.parser.parse_args()
    set_seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("Meta Prefix Training")
    print("=" * 80)
    print("model_ckpt :", args.model_ckpt)
    print("pseudofile :", args.pseudofile)
    print("conall_path:", args.conall_path)
    print("nalist_path_meta:", args.nalist_path_meta)
    print("device     :", device)

    detector = build_detector(args, device)
    meta_adapter = build_meta_adapter(device)

    outer_optimizer = torch.optim.Adam(
        [meta_adapter.ln.weight, meta_adapter.ln.bias],
        lr=args.outer_lr
    )

    X_flat, nalist, pseudo_all = load_train_arrays(args)

    best_outer = float("inf")
    best_path = os.path.join(args.save_dir, "meta_adapter_best.pt")
    last_path = os.path.join(args.save_dir, "meta_adapter_last.pt")

    for epoch in range(1, args.meta_epochs + 1):
        inner_loss, outer_loss = train_one_epoch(
            meta_adapter=meta_adapter,
            detector=detector,
            X_flat=X_flat,
            nalist=nalist,
            pseudo_all=pseudo_all,
            device=device,
            outer_optimizer=outer_optimizer,
            args=args
        )

        print(f"[Epoch {epoch}/{args.meta_epochs}] inner={inner_loss:.4f} outer={outer_loss:.4f}")

        state = {
            "epoch": epoch,
            "adapter_state_dict": meta_adapter.state_dict(),
            "inner_loss": inner_loss,
            "outer_loss": outer_loss,
            "args": vars(args),
        }

        torch.save(state, last_path)

        if outer_loss < best_outer:
            best_outer = outer_loss
            torch.save(state, best_path)
            print(f"  -> saved best: {best_path}")

    print("=" * 80)
    print("Done")
    print("best outer loss:", best_outer)
    print("best ckpt      :", best_path)
    print("=" * 80)


if __name__ == "__main__":
    main()