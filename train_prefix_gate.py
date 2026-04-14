import os
import copy
import random
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from gate_dataset import PrefixGateVideoDataset
from model import Model_V2_AllCNN
from adapter import ResidualAdapter2048
from prefix_gate import PrefixGateNet


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def freeze_module(module):
    module.eval()
    for p in module.parameters():
        p.requires_grad = False


def enable_ln_only(adapter):
    for p in adapter.parameters():
        p.requires_grad = False
    adapter.ln.weight.requires_grad = True
    adapter.ln.bias.requires_grad = True


def normalize_video_feature_shape(x_video_np):
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
    x_video_np = normalize_video_feature_shape(x_video_np)
    x_video = torch.from_numpy(x_video_np).float().to(device)

    T = x_video.shape[0]
    prefix_len = min(warmup_segments, T)
    x_prefix = x_video[:prefix_len]

    if prefix_len == 0:
        return torch.zeros(7, device=device)

    x_prefix_in = x_prefix.unsqueeze(0)

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



def build_fixed_default_adapter(device, init_ckpt="adapter_init.pt"):
    adapter = ResidualAdapter2048(d=2048, use_ln=True).to(device)

    if os.path.exists(init_ckpt):
        adapter.load_state_dict(torch.load(init_ckpt, map_location=device))
        print(f"[Adapter Init] loaded fixed init from {init_ckpt}")
    else:
        torch.save(adapter.state_dict(), init_ckpt)
        print(f"[Adapter Init] saved new fixed init to {init_ckpt}")

    adapter.eval()
    return adapter


def build_detector(args, device):
    model = Model_V2_AllCNN(args.feature_size).to(device)

    ckpt = torch.load(args.model_ckpt, map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        ckpt = ckpt["model"]
    ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}

    model.load_state_dict(ckpt, strict=True)
    freeze_module(model)
    print(f"[Model] loaded from {args.model_ckpt}")
    return model


def get_valid_video(features, labels, masks, lengths):
    T = int(lengths[0].item())
    x = features[:, :T, :]
    y = labels[:, :T]
    m = masks[:, :T]
    return x, y, m, T


def adapt_prefix_once(
    x_video,               # (T, D), torch
    base_adapter,
    model,
    warmup_segments=5,
    tea_lr=1e-2,
    tea_steps=30,
):
    adapter_ep = copy.deepcopy(base_adapter)
    adapter_ep.train()
    enable_ln_only(adapter_ep)

    prefix_len = min(warmup_segments, x_video.shape[0])
    x_prefix = x_video[:prefix_len]

    if prefix_len == 0:
        return adapter_ep

    optimizer = torch.optim.Adam(
        [adapter_ep.ln.weight, adapter_ep.ln.bias],
        lr=tea_lr
    )

    for _ in range(tea_steps):
        x_prefix_in = x_prefix.unsqueeze(0)  # (1, K, D)
        x_2048 = adapter_ep(x_prefix_in)
        _, logit = model(x_2048, return_logits=True)
        logit = logit[0, :, 0]

        loss = F.softplus(logit).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return adapter_ep


def suffix_bce_loss(
    x_video,               # (T, D), torch
    y_video,               # (T,), torch
    base_adapter,
    model,
    warmup_segments=5,
    label_smoothing=True,
):
    T = x_video.shape[0]
    prefix_len = min(warmup_segments, T)

    if T <= prefix_len:
        return None

    x_suffix = x_video[prefix_len:]
    y_suffix = y_video[prefix_len:]

    x_suffix_in = x_suffix.unsqueeze(0)
    with torch.no_grad():
        x_2048 = base_adapter(x_suffix_in)
        _, logit = model(x_2048, return_logits=True)
        logits = logit[0, :, 0]

    if label_smoothing:
        y_suffix = y_suffix * 0.9 + 0.05

    loss = F.binary_cross_entropy_with_logits(logits, y_suffix, reduction="mean")
    return loss.item()


def build_gate_dataset(loader, base_adapter, model, device, args):
    stats_list = []
    target_list = []
    meta_rows = []

    pbar = tqdm(loader, desc="Build Gate Dataset", dynamic_ncols=True)

    for x_video, y_video, vid_idx in pbar:
        # batch_size=1 이므로 앞 차원 제거
        x_video = x_video[0].to(device).float()   # (T, D)
        y_video = y_video[0].to(device).float()   # (T,)
        vid_idx = int(vid_idx.item())

        T = x_video.shape[0]
        if T <= args.warmup_segments:
            continue

        x_video_np = x_video.detach().cpu().numpy()

        stats = compute_prefix_stats(
            x_video_np=x_video_np,
            adapter=base_adapter,
            model=model,
            device=device,
            warmup_segments=args.warmup_segments
        )

        # baseline suffix loss
        base_loss = suffix_bce_loss(
            x_video=x_video,
            y_video=y_video,
            base_adapter=base_adapter,
            model=model,
            warmup_segments=args.warmup_segments,
            label_smoothing=args.label_smoothing,
        )

        if base_loss is None:
            continue

        # adapted suffix loss
        adapted_adapter = adapt_prefix_once(
            x_video=x_video,
            base_adapter=base_adapter,
            model=model,
            warmup_segments=args.warmup_segments,
            tea_lr=args.tea_lr,
            tea_steps=args.tea_steps_per_video,
        )

        adapt_loss = suffix_bce_loss(
            x_video=x_video,
            y_video=y_video,
            base_adapter=adapted_adapter,
            model=model,
            warmup_segments=args.warmup_segments,
            label_smoothing=args.label_smoothing,
        )

        # TTA가 실제로 도움이 되었는지
        target = 1.0 if (adapt_loss + args.improve_margin < base_loss) else 0.0

        stats_list.append(stats.detach().cpu().numpy())
        target_list.append(target)

        meta_rows.append({
            "vid_idx": vid_idx,
            "base_loss": base_loss,
            "adapt_loss": adapt_loss,
            "delta": adapt_loss - base_loss,
            "target": target,
        })

    X = np.stack(stats_list, axis=0).astype(np.float32)
    y = np.array(target_list, dtype=np.float32)

    print(f"[Gate Dataset] num_videos = {len(X)}")
    print(f"[Gate Dataset] positive ratio = {y.mean():.4f}")

    return X, y, meta_rows


def train_gate_model(X, y, device, args):
    N = len(X)
    idx = np.random.permutation(N)

    n_train = int(0.8 * N)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:]

    X_train = X[train_idx]
    y_train = y[train_idx]
    X_val = X[val_idx]
    y_val = y[val_idx]

    feat_mean = X_train.mean(axis=0, keepdims=True)
    feat_std = X_train.std(axis=0, keepdims=True) + 1e-6

    X_train = (X_train - feat_mean) / feat_std
    X_val = (X_val - feat_mean) / feat_std

    X_train = torch.from_numpy(X_train).float().to(device)
    y_train = torch.from_numpy(y_train).float().to(device)

    X_val = torch.from_numpy(X_val).float().to(device)
    y_val = torch.from_numpy(y_val).float().to(device)

    gate = PrefixGateNet(in_dim=X.shape[1], hidden_dim=args.hidden_dim).to(device)

    n_pos = max(float(y_train.sum().item()), 1.0)
    n_neg = max(float(len(y_train) - y_train.sum().item()), 1.0)
    pos_weight = torch.tensor([n_neg / n_pos], device=device)

    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(gate.parameters(), lr=args.gate_lr)

    best_val = 1e9
    best_state = None

    for epoch in range(1, args.gate_epochs + 1):
        gate.train()
        train_logit = gate(X_train)
        train_loss = criterion(train_logit, y_train)

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        gate.eval()
        with torch.no_grad():
            val_logit = gate(X_val)
            val_loss = criterion(val_logit, y_val)

            train_prob = torch.sigmoid(train_logit)
            val_prob = torch.sigmoid(val_logit)

            train_acc = ((train_prob >= 0.5).float() == y_train).float().mean().item()
            val_acc = ((val_prob >= 0.5).float() == y_val).float().mean().item()

        print(
            f"[Gate Epoch {epoch}/{args.gate_epochs}] "
            f"train_loss={train_loss.item():.4f} "
            f"val_loss={val_loss.item():.4f} "
            f"train_acc={train_acc:.4f} "
            f"val_acc={val_acc:.4f}"
        )

        if val_loss.item() < best_val:
            best_val = val_loss.item()
            best_state = {
                "gate_state_dict": gate.state_dict(),
                "feat_mean": feat_mean.squeeze(0),
                "feat_std": feat_std.squeeze(0),
                "in_dim": X.shape[1],
                "hidden_dim": args.hidden_dim,
                "best_val_loss": best_val,
            }

    return best_state


def main():
    #parser = argparse.ArgumentParser("train_prefix_gate")
    
    import option
    args = option.parser.parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("Train Prefix Gate")
    print("=" * 80)

    model = build_detector(args, device)
    base_adapter = build_fixed_default_adapter(device, init_ckpt=args.adapter_init_path)

    loader = DataLoader(
        PrefixGateVideoDataset(
            conall_path=args.conall_path,
            pseudo_path=args.pseudofile,
            nalist_path=args.nalist_path,
        ),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

    X, y, meta_rows = build_gate_dataset(
        loader=loader,
        base_adapter=base_adapter,
        model=model,
        device=device,
        args=args
    )

    best_state = train_gate_model(X, y, device, args)

    save_obj = {
        **best_state,
        "args": vars(args),
    }
    torch.save(save_obj, args.save_path)

    print(f"[Saved] gate checkpoint -> {args.save_path}")


if __name__ == "__main__":
    main()