import os
import random
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.data import DataLoader

from model import Model_V2_AllCNN
from prefix_hyper import (
    PrefixHyperNet,
    build_fixed_default_adapter,
    compute_prefix_stats,
    apply_adapter_with_generated_ln,
    normalize_video_feature_shape,
)


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


class PrefixHyperVideoDataset(data.Dataset):
    """
    concat feature + pseudo label + nalist를 이용해
    video 단위 episode 반환
    """
    def __init__(self, conall_path, pseudo_path, nalist_path, dtype="float32"):
        self.nalist = np.load(nalist_path)
        self.pseudo = np.load(pseudo_path).astype(np.float32)

        total_T = int(self.nalist[-1, 1])

        # 우선 (total_T, 10, 2048)로 시도
        try:
            self.con_all = np.memmap(
                conall_path,
                dtype=dtype,
                mode="r",
                shape=(total_T, 10, 2048)
            )
            self.feature_mode = "crop10"
        except Exception:
            self.con_all = np.memmap(
                conall_path,
                dtype=dtype,
                mode="r",
                shape=(total_T, 2048)
            )
            self.feature_mode = "flat2048"

        assert len(self.pseudo) == total_T, \
            f"Pseudo length mismatch: {len(self.pseudo)} vs total_T={total_T}"

        print(f"[Dataset] feature_mode={self.feature_mode}, con_all shape={self.con_all.shape}")
        print(f"[Dataset] pseudo shape={self.pseudo.shape}")
        print(f"[Dataset] nalist shape={self.nalist.shape}")

    def __len__(self):
        return len(self.nalist)

    def __getitem__(self, vid_idx):
        s, e = self.nalist[vid_idx]
        s, e = int(s), int(e)

        x_video_np = self.con_all[s:e]
        x_video_np = normalize_video_feature_shape(x_video_np)   # (T,2048)
        y_video_np = self.pseudo[s:e]                            # (T,)

        x_video = torch.from_numpy(np.asarray(x_video_np, dtype=np.float32))
        y_video = torch.from_numpy(np.asarray(y_video_np, dtype=np.float32))

        return x_video, y_video, vid_idx


def build_detector(args, device):
    model = Model_V2_AllCNN(args.feature_size, kernel_size=args.temporal_kernel).to(device)

    ckpt = torch.load(args.model_ckpt, map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        ckpt = ckpt["model"]
    ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}

    model.load_state_dict(ckpt, strict=True)
    freeze_module(model)
    print(f"[Detector] loaded from {args.model_ckpt}")
    return model


def train_one_epoch(hypernet, detector, base_adapter, loader, optimizer, device, args):
    hypernet.train()

    total_loss = 0.0
    total_suffix = 0.0
    total_prefix = 0.0
    total_reg = 0.0
    total_gate = 0.0
    num_used = 0

    gamma0 = base_adapter.ln.weight.detach()
    beta0 = base_adapter.ln.bias.detach()

    pbar = tqdm(loader, desc="Train Prefix Hyper", dynamic_ncols=True)

    for x_video, y_video, vid_idx in pbar:
        x_video = x_video[0].to(device).float()   # (T,D)
        y_video = y_video[0].to(device).float()   # (T,)
        T = x_video.shape[0]

        if T <= args.warmup_segments:
            continue

        prefix_len = min(args.warmup_segments, T)
        x_prefix = x_video[:prefix_len]
        x_suffix = x_video[prefix_len:]
        y_suffix = y_video[prefix_len:]

        if x_suffix.shape[0] == 0:
            continue

        stats = compute_prefix_stats(
            x_video_np=x_video.detach().cpu().numpy(),
            adapter=base_adapter,
            model=detector,
            device=device,
            warmup_segments=args.warmup_segments,
        )  # (7,)

        stats = stats.unsqueeze(0)  # (1,7)

        gamma, beta, g, dgamma, dbeta = hypernet(stats, gamma0, beta0)

        # prefix
        x_prefix_in = x_prefix.unsqueeze(0)  # (1,K,D)
        x_prefix_adapted = apply_adapter_with_generated_ln(
            x_prefix_in, base_adapter, gamma[0], beta[0]
        )
        _, logit_prefix = detector(x_prefix_adapted, return_logits=True)
        logit_prefix = logit_prefix[0, :, 0]

        # suffix
        x_suffix_in = x_suffix.unsqueeze(0)  # (1,T-K,D)
        x_suffix_adapted = apply_adapter_with_generated_ln(
            x_suffix_in, base_adapter, gamma[0], beta[0]
        )
        _, logit_suffix = detector(x_suffix_adapted, return_logits=True)
        logit_suffix = logit_suffix[0, :, 0]

        # label smoothing
        if args.label_smoothing:
            y_suffix_target = y_suffix * 0.9 + 0.05
        else:
            y_suffix_target = y_suffix

        loss_suffix = F.binary_cross_entropy_with_logits(
            logit_suffix, y_suffix_target, reduction="mean"
        )

        loss_prefix = F.softplus(logit_prefix).mean()

        loss_reg = (dgamma.pow(2).mean() + dbeta.pow(2).mean())

        # adaptation을 무조건 크게 하지 않도록 약한 gate penalty
        loss_gate = g.mean()

        loss = (
            loss_suffix
            + args.lambda_prefix * loss_prefix
            + args.lambda_reg * loss_reg
            + args.lambda_gate * loss_gate
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_suffix += loss_suffix.item()
        total_prefix += loss_prefix.item()
        total_reg += loss_reg.item()
        total_gate += loss_gate.item()
        num_used += 1

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "suf": f"{loss_suffix.item():.4f}",
            "pre": f"{loss_prefix.item():.4f}",
            "g": f"{g.item():.4f}",
        })

    if num_used == 0:
        return None

    return {
        "loss": total_loss / num_used,
        "loss_suffix": total_suffix / num_used,
        "loss_prefix": total_prefix / num_used,
        "loss_reg": total_reg / num_used,
        "loss_gate": total_gate / num_used,
    }


def main():
    parser = argparse.ArgumentParser("train_prefix_hyper")

    parser.add_argument("--model_ckpt", type=str, required=True)
    parser.add_argument("--pseudofile", type=str, required=True)
    parser.add_argument("--conall_path", type=str, required=True)
    parser.add_argument("--nalist_path", type=str, required=True)

    parser.add_argument("--save_dir", type=str, default="prefix_hyper_ckpt")
    parser.add_argument("--adapter_init_path", type=str, default="adapter_init.pt")

    parser.add_argument("--feature_size", type=int, default=2048)
    parser.add_argument("--temporal_kernel", type=int, default=5)
    parser.add_argument("--warmup_segments", type=int, default=5)

    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--delta_scale", type=float, default=0.1)

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument("--lambda_prefix", type=float, default=0.3)
    parser.add_argument("--lambda_reg", type=float, default=1e-3)
    parser.add_argument("--lambda_gate", type=float, default=5e-3)

    parser.add_argument("--label_smoothing", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    set_seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("Train Prefix Hyper")
    print("=" * 80)

    detector = build_detector(args, device)
    base_adapter = build_fixed_default_adapter(device, init_ckpt=args.adapter_init_path)

    hypernet = PrefixHyperNet(
        stats_dim=7,
        feat_dim=args.feature_size,
        hidden_dim=args.hidden_dim,
        delta_scale=args.delta_scale,
    ).to(device)

    optimizer = torch.optim.Adam(hypernet.parameters(), lr=args.lr)

    loader = DataLoader(
        PrefixHyperVideoDataset(
            conall_path=args.conall_path,
            pseudo_path=args.pseudofile,
            nalist_path=args.nalist_path,
        ),
        batch_size=1,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

    best_loss = 1e9
    best_path = os.path.join(args.save_dir, "prefix_hyper_best.pt")
    last_path = os.path.join(args.save_dir, "prefix_hyper_last.pt")

    for epoch in range(1, args.epochs + 1):
        metrics = train_one_epoch(
            hypernet=hypernet,
            detector=detector,
            base_adapter=base_adapter,
            loader=loader,
            optimizer=optimizer,
            device=device,
            args=args,
        )

        if metrics is None:
            print(f"[Epoch {epoch}] no usable videos")
            continue

        print(
            f"[Epoch {epoch}/{args.epochs}] "
            f"loss={metrics['loss']:.4f} "
            f"suffix={metrics['loss_suffix']:.4f} "
            f"prefix={metrics['loss_prefix']:.4f} "
            f"reg={metrics['loss_reg']:.6f} "
            f"gate={metrics['loss_gate']:.4f}"
        )

        save_obj = {
            "hyper_state_dict": hypernet.state_dict(),
            "stats_dim": 7,
            "feat_dim": args.feature_size,
            "hidden_dim": args.hidden_dim,
            "delta_scale": args.delta_scale,
            "epoch": epoch,
            "metrics": metrics,
            "args": vars(args),
        }

        torch.save(save_obj, last_path)

        if metrics["loss"] < best_loss:
            best_loss = metrics["loss"]
            torch.save(save_obj, best_path)
            print(f"  -> saved best: {best_path}")

    print("=" * 80)
    print("Done")
    print(f"Best ckpt: {best_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()