import os
import random
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.utils.data as data
from torch.utils.data import DataLoader

from model import Model_V2_AllCNN
from safe_meta_policy import (
    SafeMetaPolicyNet,
    policy_inner_update,
    compute_safe_outer_loss,
)
from prefix_hyper import (
    build_fixed_default_adapter,
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


class PrefixPolicyVideoDataset(data.Dataset):
    def __init__(self, conall_path, pseudo_path, nalist_path, dtype="float32"):
        self.nalist = np.load(nalist_path)
        self.pseudo = np.load(pseudo_path).astype(np.float32)

        total_T = int(self.nalist[-1, 1])

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
        x_video_np = normalize_video_feature_shape(x_video_np)
        y_video_np = self.pseudo[s:e]

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


def train_one_epoch(policy_net, detector, base_adapter, loader, optimizer, device, args):
    policy_net.train()

    total_loss = 0.0
    total_norm = 0.0
    total_preserve = 0.0
    total_rank = 0.0
    total_gate = 0.0
    total_used = 0

    gate_vals = []
    alpha_vals = []
    loss_gate_vals = []
    loss_alpha_vals = []
    dgamma_vals = []
    dbeta_vals = []

    pbar = tqdm(loader, desc="Train Safe Meta Policy", dynamic_ncols=True)

    for x_video, y_video, vid_idx in pbar:
        x_video = x_video[0].to(device).float()   # (T,D)
        y_video = y_video[0].to(device).float()   # (T,)
        T = x_video.shape[0]

        if T <= args.warmup_segments:
            continue

        x_video_np = x_video.detach().cpu().numpy()

        gamma_adapt, beta_adapt, inner_info = policy_inner_update(
            x_video_np=x_video_np,
            base_adapter=base_adapter,
            model=detector,
            policy_net=policy_net,
            device=device,
            warmup_segments=args.warmup_segments,
            inner_steps=args.inner_steps,
            create_graph=True,
            gate_threshold=None,   # train에서는 hard skip 없음
        )

        outer = compute_safe_outer_loss(
            x_video=x_video,
            y_video=y_video,
            base_adapter=base_adapter,
            model=detector,
            gamma_adapt=gamma_adapt,
            beta_adapt=beta_adapt,
            warmup_segments=args.warmup_segments,
            normal_q=args.normal_q,
            anom_q=args.anom_q,
            tail_gap_score=args.tail_gap_score,
            tail_gap_pseudo=args.tail_gap_pseudo,
            preserve_margin=args.preserve_margin,
            rank_margin=args.rank_margin,
            min_keep_normal=args.min_keep_normal,
            min_keep_anom=args.min_keep_anom,
        )

        if outer is None:
            continue

        loss_norm, loss_preserve, loss_rank, outer_debug = outer

        g_tensor = inner_info["g_tensor"]
        alpha_tensor = inner_info["alpha_tensor"]
        '''
        loss_gate = g_tensor.mean()
        loss_alpha = (alpha_tensor / max(args.lr_max, 1e-8)).mean()
        '''

        if args.gate_reg_mode == "l1":
            loss_gate = g_tensor.mean()
        elif args.gate_reg_mode == "target_l2":
            loss_gate = ((g_tensor - args.target_gate) ** 2).mean()
        else:
            raise ValueError(f"Unknown gate_reg_mode: {args.gate_reg_mode}")

        loss_alpha = (alpha_tensor / max(args.lr_max, 1e-8)).mean()


        g_val = inner_info["gate"]
        alpha_val = inner_info["alpha"]
        gate_vals.append(float(g_val))
        alpha_vals.append(float(alpha_val))
        loss_gate_vals.append(float(loss_gate.item()))
        loss_alpha_vals.append(float(loss_alpha.item()))

        if "dgamma_norm" in inner_info:
            dgamma_vals.append(float(inner_info["dgamma_norm"]))
        if "dbeta_norm" in inner_info:
            dbeta_vals.append(float(inner_info["dbeta_norm"]))

        loss = (
            args.lambda_norm * loss_norm
            + args.lambda_preserve * loss_preserve
            + args.lambda_rank * loss_rank
            + args.lambda_gate * loss_gate
            + args.lambda_alpha * loss_alpha
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_norm += loss_norm.item()
        total_preserve += loss_preserve.item()
        total_rank += loss_rank.item()
        total_gate += g_val
        total_used += 1

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "norm": f"{loss_norm.item():.4f}",
            "pres": f"{loss_preserve.item():.4f}",
            "rank": f"{loss_rank.item():.4f}",
            "g": f"{g_val:.4f}",
            "a": f"{alpha_val:.5f}",
        })

    if total_used == 0:
        return None

    gate_arr = np.asarray(gate_vals, dtype=np.float64) if len(gate_vals) else np.zeros(1)
    alpha_arr = np.asarray(alpha_vals, dtype=np.float64) if len(alpha_vals) else np.zeros(1)
    loss_gate_arr = np.asarray(loss_gate_vals, dtype=np.float64) if len(loss_gate_vals) else np.zeros(1)
    loss_alpha_arr = np.asarray(loss_alpha_vals, dtype=np.float64) if len(loss_alpha_vals) else np.zeros(1)
    dgamma_arr = np.asarray(dgamma_vals, dtype=np.float64) if len(dgamma_vals) else np.zeros(1)
    dbeta_arr = np.asarray(dbeta_vals, dtype=np.float64) if len(dbeta_vals) else np.zeros(1)

    return {
        "loss": total_loss / total_used,
        "loss_norm": total_norm / total_used,
        "loss_preserve": total_preserve / total_used,
        "loss_rank": total_rank / total_used,
        "gate_mean": total_gate / total_used,

        "gate_median": float(np.median(gate_arr)),
        "gate_min": float(np.min(gate_arr)),
        "gate_max": float(np.max(gate_arr)),
        "alpha_mean": float(np.mean(alpha_arr)),
        "alpha_median": float(np.median(alpha_arr)),
        "alpha_min": float(np.min(alpha_arr)),
        "alpha_max": float(np.max(alpha_arr)),

        "gate_ratio_gt_1e3": float((gate_arr > 1e-3).mean()),
        "gate_ratio_gt_1e2": float((gate_arr > 1e-2).mean()),
        "gate_ratio_gt_5e2": float((gate_arr > 5e-2).mean()),
        "gate_ratio_gt_2e1": float((gate_arr > 2e-1).mean()),

        "loss_gate_mean": float(np.mean(loss_gate_arr)),
        "loss_alpha_mean": float(np.mean(loss_alpha_arr)),

        "dgamma_norm_mean": float(np.mean(dgamma_arr)),
        "dbeta_norm_mean": float(np.mean(dbeta_arr)),
    }

def main():
    parser = argparse.ArgumentParser("train_safe_meta_policy")

    parser.add_argument("--model_ckpt", type=str, required=True)
    parser.add_argument("--pseudofile", type=str, required=True)
    parser.add_argument("--conall_path", type=str, required=True)
    parser.add_argument("--nalist_path", type=str, required=True)

    parser.add_argument("--save_dir", type=str, default="safe_meta_policy_ckpt")
    parser.add_argument("--adapter_init_path", type=str, default="adapter_init.pt")

    parser.add_argument("--feature_size", type=int, default=2048)
    parser.add_argument("--temporal_kernel", type=int, default=5)
    parser.add_argument("--warmup_segments", type=int, default=5)

    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--lr_max", type=float, default=1e-2)
    parser.add_argument("--inner_steps", type=int, default=3)

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument("--normal_quantile", type=float, default=0.1)
    parser.add_argument("--anom_quantile", type=float, default=0.3)
    parser.add_argument("--preserve_margin", type=float, default=0.02)
    parser.add_argument("--rank_margin", type=float, default=0.05)
    parser.add_argument("--min_keep", type=int, default=4)

    parser.add_argument("--lambda_norm", type=float, default=1.0)
    parser.add_argument("--lambda_preserve", type=float, default=1.5)
    parser.add_argument("--lambda_rank", type=float, default=0.5)
    parser.add_argument("--lambda_gate", type=float, default=0.2)
    parser.add_argument("--lambda_alpha", type=float, default=0.05)
    
    parser.add_argument("--normal_q", type=float, default=0.40)
    parser.add_argument("--anom_q", type=float, default=0.10)
    parser.add_argument("--tail_gap_score", type=float, default=0.03)
    parser.add_argument("--tail_gap_pseudo", type=float, default=0.02)
    parser.add_argument("--min_keep_normal", type=int, default=4)
    parser.add_argument("--min_keep_anom", type=int, default=2)


    parser.add_argument("--run_name", type=str, default="safe_meta_policy")
    parser.add_argument("--gate_init_bias", type=float, default=-0.5)
    parser.add_argument("--gate_reg_mode", type=str, default="l1",
                        choices=["l1", "target_l2"])
    parser.add_argument("--target_gate", type=float, default=0.1)
    
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    set_seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("Train Safe Meta Policy")
    print("=" * 80)

    detector = build_detector(args, device)
    base_adapter = build_fixed_default_adapter(device, init_ckpt=args.adapter_init_path)

    policy_net = SafeMetaPolicyNet(
        warmup_segments=args.warmup_segments,
        stats_dim=7,
        hidden_dim=args.hidden_dim,
        lr_max=args.lr_max,
        gate_init_bias=args.gate_init_bias,
    ).to(device)

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=args.lr)

    loader = DataLoader(
        PrefixPolicyVideoDataset(
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
    #best_path = os.path.join(args.save_dir, "safe_meta_policy_best.pt")
    #last_path = os.path.join(args.save_dir, "safe_meta_policy_last.pt")
    best_path = os.path.join(args.save_dir, f"{args.run_name}_best.pt")
    last_path = os.path.join(args.save_dir, f"{args.run_name}_last.pt")
    for epoch in range(1, args.epochs + 1):
        metrics = train_one_epoch(
            policy_net=policy_net,
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
            f"norm={metrics['loss_norm']:.4f} "
            f"preserve={metrics['loss_preserve']:.4f} "
            f"rank={metrics['loss_rank']:.4f} "
            f"gate_mean={metrics['gate_mean']:.6f} "
            f"gate_med={metrics['gate_median']:.6f} "
            f"gate_min={metrics['gate_min']:.3e} "
            f"gate_max={metrics['gate_max']:.3e} "
            f"g>1e-3={metrics['gate_ratio_gt_1e3']:.3f} "
            f"g>1e-2={metrics['gate_ratio_gt_1e2']:.3f} "
            f"g>5e-2={metrics['gate_ratio_gt_5e2']:.3f} "
            f"g>0.2={metrics['gate_ratio_gt_2e1']:.3f} "
            f"alpha_mean={metrics['alpha_mean']:.6f} "
            f"alpha_med={metrics['alpha_median']:.6f} "
            f"Lgate={metrics['loss_gate_mean']:.6f} "
            f"Lalpha={metrics['loss_alpha_mean']:.6f} "
            f"|dg|={metrics['dgamma_norm_mean']:.6f} "
            f"|db|={metrics['dbeta_norm_mean']:.6f}"
        )

        save_obj = {
            "policy_state_dict": policy_net.state_dict(),
            "warmup_segments": args.warmup_segments,
            "stats_dim": 7,
            "hidden_dim": args.hidden_dim,
            "lr_max": args.lr_max,
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