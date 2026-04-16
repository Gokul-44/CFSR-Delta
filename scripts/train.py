"""
train.py — Training script for CFSR-Delta RefineNet.

Trains the RefineNet post-processor with a frozen CFSR backbone using
peak-capture strategy: evaluate frequently, save the best checkpoint,
stop when PSNR stops improving.

Usage:
    # Quick validation run (2000 iters):
    python scripts/train.py --backbone_weights model_zoo/CFSR_x4.pth \
        --data_dir datasets/DF2K/HR --validation_mode

    # Full training:
    python scripts/train.py --backbone_weights model_zoo/CFSR_x4.pth \
        --data_dir datasets/DF2K/HR --max_iters 10000
"""

import os
import sys
import copy
import random
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.cfsr import CFSR
from src.models.refine_net import RefineNet
from src.models.cfsr_delta import CFSRDelta
from src.data.df2k_dataset import DF2KDataset
from src.metrics.sr_metrics import evaluate_dataset


def train(args):
    """Main training loop."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Build backbone
    print("\n[1/4] Loading CFSR backbone...")
    backbone = CFSR(dim=48, depths=(6, 6), dw_size=9, mlp_ratio=2, scale=args.scale)
    backbone.merge_all()

    state_dict = torch.load(args.backbone_weights, map_location="cpu", weights_only=True)
    if "params" in state_dict:
        state_dict = state_dict["params"]
    backbone.load_state_dict(state_dict, strict=True)
    backbone = backbone.to(device).eval()
    print(f"  Backbone: {backbone.param_count():,} params (frozen)")

    # Build RefineNet
    print("[2/4] Initializing RefineNet...")
    torch.manual_seed(args.seed)
    refine = RefineNet(hidden_channels=args.hidden_ch, init_scale=args.init_scale)
    print(f"  RefineNet: {refine.param_count():,} params (trainable)")

    # Load checkpoint if provided
    if args.refine_checkpoint and os.path.exists(args.refine_checkpoint):
        ckpt = torch.load(args.refine_checkpoint, map_location="cpu", weights_only=False)
        if "refine" in ckpt:
            refine.load_state_dict(ckpt["refine"])
        else:
            refine.load_state_dict(ckpt)
        print(f"  Loaded checkpoint: {args.refine_checkpoint}")

    model = CFSRDelta(backbone, refine).to(device)

    # Dataset
    print("[3/4] Loading dataset...")
    dataset = DF2KDataset(args.data_dir, args.scale, args.patch_size)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    print(f"  {len(dataset.hr_paths)} HR images, virtual epoch: {len(dataset)}")

    # Optimizer
    optimizer = optim.Adam(refine.parameters(), lr=args.lr, betas=(0.9, 0.99))
    criterion = nn.L1Loss()

    # Initial evaluation
    print("[4/4] Initial evaluation on Set5...")
    init_result = evaluate_dataset(model, "Set5", args.scale, args.benchmark_dir, str(device))
    baseline_psnr = init_result["psnr"]
    print(f"  Starting PSNR: {baseline_psnr:.4f} dB")

    # Training state
    best_psnr = baseline_psnr or 0.0
    best_state = copy.deepcopy(refine.state_dict())
    best_iter = 0
    non_improve = 0
    os.makedirs(args.save_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"  Training: max {args.max_iters} iters, eval every {args.eval_interval}")
    print(f"  LR={args.lr}, patience={args.patience}")
    print(f"{'=' * 60}\n")

    model.train()
    it = 0
    stopped = False

    for lr_batch, hr_batch in loader:
        if it >= args.max_iters or stopped:
            break

        lr_batch = lr_batch.to(device)
        hr_batch = hr_batch.to(device)

        loss = criterion(model(lr_batch), hr_batch)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(refine.parameters(), args.grad_clip)
        optimizer.step()
        it += 1

        # Evaluate
        if it % args.eval_interval == 0:
            result = evaluate_dataset(
                model, "Set5", args.scale, args.benchmark_dir, str(device)
            )
            model.train()
            psnr = result["psnr"]

            if psnr is None:
                continue

            gain = psnr - (baseline_psnr or psnr)
            mark = ""
            if psnr > best_psnr:
                best_psnr = psnr
                best_iter = it
                best_state = copy.deepcopy(refine.state_dict())
                mark = " *** BEST"
                non_improve = 0
            else:
                non_improve += 1

            print(
                f"  [iter {it:>6d}] PSNR={psnr:.4f} ({gain:+.4f} dB) "
                f"loss={loss.item():.6f}{mark}"
            )

            if non_improve >= args.patience:
                print(f"\n  [EARLY STOP] {args.patience} non-improvements. Peak at iter {best_iter}.")
                stopped = True

    # Save best checkpoint
    save_path = os.path.join(args.save_dir, f"refine_best_x{args.scale}.pth")
    torch.save({
        "refine": best_state,
        "psnr": float(best_psnr),
        "iter": int(best_iter),
        "config": vars(args),
    }, save_path)

    print(f"\n{'=' * 60}")
    print(f"  Training complete.")
    print(f"  Best PSNR: {best_psnr:.4f} dB at iter {best_iter}")
    print(f"  Saved: {save_path}")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(description="Train CFSR-Delta RefineNet")

    # Required
    parser.add_argument("--backbone_weights", type=str, required=True,
                        help="Path to official CFSR pretrained weights")
    parser.add_argument("--data_dir", type=str, default="datasets/DF2K/HR",
                        help="Path to DF2K training images (HR folder)")

    # Model
    parser.add_argument("--scale", type=int, default=4, choices=[2, 3, 4])
    parser.add_argument("--hidden_ch", type=int, default=32)
    parser.add_argument("--init_scale", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=69)

    # Training
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--max_iters", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--patch_size", type=int, default=64)
    parser.add_argument("--grad_clip", type=float, default=0.5)
    parser.add_argument("--num_workers", type=int, default=0)

    # Evaluation
    parser.add_argument("--eval_interval", type=int, default=50)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--benchmark_dir", type=str, default="datasets/benchmark")

    # Checkpoints
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--refine_checkpoint", type=str, default=None)

    # Quick mode
    parser.add_argument("--validation_mode", action="store_true",
                        help="Short 2000-iter run with frequent eval")

    args = parser.parse_args()

    if args.validation_mode:
        args.max_iters = 2000
        args.eval_interval = 25
        args.patience = 5

    train(args)


if __name__ == "__main__":
    main()
