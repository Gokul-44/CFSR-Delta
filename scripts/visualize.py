"""
visualize.py — Generate visual comparison figures for all benchmark images.

Produces side-by-side comparison strips (LR | Baseline | Delta | HR)
for each image in a specified benchmark dataset.

Usage:
    python scripts/visualize.py --backbone_weights model_zoo/CFSR_x4.pth \
        --refine_weights checkpoints/refine_best_x4.pth \
        --dataset Set5 --output_dir results/figures
"""

import os
import sys
import argparse

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.cfsr import load_cfsr_model
from src.models.cfsr_delta import load_delta_model
from src.metrics.sr_metrics import calc_psnr
from src.utils.visualization import (
    load_image, to_tensor, to_numpy, create_comparison, save_comparison_figure,
)


def main():
    parser = argparse.ArgumentParser(description="Generate SR comparison figures")
    parser.add_argument("--backbone_weights", type=str, required=True)
    parser.add_argument("--refine_weights", type=str, required=True)
    parser.add_argument("--scale", type=int, default=4, choices=[2, 3, 4])
    parser.add_argument("--dataset", type=str, default="Set5")
    parser.add_argument("--benchmark_dir", type=str, default="datasets/benchmark")
    parser.add_argument("--output_dir", type=str, default="results/figures")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load models
    print("Loading baseline model...")
    baseline_model, _ = load_cfsr_model(args.scale, args.backbone_weights, args.device)

    print("Loading delta model...")
    delta_model, _ = load_delta_model(
        args.scale, args.backbone_weights, args.refine_weights,
        device=args.device,
    )

    # Process images
    hr_dir = os.path.join(args.benchmark_dir, args.dataset, "HR")
    lr_dir = os.path.join(args.benchmark_dir, args.dataset, "LR_bicubic", f"X{args.scale}")

    hr_files = sorted([f for f in os.listdir(hr_dir) if f.lower().endswith(".png")])
    print(f"\nProcessing {len(hr_files)} images from {args.dataset}...\n")

    for fname in hr_files:
        base_name = fname.replace(".png", "")
        lr_name = f"{base_name}x{args.scale}.png"
        lr_path = os.path.join(lr_dir, lr_name)
        hr_path = os.path.join(hr_dir, fname)

        if not os.path.exists(lr_path):
            print(f"  [SKIP] LR not found: {lr_name}")
            continue

        # Load images
        lr_img = load_image(lr_path)
        hr_img = load_image(hr_path)
        lr_tensor = to_tensor(lr_img).to(args.device)

        # Inference
        with torch.no_grad():
            baseline_sr = to_numpy(baseline_model(lr_tensor))
            delta_sr = to_numpy(delta_model(lr_tensor))

        # Compute PSNR
        h = min(baseline_sr.shape[0], hr_img.shape[0])
        w = min(baseline_sr.shape[1], hr_img.shape[1])
        bp = calc_psnr(baseline_sr[:h, :w], hr_img[:h, :w], args.scale)
        dp = calc_psnr(delta_sr[:h, :w], hr_img[:h, :w], args.scale)

        # Create and save figure
        fig = create_comparison(
            lr_img, baseline_sr, delta_sr, hr_img, args.scale, bp, dp
        )
        out_path = os.path.join(args.output_dir, f"comparison_{base_name}.png")
        save_comparison_figure(fig, out_path)
        print(f"  {fname}: Base={bp:.4f} Delta={dp:.4f} Gain={dp-bp:+.4f} → {out_path}")

    print(f"\nDone. Figures saved to {args.output_dir}")


if __name__ == "__main__":
    main()
