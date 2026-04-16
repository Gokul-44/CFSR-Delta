"""
inference.py — Single-image super-resolution inference.

Runs the CFSR or CFSR-Delta model on a single input image and saves
the super-resolved output.

Usage:
    # Baseline inference:
    python scripts/inference.py --input photo.png --output sr.png \
        --backbone_weights model_zoo/CFSR_x4.pth

    # Delta inference:
    python scripts/inference.py --input photo.png --output sr_delta.png \
        --backbone_weights model_zoo/CFSR_x4.pth \
        --refine_weights checkpoints/refine_best_x4.pth

    # With comparison output:
    python scripts/inference.py --input photo.png --compare \
        --backbone_weights model_zoo/CFSR_x4.pth \
        --refine_weights checkpoints/refine_best_x4.pth
"""

import os
import sys
import argparse

import numpy as np
import cv2
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.cfsr import load_cfsr_model
from src.models.cfsr_delta import load_delta_model
from src.utils.visualization import (
    load_image, to_tensor, to_numpy, create_comparison, save_comparison_figure,
)


def main():
    parser = argparse.ArgumentParser(description="Single-image super-resolution")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to input LR image")
    parser.add_argument("--output", type=str, default="output_sr.png",
                        help="Path to save SR output")
    parser.add_argument("--backbone_weights", type=str, required=True,
                        help="Path to CFSR pretrained weights")
    parser.add_argument("--refine_weights", type=str, default=None,
                        help="Path to RefineNet weights (enables delta mode)")
    parser.add_argument("--scale", type=int, default=4, choices=[2, 3, 4])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--compare", action="store_true",
                        help="Save side-by-side comparison (requires --refine_weights)")
    parser.add_argument("--hr", type=str, default=None,
                        help="Optional HR ground truth for comparison")
    args = parser.parse_args()

    # Load input
    print(f"Loading input: {args.input}")
    lr_img = load_image(args.input)
    lr_tensor = to_tensor(lr_img).to(args.device)

    # Load model(s)
    print("Loading model...")
    baseline_model, _ = load_cfsr_model(args.scale, args.backbone_weights, args.device)

    with torch.no_grad():
        baseline_sr = to_numpy(baseline_model(lr_tensor))

    if args.refine_weights:
        delta_model, counts = load_delta_model(
            args.scale, args.backbone_weights, args.refine_weights,
            device=args.device,
        )
        with torch.no_grad():
            delta_sr = to_numpy(delta_model(lr_tensor))
        sr_output = delta_sr
        print(f"  Using delta model ({counts['trainable']:,} trainable params)")
    else:
        sr_output = baseline_sr
        delta_sr = None
        print("  Using baseline model")

    # Save SR output
    sr_bgr = cv2.cvtColor(
        np.clip(sr_output * 255 + 0.5, 0, 255).astype(np.uint8),
        cv2.COLOR_RGB2BGR,
    )
    cv2.imwrite(args.output, sr_bgr)
    print(f"Saved: {args.output} ({sr_output.shape[1]}×{sr_output.shape[0]})")

    # Comparison figure
    if args.compare and delta_sr is not None:
        hr_img = load_image(args.hr) if args.hr else None
        if hr_img is None:
            hr_img = np.zeros_like(baseline_sr)

        fig = create_comparison(lr_img, baseline_sr, delta_sr, hr_img, args.scale)
        comp_path = args.output.replace(".png", "_comparison.png")
        save_comparison_figure(fig, comp_path)
        print(f"Saved comparison: {comp_path}")


if __name__ == "__main__":
    main()
