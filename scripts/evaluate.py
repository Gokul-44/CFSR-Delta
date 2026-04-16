"""
evaluate.py — Benchmark evaluation for CFSR baseline and delta models.

Evaluates on standard SR benchmarks (Set5, Set14, B100, Urban100, Manga109)
and prints a comparison table with PSNR/SSIM gains.

Usage:
    # Evaluate baseline only:
    python scripts/evaluate.py --model baseline --backbone_weights model_zoo/CFSR_x4.pth

    # Evaluate delta only:
    python scripts/evaluate.py --model delta --backbone_weights model_zoo/CFSR_x4.pth \
        --refine_weights checkpoints/refine_best_x4.pth

    # Compare both:
    python scripts/evaluate.py --model both --backbone_weights model_zoo/CFSR_x4.pth \
        --refine_weights checkpoints/refine_best_x4.pth
"""

import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.cfsr import load_cfsr_model
from src.models.cfsr_delta import load_delta_model
from src.metrics.sr_metrics import evaluate_all_benchmarks


DATASETS = ["Set5", "Set14", "B100", "Urban100", "Manga109"]


def print_results(name: str, results: dict, params: int):
    """Print formatted results table."""
    print(f"\n{'=' * 50}")
    print(f"  {name} — {params:,} parameters")
    print(f"{'=' * 50}")
    print(f"  {'Dataset':<12} {'PSNR (dB)':<12} {'SSIM':<10}")
    print(f"  {'-' * 34}")
    for ds in DATASETS:
        if ds in results:
            print(f"  {ds:<12} {results[ds]['psnr']:<12.4f} {results[ds]['ssim']:<10.4f}")


def print_comparison(baseline_results: dict, delta_results: dict):
    """Print side-by-side comparison table."""
    print(f"\n{'=' * 80}")
    print(f"  Baseline vs Delta Comparison")
    print(f"{'=' * 80}")
    print(f"  {'Dataset':<12} {'Base PSNR':<12} {'Delta PSNR':<12} "
          f"{'Gain':<10} {'Base SSIM':<12} {'Delta SSIM':<12}")
    print(f"  {'-' * 70}")

    total_gain = 0
    count = 0
    for ds in DATASETS:
        if ds in baseline_results and ds in delta_results:
            bp = baseline_results[ds]["psnr"]
            dp = delta_results[ds]["psnr"]
            bs = baseline_results[ds]["ssim"]
            ds_ssim = delta_results[ds]["ssim"]
            gain = dp - bp
            total_gain += gain
            count += 1

            gain_str = f"+{gain:.4f}" if gain >= 0 else f"{gain:.4f}"
            print(f"  {ds:<12} {bp:<12.4f} {dp:<12.4f} {gain_str:<10} "
                  f"{bs:<12.4f} {ds_ssim:<12.4f}")

    if count > 0:
        avg_gain = total_gain / count
        status = "IMPROVED" if avg_gain > 0 else "DEGRADED"
        print(f"\n  Average PSNR gain: {avg_gain:+.4f} dB ({status})")
    print(f"{'=' * 80}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate CFSR models")
    parser.add_argument("--model", type=str, required=True,
                        choices=["baseline", "delta", "both"])
    parser.add_argument("--backbone_weights", type=str, required=True)
    parser.add_argument("--refine_weights", type=str, default=None)
    parser.add_argument("--scale", type=int, default=4, choices=[2, 3, 4])
    parser.add_argument("--benchmark_dir", type=str, default="datasets/benchmark")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output_json", type=str, default=None,
                        help="Save results to JSON file")
    args = parser.parse_args()

    all_results = {}

    # Evaluate baseline
    if args.model in ("baseline", "both"):
        print("Loading baseline model...")
        model, params = load_cfsr_model(args.scale, args.backbone_weights, args.device)
        results = evaluate_all_benchmarks(
            model, args.scale, args.benchmark_dir, args.device
        )
        print_results("CFSR Baseline", results, params)
        all_results["baseline"] = results

    # Evaluate delta
    if args.model in ("delta", "both"):
        if args.refine_weights is None:
            print("[ERROR] --refine_weights required for delta model")
            sys.exit(1)
        print("\nLoading delta model...")
        model, counts = load_delta_model(
            args.scale, args.backbone_weights, args.refine_weights,
            device=args.device,
        )
        results = evaluate_all_benchmarks(
            model, args.scale, args.benchmark_dir, args.device
        )
        print_results("CFSR-Delta", results, counts["total"])
        all_results["delta"] = results

    # Comparison
    if args.model == "both" and "baseline" in all_results and "delta" in all_results:
        print_comparison(all_results["baseline"], all_results["delta"])

    # Save JSON
    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {args.output_json}")


if __name__ == "__main__":
    main()
