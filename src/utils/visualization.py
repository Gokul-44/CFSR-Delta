"""
visualization.py — Visual comparison utilities for super-resolution.

Generates publication-quality side-by-side comparison images:
    LR (upscaled) | Baseline SR | Delta SR | HR (Ground Truth)

Each panel includes a zoomed crop region for examining fine detail
differences between the baseline and delta outputs.
"""

import os

import numpy as np
import cv2
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle


def load_image(path: str) -> np.ndarray:
    """Load image as RGB float32 [0, 1]."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def to_tensor(img: np.ndarray) -> torch.Tensor:
    """Convert [H, W, C] float32 array to [1, C, H, W] tensor."""
    return torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert [1, C, H, W] tensor to [H, W, C] float32 array."""
    return tensor.squeeze(0).cpu().clamp(0, 1).numpy().transpose(1, 2, 0)


def create_comparison(
    lr_img: np.ndarray,
    baseline_sr: np.ndarray,
    delta_sr: np.ndarray,
    hr_img: np.ndarray,
    scale: int = 4,
    baseline_psnr: float = None,
    delta_psnr: float = None,
) -> plt.Figure:
    """
    Create a publication-quality 4-panel comparison figure.

    Top row: Full images with crop region overlay.
    Bottom row: Zoomed crops for detail inspection.

    Args:
        lr_img: Low-resolution input [H/s, W/s, 3].
        baseline_sr: Baseline SR output [H, W, 3].
        delta_sr: Delta SR output [H, W, 3].
        hr_img: Ground truth HR image [H, W, 3].
        scale: Upscale factor.
        baseline_psnr: Optional PSNR value for baseline.
        delta_psnr: Optional PSNR value for delta.

    Returns:
        matplotlib Figure object.
    """
    H, W = hr_img.shape[:2]

    # Define crop region (upper-left quadrant for interesting detail)
    ch = min(96, H // 3)
    cw = min(96, W // 3)
    cy = H // 4
    cx = W // 4

    # Upscale LR for visual comparison (nearest neighbor to show pixelation)
    lr_up = cv2.resize(lr_img, (W, H), interpolation=cv2.INTER_NEAREST)

    # Dark theme colors
    BG = "#0f1117"
    PANEL_BG = "#1a1d27"
    TEXT = "#e8eaf6"
    ACCENT = "#7c83fd"
    GREEN = "#69f0ae"
    GOLD = "#ffd740"

    # Build figure
    fig = plt.figure(figsize=(20, 10), facecolor=BG)
    gs = gridspec.GridSpec(
        2, 4, figure=fig, hspace=0.08, wspace=0.06,
        top=0.88, bottom=0.04, left=0.02, right=0.98,
    )

    images = [lr_up, baseline_sr, delta_sr, hr_img]
    labels = [f"LR Input (×{scale})", "Baseline SR", "Delta SR (Ours)", "Ground Truth"]
    colors = [ACCENT, ACCENT, GREEN, GOLD]

    for i, (img, label, color) in enumerate(zip(images, labels, colors)):
        # Top row: full image
        ax_top = fig.add_subplot(gs[0, i])
        ax_top.set_facecolor(PANEL_BG)
        ax_top.imshow(np.clip(img, 0, 1))
        ax_top.add_patch(Rectangle(
            (cx, cy), cw, ch,
            linewidth=2, edgecolor=GOLD, facecolor="none", linestyle="--",
        ))
        ax_top.set_title(label, color=color, fontsize=12, fontweight="bold", pad=6)

        # Add PSNR sub-label
        if label == "Baseline SR" and baseline_psnr is not None:
            ax_top.set_xlabel(f"PSNR = {baseline_psnr:.4f} dB", color=TEXT, fontsize=10)
        elif label == "Delta SR (Ours)" and delta_psnr is not None and baseline_psnr is not None:
            gain = delta_psnr - baseline_psnr
            gain_color = GREEN if gain >= 0 else "#ff5252"
            ax_top.set_xlabel(
                f"PSNR = {delta_psnr:.4f} dB ({gain:+.4f})",
                color=gain_color, fontsize=10,
            )

        ax_top.tick_params(
            left=False, bottom=False, labelleft=False, labelbottom=False
        )
        for sp in ax_top.spines.values():
            sp.set_color(color)
            sp.set_linewidth(1.5)

        # Bottom row: zoomed crop
        ax_bot = fig.add_subplot(gs[1, i])
        ax_bot.set_facecolor(PANEL_BG)
        crop = np.clip(img[cy:cy + ch, cx:cx + cw], 0, 1)
        ax_bot.imshow(crop, interpolation="nearest")
        ax_bot.set_title("Zoom ×4", color=GOLD, fontsize=9, pad=4)
        ax_bot.tick_params(
            left=False, bottom=False, labelleft=False, labelbottom=False
        )
        for sp in ax_bot.spines.values():
            sp.set_color(GOLD)
            sp.set_linewidth(1.5)

    # Title
    title = "CFSR-Delta — Visual Comparison"
    if baseline_psnr and delta_psnr:
        title += f"  |  Gain: {delta_psnr - baseline_psnr:+.4f} dB"
    fig.suptitle(title, color=TEXT, fontsize=14, fontweight="bold", y=0.95)

    return fig


def save_comparison_figure(
    fig: plt.Figure,
    output_path: str,
    dpi: int = 150,
) -> None:
    """Save comparison figure to disk."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
