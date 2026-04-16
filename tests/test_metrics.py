"""
test_metrics.py — Correctness tests for PSNR and SSIM metrics.

Verifies:
    - PSNR of identical images → inf
    - SSIM of identical images → 1.0
    - PSNR decreases with increasing noise
    - Y-channel conversion is in valid range
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.metrics.sr_metrics import calc_psnr, calc_ssim, rgb2ycbcr


def test_psnr_identical():
    """PSNR of identical images should be inf."""
    img = np.random.rand(64, 64, 3).astype(np.float32)
    psnr = calc_psnr(img, img, scale=4)
    assert psnr == float('inf'), f"Expected inf, got {psnr}"
    print("[PASS] test_psnr_identical: PSNR = inf")


def test_ssim_identical():
    """SSIM of identical images should be ~1.0."""
    img = np.random.rand(64, 64, 3).astype(np.float32) * 0.5 + 0.25
    ssim = calc_ssim(img, img, scale=4)
    assert ssim > 0.999, f"Expected ~1.0, got {ssim}"
    print(f"[PASS] test_ssim_identical: SSIM = {ssim:.6f}")


def test_psnr_decreases_with_noise():
    """PSNR should decrease as noise increases."""
    img = np.random.rand(64, 64, 3).astype(np.float32)
    psnr_values = []

    for noise_level in [0.001, 0.01, 0.05, 0.1]:
        noisy = np.clip(img + np.random.randn(*img.shape).astype(np.float32) * noise_level, 0, 1)
        psnr = calc_psnr(img, noisy, scale=4)
        psnr_values.append(psnr)

    for i in range(len(psnr_values) - 1):
        assert psnr_values[i] > psnr_values[i + 1], (
            f"PSNR not decreasing: {psnr_values[i]:.2f} vs {psnr_values[i+1]:.2f}"
        )

    print(f"[PASS] test_psnr_decreases_with_noise: {[f'{v:.1f}' for v in psnr_values]}")


def test_ycbcr_range():
    """Y channel should be in valid range for natural images."""
    # Black image
    black = np.zeros((32, 32, 3), dtype=np.float32)
    y_black = rgb2ycbcr(black)
    assert y_black.min() >= 0, f"Y < 0 for black: {y_black.min()}"

    # White image
    white = np.ones((32, 32, 3), dtype=np.float32)
    y_white = rgb2ycbcr(white)
    assert y_white.max() <= 1.1, f"Y > 1.1 for white: {y_white.max()}"

    # Random natural image
    img = np.random.rand(32, 32, 3).astype(np.float32)
    y = rgb2ycbcr(img)
    assert 0 <= y.min() and y.max() <= 1.1, f"Y out of range: [{y.min():.3f}, {y.max():.3f}]"

    print(f"[PASS] test_ycbcr_range: Y in [{y.min():.4f}, {y.max():.4f}]")


def test_psnr_symmetry():
    """PSNR(a, b) should equal PSNR(b, a)."""
    a = np.random.rand(64, 64, 3).astype(np.float32)
    b = np.clip(a + np.random.randn(*a.shape).astype(np.float32) * 0.01, 0, 1)

    psnr_ab = calc_psnr(a, b, scale=4)
    psnr_ba = calc_psnr(b, a, scale=4)

    assert abs(psnr_ab - psnr_ba) < 1e-6, f"Not symmetric: {psnr_ab} vs {psnr_ba}"
    print(f"[PASS] test_psnr_symmetry: {psnr_ab:.4f} == {psnr_ba:.4f}")


if __name__ == "__main__":
    test_psnr_identical()
    test_ssim_identical()
    test_psnr_decreases_with_noise()
    test_ycbcr_range()
    test_psnr_symmetry()
    print("\nAll metric tests passed!")
