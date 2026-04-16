"""
test_losses.py — Loss function correctness tests.

Verifies:
    - FrequencyLoss is zero for identical inputs
    - CombinedLoss warm-up schedule works correctly
    - Loss values are non-negative and finite
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.losses.frequency_loss import FrequencyLoss, CombinedLoss


def test_freq_loss_identical():
    """Frequency loss of identical tensors should be ~0."""
    fl = FrequencyLoss()
    x = torch.randn(2, 3, 64, 64)
    loss = fl(x, x)
    assert loss.item() < 1e-6, f"Expected ~0, got {loss.item()}"
    print(f"[PASS] test_freq_loss_identical: loss = {loss.item():.8f}")


def test_freq_loss_nonzero():
    """Frequency loss of different tensors should be > 0."""
    fl = FrequencyLoss()
    x = torch.randn(2, 3, 64, 64)
    y = torch.randn(2, 3, 64, 64)
    loss = fl(x, y)
    assert loss.item() > 0.0, f"Expected > 0, got {loss.item()}"
    print(f"[PASS] test_freq_loss_nonzero: loss = {loss.item():.4f}")


def test_combined_warmup_zero():
    """Alpha should be 0 at iteration 0."""
    cl = CombinedLoss(alpha=0.01, warmup_iters=1000)
    x = torch.randn(2, 3, 32, 32)
    y = torch.randn(2, 3, 32, 32)
    _, info = cl(x, y, current_iter=0)
    assert info["alpha"] == 0.0, f"Expected alpha=0 at iter=0, got {info['alpha']}"
    print(f"[PASS] test_combined_warmup_zero: alpha = {info['alpha']}")


def test_combined_warmup_half():
    """Alpha should be half at warmup_iters/2."""
    cl = CombinedLoss(alpha=0.01, warmup_iters=1000)
    x = torch.randn(2, 3, 32, 32)
    y = torch.randn(2, 3, 32, 32)
    _, info = cl(x, y, current_iter=500)
    assert abs(info["alpha"] - 0.005) < 1e-6, f"Expected alpha=0.005, got {info['alpha']}"
    print(f"[PASS] test_combined_warmup_half: alpha = {info['alpha']}")


def test_combined_warmup_full():
    """Alpha should be max at warmup_iters."""
    cl = CombinedLoss(alpha=0.01, warmup_iters=1000)
    x = torch.randn(2, 3, 32, 32)
    y = torch.randn(2, 3, 32, 32)
    _, info = cl(x, y, current_iter=1000)
    assert abs(info["alpha"] - 0.01) < 1e-6, f"Expected alpha=0.01, got {info['alpha']}"
    print(f"[PASS] test_combined_warmup_full: alpha = {info['alpha']}")


def test_combined_warmup_beyond():
    """Alpha should stay at max beyond warmup_iters."""
    cl = CombinedLoss(alpha=0.01, warmup_iters=1000)
    x = torch.randn(2, 3, 32, 32)
    y = torch.randn(2, 3, 32, 32)
    _, info = cl(x, y, current_iter=5000)
    assert abs(info["alpha"] - 0.01) < 1e-6, f"Expected alpha=0.01, got {info['alpha']}"
    print(f"[PASS] test_combined_warmup_beyond: alpha = {info['alpha']}")


def test_loss_finite():
    """All loss components should be finite."""
    cl = CombinedLoss(alpha=0.01, warmup_iters=100)
    x = torch.randn(2, 3, 32, 32)
    y = torch.randn(2, 3, 32, 32)
    total, info = cl(x, y, current_iter=100)

    assert torch.isfinite(total), f"Total loss not finite: {total}"
    assert all(
        v == v for k, v in info.items()
    ), f"NaN in loss info: {info}"
    print(f"[PASS] test_loss_finite: total={info['total']:.4f}")


if __name__ == "__main__":
    test_freq_loss_identical()
    test_freq_loss_nonzero()
    test_combined_warmup_zero()
    test_combined_warmup_half()
    test_combined_warmup_full()
    test_combined_warmup_beyond()
    test_loss_finite()
    print("\nAll loss tests passed!")
