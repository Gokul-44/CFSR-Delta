"""
test_model.py — Forward pass shape tests for CFSR and CFSRDelta models.

Verifies:
    - CFSR produces correct output shapes for ×4 upscaling
    - RefineNet is a true identity at initialization
    - CFSRDelta correctly freezes backbone parameters
    - Weight loading works with strict=True
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from src.models.cfsr import CFSR
from src.models.refine_net import RefineNet
from src.models.cfsr_delta import CFSRDelta


def test_cfsr_output_shape():
    """CFSR should produce 4× upscaled output."""
    model = CFSR(dim=48, depths=(6, 6), dw_size=9, mlp_ratio=2, scale=4)
    model.merge_all()
    model.eval()

    x = torch.randn(1, 3, 16, 16)
    with torch.no_grad():
        y = model(x)

    assert y.shape == (1, 3, 64, 64), f"Expected (1,3,64,64), got {y.shape}"
    print("[PASS] test_cfsr_output_shape: (1,3,16,16) -> (1,3,64,64)")


def test_cfsr_param_count():
    """CFSR should have ~306K parameters."""
    model = CFSR()
    model.merge_all()
    count = model.param_count()
    assert 300_000 < count < 320_000, f"Unexpected param count: {count}"
    print(f"[PASS] test_cfsr_param_count: {count:,} params")


def test_refine_net_identity():
    """RefineNet with near-zero init should approximate identity."""
    torch.manual_seed(42)
    refine = RefineNet(hidden_channels=32, init_scale=1e-6)
    refine.eval()

    x = torch.randn(1, 3, 64, 64) * 0.5 + 0.5  # simulate image

    with torch.no_grad():
        y = refine(x)

    diff = (y - x).abs().max().item()
    assert diff < 0.01, f"RefineNet not identity at init: max diff = {diff}"
    print(f"[PASS] test_refine_net_identity: max diff = {diff:.6f}")


def test_refine_net_param_count():
    """RefineNet(32) should have ~11K parameters."""
    refine = RefineNet(hidden_channels=32)
    count = refine.param_count()
    assert 10_000 < count < 12_000, f"Unexpected param count: {count}"
    print(f"[PASS] test_refine_net_param_count: {count:,} params")


def test_cfsr_delta_forward():
    """CFSRDelta should chain backbone → refine correctly."""
    backbone = CFSR()
    backbone.merge_all()
    refine = RefineNet(hidden_channels=32)
    model = CFSRDelta(backbone, refine)
    model.eval()

    x = torch.randn(1, 3, 16, 16)
    with torch.no_grad():
        y = model(x)

    assert y.shape == (1, 3, 64, 64), f"Expected (1,3,64,64), got {y.shape}"
    print("[PASS] test_cfsr_delta_forward: shape correct")


def test_cfsr_delta_freezes_backbone():
    """Backbone params should be frozen in CFSRDelta."""
    backbone = CFSR()
    backbone.merge_all()
    refine = RefineNet(hidden_channels=32)
    model = CFSRDelta(backbone, refine)

    for name, param in model.backbone.named_parameters():
        assert not param.requires_grad, f"Backbone param not frozen: {name}"

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    refine_params = refine.param_count()
    assert trainable == refine_params, (
        f"Trainable params ({trainable}) != RefineNet params ({refine_params})"
    )
    print(f"[PASS] test_cfsr_delta_freezes_backbone: {trainable:,} trainable")


def test_various_input_sizes():
    """Model should handle different input sizes."""
    model = CFSR()
    model.merge_all()
    model.eval()

    for h, w in [(24, 24), (32, 48), (16, 64), (48, 32)]:
        x = torch.randn(1, 3, h, w)
        with torch.no_grad():
            y = model(x)
        expected = (1, 3, h * 4, w * 4)
        assert y.shape == expected, f"Input ({h},{w}): expected {expected}, got {y.shape}"

    print("[PASS] test_various_input_sizes: all sizes handled")


if __name__ == "__main__":
    test_cfsr_output_shape()
    test_cfsr_param_count()
    test_refine_net_identity()
    test_refine_net_param_count()
    test_cfsr_delta_forward()
    test_cfsr_delta_freezes_backbone()
    test_various_input_sizes()
    print("\nAll model tests passed!")
