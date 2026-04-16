"""
cfsr_delta.py — Combined CFSR + RefineNet pipeline.

Wraps the frozen CFSR backbone with the trainable RefineNet post-processor
into a single nn.Module for end-to-end inference and training.

Pipeline:
    LR → [Frozen CFSR backbone] → SR_baseline → [RefineNet] → SR_delta

The backbone runs under torch.no_grad() for memory efficiency.
Only RefineNet parameters receive gradients during training.
"""

import os
import torch
import torch.nn as nn

from src.models.cfsr import CFSR
from src.models.refine_net import RefineNet


class CFSRDelta(nn.Module):
    """
    Frozen CFSR backbone + trainable RefineNet post-processor.

    Args:
        backbone: Pretrained CFSR model (will be frozen).
        refine: RefineNet module (trainable).
    """

    def __init__(self, backbone: CFSR, refine: RefineNet):
        super().__init__()
        self.backbone = backbone
        self.refine = refine

        # Freeze all backbone parameters
        for p in self.backbone.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: frozen backbone → refinement."""
        with torch.no_grad():
            sr = self.backbone(x)
        return self.refine(sr)

    def param_count(self) -> dict:
        """Return parameter counts for backbone and refiner."""
        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        refine_params = sum(p.numel() for p in self.refine.parameters())
        return {
            "backbone": backbone_params,
            "refine": refine_params,
            "total": backbone_params + refine_params,
            "trainable": refine_params,
        }


def load_delta_model(
    scale: int = 4,
    backbone_weights: str = None,
    refine_weights: str = None,
    hidden_channels: int = 32,
    init_scale: float = 1e-4,
    device: str = "cpu",
    seed: int = 69,
) -> tuple:
    """
    Build and load a complete CFSR-Delta model.

    Args:
        scale: Upscale factor (2, 3, or 4).
        backbone_weights: Path to official CFSR pretrained weights.
        refine_weights: Path to trained RefineNet checkpoint.
        hidden_channels: RefineNet hidden channels.
        init_scale: RefineNet last-layer init scale.
        device: Target device.
        seed: Random seed for RefineNet initialization.

    Returns:
        Tuple of (model, param_counts_dict).
    """
    # Build backbone
    backbone = CFSR(dim=48, depths=(6, 6), dw_size=9, mlp_ratio=2, scale=scale)
    backbone.merge_all()

    if backbone_weights is not None:
        if not os.path.exists(backbone_weights):
            raise FileNotFoundError(f"Backbone weights not found: {backbone_weights}")
        state_dict = torch.load(
            backbone_weights, map_location=device, weights_only=True
        )
        if "params" in state_dict:
            state_dict = state_dict["params"]
        elif "params_ema" in state_dict:
            state_dict = state_dict["params_ema"]
        backbone.load_state_dict(state_dict, strict=True)

    backbone = backbone.to(device).eval()

    # Build RefineNet
    torch.manual_seed(seed)
    refine = RefineNet(hidden_channels=hidden_channels, init_scale=init_scale)

    if refine_weights is not None:
        if not os.path.exists(refine_weights):
            raise FileNotFoundError(f"RefineNet weights not found: {refine_weights}")
        ckpt = torch.load(refine_weights, map_location=device, weights_only=False)
        if "refine" in ckpt:
            refine.load_state_dict(ckpt["refine"])
        else:
            refine.load_state_dict(ckpt)

    # Combine
    model = CFSRDelta(backbone, refine).to(device).eval()
    counts = model.param_count()
    return model, counts
