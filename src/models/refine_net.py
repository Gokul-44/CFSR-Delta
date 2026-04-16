"""
refine_net.py — Lightweight post-processing refinement network.

RefineNet is a tiny 3-layer convolutional network that learns additive
corrections to the CFSR backbone output. The key equation:

    SR_delta = SR_baseline + RefineNet.body(SR_baseline)

Design decisions:
    - 3 conv layers with 32 hidden channels (~11K parameters)
    - Residual connection ensures RefineNet(x) = x when body(x) ≈ 0
    - Last conv initialized to near-zero (N(0, 1e-4)) so output
      starts as an exact copy of the backbone SR
    - Only RefineNet is trained; the backbone remains frozen

This approach avoids the distribution mismatch problem that occurs when
fine-tuning the pretrained backbone with a different training pipeline.
"""

import torch
import torch.nn as nn


class RefineNet(nn.Module):
    """
    Residual refinement network for post-processing SR output.

    Architecture:
        Conv(3→hidden, 3×3) → ReLU → Conv(hidden→hidden, 3×3) → ReLU → Conv(hidden→3, 3×3)
        output = input + body(input)

    Args:
        hidden_channels: Number of hidden feature channels (default: 32).
        init_scale: Std of last conv weight initialization (default: 1e-4).
            Smaller values make the initial correction closer to zero.
    """

    def __init__(self, hidden_channels: int = 32, init_scale: float = 1e-4):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(3, hidden_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 3, 3, 1, 1),
        )
        # Initialize last layer to near-zero for identity at startup
        nn.init.normal_(self.body[-1].weight, 0, init_scale)
        nn.init.zeros_(self.body[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply residual correction: output = x + body(x)."""
        return x + self.body(x)

    def param_count(self) -> int:
        """Return total trainable parameter count."""
        return sum(p.numel() for p in self.parameters())
