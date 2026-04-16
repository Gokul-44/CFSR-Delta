"""
cfsr.py — Official CFSR (ConvFormer Super-Resolution) backbone.

Implements the complete CFSR architecture as described in:
    "CFSR: Efficient Lightweight Image Super-Resolution with ConvFormer"

The model uses two key building blocks:
    - ConvMod: Gated spatial modulation via large-kernel depthwise convolution
    - EFN (Edge-aware Feed-forward Network): MLP augmented with Sobel and
      Laplacian edge-detection priors

Architecture:
    LR → conv_first → [ResGroup × 2 (6 blocks each)] → norm → conv → upsample → SR

This implementation is weight-compatible with the official pretrained checkpoints.
Load weights with `strict=True` after calling `merge_all()`.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    """Channel-first LayerNorm (from ConvNeXt)."""

    def __init__(self, normalized_shape: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class LargeKernelConv(nn.Module):
    """Large-kernel depthwise convolution (re-parameterized form)."""

    def __init__(self, channels: int, kernel_size: int, groups: int = 1):
        super().__init__()
        self.lkb_reparam = nn.Conv2d(
            channels, channels, kernel_size, stride=1,
            padding=kernel_size // 2, groups=groups, bias=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lkb_reparam(x)


class MLP(nn.Module):
    """
    Edge-aware Feed-forward Network (EFN).

    Standard MLP augmented with fixed Sobel-X, Sobel-Y, and Laplacian
    edge-detection kernels with learnable scale factors. At inference,
    these auxiliary kernels are merged into the depthwise conv via
    `merge_mlp()` for zero overhead.
    """

    def __init__(self, dim: int, mlp_ratio: int = 2):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.out_channels = hidden
        self.norm = LayerNorm(dim)
        self.fc1 = nn.Conv2d(dim, hidden, 1)
        self.pos = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
        self.fc2 = nn.Conv2d(hidden, dim, 1)
        self.act = nn.GELU()
        self.merge_kernel = False

        # Edge-Detection Convolutions (EDC): fixed masks, learnable scales
        edge_kernels = {
            'sobel_x': [(0, 1, 1), (1, 0, 2), (2, 0, 1),
                        (0, 2, -1), (1, 2, -2), (2, 2, -1)],
            'sobel_y': [(0, 0, 1), (0, 1, 2), (0, 2, 1),
                        (2, 0, -1), (2, 1, -2), (2, 2, -1)],
            'laplacian': [(0, 0, 1), (1, 0, 1), (1, 2, 1),
                          (2, 1, 1), (1, 1, -4)],
        }
        for name, kernel_values in edge_kernels.items():
            setattr(self, f'scale_{name}',
                    nn.Parameter(torch.randn(hidden, 1, 1, 1) * 1e-3))
            setattr(self, f'bias_{name}',
                    nn.Parameter(torch.randn(hidden) * 1e-3))
            mask = torch.zeros(hidden, 1, 3, 3)
            for r, c, v in kernel_values:
                mask[:, 0, r, c] = v
            setattr(self, f'mask_{name}',
                    nn.Parameter(mask, requires_grad=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        out = self.pos(x)
        if not self.merge_kernel:
            for name in ['sobel_x', 'sobel_y', 'laplacian']:
                scale = getattr(self, f'scale_{name}')
                mask = getattr(self, f'mask_{name}')
                bias = getattr(self, f'bias_{name}')
                out = out + F.conv2d(
                    x, scale * mask, bias,
                    padding=1, groups=self.out_channels
                )
        return self.fc2(x + self.act(out))

    def merge_mlp(self):
        """Merge EDC kernels into depthwise conv for faster inference."""
        k = self.pos.weight.data.clone()
        b = self.pos.bias.data.clone()
        for name in ['sobel_x', 'sobel_y', 'laplacian']:
            k += getattr(self, f'scale_{name}') * getattr(self, f'mask_{name}')
            b += getattr(self, f'bias_{name}')
        self.merge_kernel = True
        self.pos.weight.data = k
        self.pos.bias.data = b
        for name in ['sobel_x', 'sobel_y', 'laplacian']:
            for attr in [f'scale_{name}', f'mask_{name}', f'bias_{name}']:
                delattr(self, attr)


class ConvMod(nn.Module):
    """
    ConvFormer spatial attention module.

    Computes gated spatial modulation: output = proj(gate * value)
    where the gate uses a large-kernel depthwise conv for wide receptive field.
    """

    def __init__(self, dim: int, dw_size: int):
        super().__init__()
        self.norm = LayerNorm(dim)
        self.a = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.GELU(),
            LargeKernelConv(dim, dw_size, groups=dim)
        )
        self.v = nn.Conv2d(dim, dim, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        return self.proj(self.a(x) * self.v(x))


class Block(nn.Module):
    """ConvFormer block: ConvMod + EFN with layer scale."""

    def __init__(self, dim: int, dw_size: int, mlp_ratio: int = 2):
        super().__init__()
        self.attn = ConvMod(dim, dw_size)
        self.mlp = MLP(dim, mlp_ratio)
        self.layer_scale_1 = nn.Parameter(1e-6 * torch.ones(dim))
        self.layer_scale_2 = nn.Parameter(1e-6 * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.layer_scale_1[:, None, None] * self.attn(x)
        x = x + self.layer_scale_2[:, None, None] * self.mlp(x)
        return x


class ResidualGroup(nn.Module):
    """Residual group of N ConvFormer blocks with a 1×1 skip."""

    def __init__(self, dim: int, depth: int, dw_size: int, mlp_ratio: int):
        super().__init__()
        self.blocks = nn.ModuleList(
            [Block(dim, dw_size, mlp_ratio) for _ in range(depth)]
        )
        self.conv = nn.Conv2d(dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return self.conv(x) + x


class CFSR(nn.Module):
    """
    CFSR (ConvFormer Super-Resolution) backbone.

    Official architecture with PixelShuffle upsampling.
    Pretrained on DF2K with L1 loss for 300K iterations.

    Args:
        dim: Embedding dimension (default: 48)
        depths: Number of blocks per residual group (default: (6, 6))
        dw_size: Large-kernel size for ConvMod (default: 9)
        mlp_ratio: MLP expansion ratio (default: 2)
        scale: Upscaling factor (default: 4)
    """

    def __init__(
        self,
        dim: int = 48,
        depths: tuple = (6, 6),
        dw_size: int = 9,
        mlp_ratio: int = 2,
        scale: int = 4,
    ):
        super().__init__()
        self.img_range = 1.0
        self.upscale = scale
        self.mean = torch.Tensor([0.4488, 0.4371, 0.4040]).view(1, 3, 1, 1)

        self.conv_first = nn.Conv2d(3, dim, 3, 1, 1)
        self.layers = nn.ModuleList(
            [ResidualGroup(dim, d, dw_size, mlp_ratio) for d in depths]
        )
        self.norm = LayerNorm(dim)
        self.conv_after_body = nn.Conv2d(dim, dim, 3, 1, 1)
        self.upsample = nn.Sequential(
            nn.Conv2d(dim, (scale ** 2) * 3, 3, 1, 1),
            nn.PixelShuffle(scale)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        f = self.conv_first(x)
        b = f
        for layer in self.layers:
            b = layer(b)
        b = self.conv_after_body(self.norm(b)) + f

        return self.upsample(b) / self.img_range + self.mean

    def merge_all(self):
        """Merge EDC kernels into depthwise convs for inference."""
        for _, module in self.named_modules():
            if isinstance(module, MLP) and not module.merge_kernel:
                module.merge_mlp()

    def param_count(self) -> int:
        """Return total parameter count."""
        return sum(p.numel() for p in self.parameters())


def load_cfsr_model(
    scale: int = 4,
    weights_path: str = None,
    device: str = "cpu",
) -> tuple:
    """
    Load the official CFSR model with pretrained weights.

    Args:
        scale: Upscale factor (2, 3, or 4).
        weights_path: Path to .pth checkpoint. None returns uninitialized model.
        device: Target device ('cpu' or 'cuda').

    Returns:
        Tuple of (model, parameter_count).
    """
    model = CFSR(dim=48, depths=(6, 6), dw_size=9, mlp_ratio=2, scale=scale)
    model.merge_all()

    if weights_path is not None:
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights not found: {weights_path}")
        state_dict = torch.load(weights_path, map_location=device, weights_only=True)
        if 'params' in state_dict:
            state_dict = state_dict['params']
        elif 'params_ema' in state_dict:
            state_dict = state_dict['params_ema']
        model.load_state_dict(state_dict, strict=True)

    model = model.to(device).eval()
    param_count = model.param_count()
    return model, param_count
