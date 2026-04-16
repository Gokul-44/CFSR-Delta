"""
frequency_loss.py — Frequency-aware loss for super-resolution training.

Contains:
    - FrequencyLoss: FFT-based loss targeting high-frequency details
    - CombinedLoss: L1 + alpha * FrequencyLoss (with warm-up schedule)

The frequency loss encourages the model to reconstruct high-frequency
details (edges, textures) that spatial L1 loss tends to smooth out.

Usage:
    criterion = CombinedLoss(alpha=0.005, warmup_iters=2000)
    loss, info = criterion(sr, hr, current_iter=500)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FrequencyLoss(nn.Module):
    """
    Frequency-domain loss using 2D FFT.

    Computes L1 distance between magnitude spectra of SR and HR images.
    This encourages reconstruction of high-frequency content that spatial
    L1 loss alone tends to suppress.
    """

    def __init__(self):
        super().__init__()

    def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sr: Super-resolved image [B, C, H, W].
            hr: Ground truth image [B, C, H, W].

        Returns:
            Scalar loss value.
        """
        # Match spatial dimensions
        h = min(sr.shape[2], hr.shape[2])
        w = min(sr.shape[3], hr.shape[3])
        sr = sr[:, :, :h, :w]
        hr = hr[:, :, :h, :w]

        # 2D FFT → magnitude spectrum
        sr_fft = torch.fft.fft2(sr, norm='ortho')
        hr_fft = torch.fft.fft2(hr, norm='ortho')

        sr_mag = torch.abs(sr_fft)
        hr_mag = torch.abs(hr_fft)

        return F.l1_loss(sr_mag, hr_mag)


class CombinedLoss(nn.Module):
    """
    Combined spatial + frequency loss with linear warm-up.

    Loss = L1(sr, hr) + alpha(iter) * FrequencyLoss(sr, hr)

    where alpha(iter) = alpha_max * min(1.0, iter / warmup_iters)

    This lets L1 stabilize the model first before frequency gradients
    are introduced, preventing early training instability.

    Args:
        alpha: Maximum weight for frequency loss (default: 0.005).
        warmup_iters: Iterations to linearly ramp alpha (default: 2000).
    """

    def __init__(self, alpha: float = 0.005, warmup_iters: int = 2000):
        super().__init__()
        self.alpha_max = alpha
        self.warmup_iters = max(warmup_iters, 1)
        self.l1_loss = nn.L1Loss()
        self.freq_loss = FrequencyLoss()

    def forward(
        self,
        sr: torch.Tensor,
        hr: torch.Tensor,
        current_iter: int = 0,
    ) -> tuple:
        """
        Compute combined loss.

        Args:
            sr: Super-resolved image [B, C, H, W].
            hr: Ground truth image [B, C, H, W].
            current_iter: Current training iteration (for warm-up).

        Returns:
            Tuple of (total_loss, loss_dict) where loss_dict contains
            individual loss components and current alpha value.
        """
        warmup_ratio = min(1.0, current_iter / self.warmup_iters)
        alpha = self.alpha_max * warmup_ratio

        l1 = self.l1_loss(sr, hr)
        freq = self.freq_loss(sr, hr)
        total = l1 + alpha * freq

        loss_dict = {
            "l1": l1.item(),
            "freq": freq.item(),
            "alpha": alpha,
            "total": total.item(),
        }
        return total, loss_dict
