"""
sr_metrics.py — Standard super-resolution evaluation metrics.

Implements PSNR and SSIM on the Y channel of YCbCr, following the
standard protocol used by all major SR benchmarks:
    1. Convert RGB to Y channel using ITU-R BT.601
    2. Crop `scale` pixels from each border
    3. Compute metrics on the cropped Y channel

Reference values (CFSR ×4):
    Set5: PSNR=32.33 dB, SSIM=0.8982 (paper)
"""

import os
import numpy as np
import cv2
import torch


def rgb2ycbcr(img: np.ndarray) -> np.ndarray:
    """
    Convert RGB image to Y channel (luminance) using ITU-R BT.601.

    Args:
        img: Float32 array [H, W, 3] in range [0, 1], RGB format.

    Returns:
        Y channel as float32 array [H, W].
    """
    y = (16.0 / 255.0
         + (65.481 * img[..., 0]
            + 128.553 * img[..., 1]
            + 24.966 * img[..., 2]) / 255.0)
    return y


def calc_psnr(
    sr: np.ndarray,
    hr: np.ndarray,
    scale: int,
    only_y: bool = True,
) -> float:
    """
    Calculate PSNR between SR and HR images.

    Args:
        sr: Super-resolved image [H, W, C], float32 [0, 1], RGB.
        hr: Ground truth image [H, W, C], float32 [0, 1], RGB.
        scale: Upscale factor (for border cropping).
        only_y: If True, compute on Y channel only (standard protocol).

    Returns:
        PSNR value in dB.
    """
    if only_y:
        sr_y = rgb2ycbcr(sr)
        hr_y = rgb2ycbcr(hr)
    else:
        sr_y = sr
        hr_y = hr

    if scale > 0:
        sr_y = sr_y[scale:-scale, scale:-scale]
        hr_y = hr_y[scale:-scale, scale:-scale]

    mse = np.mean((sr_y - hr_y) ** 2)
    if mse == 0:
        return float('inf')
    return 10.0 * np.log10(1.0 / mse)


def calc_ssim(
    sr: np.ndarray,
    hr: np.ndarray,
    scale: int,
    only_y: bool = True,
) -> float:
    """
    Calculate SSIM between SR and HR images on Y channel.

    Uses the standard 11×11 Gaussian window (sigma=1.5) approach.

    Args:
        sr: Super-resolved image [H, W, C], float32 [0, 1], RGB.
        hr: Ground truth image [H, W, C], float32 [0, 1], RGB.
        scale: Upscale factor (for border cropping).
        only_y: If True, compute on Y channel only.

    Returns:
        SSIM value in [0, 1].
    """
    if only_y:
        sr_y = rgb2ycbcr(sr)
        hr_y = rgb2ycbcr(hr)
    else:
        sr_y = sr
        hr_y = hr

    if scale > 0:
        sr_y = sr_y[scale:-scale, scale:-scale]
        hr_y = hr_y[scale:-scale, scale:-scale]

    C1 = (0.01 * 1.0) ** 2
    C2 = (0.03 * 1.0) ** 2

    sr_y = sr_y.astype(np.float64)
    hr_y = hr_y.astype(np.float64)

    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(sr_y, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(hr_y, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.filter2D(sr_y ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(hr_y ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(sr_y * hr_y, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return float(ssim_map.mean())


def evaluate_dataset(
    model: torch.nn.Module,
    dataset_name: str,
    scale: int = 4,
    benchmark_dir: str = "datasets/benchmark",
    device: str = "cpu",
) -> dict:
    """
    Evaluate a model on a standard SR benchmark dataset.

    Args:
        model: SR model (in eval mode).
        dataset_name: Name of dataset (e.g., 'Set5', 'Set14').
        scale: Upscale factor.
        benchmark_dir: Root directory containing benchmark datasets.
        device: Compute device.

    Returns:
        Dict with 'psnr', 'ssim', and 'per_image' results.
        Returns None values if dataset is not found.
    """
    hr_dir = os.path.join(benchmark_dir, dataset_name, "HR")
    lr_dir = os.path.join(benchmark_dir, dataset_name, "LR_bicubic", f"X{scale}")

    if not os.path.isdir(hr_dir) or not os.path.isdir(lr_dir):
        return {"psnr": None, "ssim": None, "per_image": {}}

    psnr_list = []
    ssim_list = []
    per_image = {}

    model.eval()
    with torch.no_grad():
        for fname in sorted(os.listdir(hr_dir)):
            if not fname.lower().endswith(".png"):
                continue

            base_name = fname.replace(".png", "")
            lr_name = f"{base_name}x{scale}.png"
            lr_path = os.path.join(lr_dir, lr_name)
            hr_path = os.path.join(hr_dir, fname)

            if not os.path.exists(lr_path):
                continue

            # Load images: BGR → RGB, normalize to [0, 1]
            lr_img = cv2.imread(lr_path, cv2.IMREAD_COLOR)
            hr_img = cv2.imread(hr_path, cv2.IMREAD_COLOR)
            lr_img = cv2.cvtColor(lr_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

            # Inference
            lr_tensor = torch.from_numpy(
                lr_img.transpose(2, 0, 1)
            ).unsqueeze(0).to(device)
            sr_tensor = model(lr_tensor)
            sr_img = sr_tensor.squeeze(0).cpu().clamp(0, 1).numpy().transpose(1, 2, 0)

            # Match dimensions
            h = min(sr_img.shape[0], hr_img.shape[0])
            w = min(sr_img.shape[1], hr_img.shape[1])
            sr_img = sr_img[:h, :w, :]
            hr_img = hr_img[:h, :w, :]

            # Compute metrics
            psnr_val = calc_psnr(sr_img, hr_img, scale=scale)
            ssim_val = calc_ssim(sr_img, hr_img, scale=scale)
            psnr_list.append(psnr_val)
            ssim_list.append(ssim_val)
            per_image[fname] = {"psnr": psnr_val, "ssim": ssim_val}

    if len(psnr_list) == 0:
        return {"psnr": None, "ssim": None, "per_image": {}}

    return {
        "psnr": float(np.mean(psnr_list)),
        "ssim": float(np.mean(ssim_list)),
        "per_image": per_image,
    }


def evaluate_all_benchmarks(
    model: torch.nn.Module,
    scale: int = 4,
    benchmark_dir: str = "datasets/benchmark",
    device: str = "cpu",
    datasets: list = None,
) -> dict:
    """
    Evaluate a model on all standard SR benchmarks.

    Args:
        model: SR model (in eval mode).
        scale: Upscale factor.
        benchmark_dir: Root directory containing benchmark datasets.
        device: Compute device.
        datasets: List of dataset names. Defaults to standard 5.

    Returns:
        Dict mapping dataset names to results.
    """
    if datasets is None:
        datasets = ["Set5", "Set14", "B100", "Urban100", "Manga109"]

    results = {}
    for ds in datasets:
        result = evaluate_dataset(model, ds, scale, benchmark_dir, device)
        if result["psnr"] is not None:
            results[ds] = {
                "psnr": round(result["psnr"], 4),
                "ssim": round(result["ssim"], 4),
            }
    return results
