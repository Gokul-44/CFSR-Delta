"""
df2k_dataset.py — DF2K training dataset for super-resolution.

Implements on-the-fly LR generation via bicubic downsampling with
random patch extraction and standard augmentation (flip, rotate 90°).

The DF2K dataset is a combination of DIV2K (800 images) and Flickr2K
(~2650 images), totaling ~3450 high-quality natural images.
"""

import os
import glob
import random

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset


class DF2KDataset(Dataset):
    """
    DF2K training dataset with random patch extraction.

    Expects HR images in `data_dir`. LR patches are generated on-the-fly
    using bicubic downsampling. Virtual epoch size is 50× the number of
    images, so each epoch samples each image ~50 times with different
    random crops.

    Args:
        data_dir: Path to directory containing HR images (PNG/JPG).
        scale: Downscaling factor (default: 4).
        patch_size: HR patch size for training (default: 64).
        augment: Whether to apply random flip/rotation (default: True).
    """

    def __init__(
        self,
        data_dir: str,
        scale: int = 4,
        patch_size: int = 64,
        augment: bool = True,
    ):
        self.scale = scale
        self.patch_size = patch_size
        self.augment = augment

        self.hr_paths = []
        for ext in ["*.png", "*.jpg", "*.jpeg"]:
            self.hr_paths.extend(
                glob.glob(os.path.join(data_dir, "**", ext), recursive=True)
            )
        self.hr_paths.sort()

        if len(self.hr_paths) == 0:
            raise RuntimeError(f"No images found in {data_dir}")

    def __len__(self) -> int:
        return len(self.hr_paths) * 50  # virtual epoch size

    def __getitem__(self, idx: int) -> tuple:
        # Select image with wraparound
        img_idx = idx % len(self.hr_paths)
        hr_img = cv2.imread(self.hr_paths[img_idx], cv2.IMREAD_COLOR)
        hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # Random crop
        lr_patch, hr_patch = self._random_crop(hr_img)

        # Augmentation
        if self.augment:
            lr_patch, hr_patch = self._augment(lr_patch, hr_patch)

        # To tensor [C, H, W]
        lr_tensor = torch.from_numpy(lr_patch.transpose(2, 0, 1).copy()).float()
        hr_tensor = torch.from_numpy(hr_patch.transpose(2, 0, 1).copy()).float()

        return lr_tensor, hr_tensor

    def _random_crop(self, hr_img: np.ndarray) -> tuple:
        """Random crop HR patch → generate LR by bicubic downscaling."""
        h, w = hr_img.shape[:2]
        ps = self.patch_size

        # Pad if image is too small
        if h < ps or w < ps:
            hr_img = cv2.resize(
                hr_img, (max(w, ps), max(h, ps)),
                interpolation=cv2.INTER_CUBIC,
            )
            h, w = hr_img.shape[:2]

        top = random.randint(0, h - ps)
        left = random.randint(0, w - ps)
        hr_patch = hr_img[top:top + ps, left:left + ps]

        # Generate LR by bicubic downscaling
        lr_h, lr_w = ps // self.scale, ps // self.scale
        lr_patch = cv2.resize(
            hr_patch, (lr_w, lr_h), interpolation=cv2.INTER_CUBIC
        )

        return lr_patch, hr_patch

    def _augment(self, lr: np.ndarray, hr: np.ndarray) -> tuple:
        """Random horizontal flip, vertical flip, and 90° rotation."""
        if random.random() > 0.5:
            lr = np.flip(lr, axis=1).copy()
            hr = np.flip(hr, axis=1).copy()
        if random.random() > 0.5:
            lr = np.flip(lr, axis=0).copy()
            hr = np.flip(hr, axis=0).copy()
        if random.random() > 0.5:
            lr = np.transpose(lr, (1, 0, 2)).copy()
            hr = np.transpose(hr, (1, 0, 2)).copy()
        return lr, hr
