# Architecture — CFSR-Delta

## Overview

CFSR-Delta is a two-stage super-resolution pipeline:

```
LR Input → [Frozen CFSR Backbone] → SR_baseline → [RefineNet] → SR_delta
              306K params (fixed)                   11K params (trained)
```

The backbone is the official CFSR model, pre-trained on DF2K for 300K iterations.
RefineNet is our contribution — a tiny post-processor that learns additive corrections.

## Stage 1: CFSR Backbone

### ConvFormer Architecture

CFSR uses **ConvFormer** blocks — a hybrid of convolutions and attention-like gating:

| Component | Mechanism | Purpose |
|-----------|-----------|---------|
| **ConvMod** | `proj(gate(x) × value(x))` where gate uses 9×9 depthwise conv | Spatial attention at O(N) cost |
| **EFN** (Edge-aware Feed-forward) | MLP + fixed Sobel/Laplacian kernels with learnable scales | High-frequency bias |
| **LayerScale** | Learned per-channel scaling (init=1e-6) | Stable deep training |
| **PixelShuffle(4)** | Rearrange channels → spatial upsampling | Artifact-free upscaling |

### Architecture Dimensions

```
Input:  [B, 3, H, W]           # LR image
├── conv_first: 3 → 48         # Feature extraction
├── ResGroup 1: 6 × Block(48)  # Deep feature processing
├── ResGroup 2: 6 × Block(48)  # Deep feature processing
├── LayerNorm + conv: 48 → 48  # Post-processing + global skip
└── PixelShuffle: 48 → 3×16    # 4× spatial upsampling
Output: [B, 3, 4H, 4W]         # SR image
```

### Re-parameterization

At inference, the EFN's Sobel+Laplacian auxiliary convolutions are **merged**
into the depthwise conv via `merge_all()`. This gives zero inference overhead
while keeping the edge-aware bias during training.

## Stage 2: RefineNet (Our Contribution)

### Design Philosophy

1. **Frozen backbone**: No gradients flow into CFSR. This prevents the
   distribution mismatch that occurs when fine-tuning with a different
   data pipeline than the original training.

2. **Residual connection**: `output = input + body(input)`. If the body
   learns nothing, the output is exactly the baseline — no regression possible.

3. **Near-zero initialization**: The final conv layer is initialized with
   `N(0, 1e-4)` weights and zero bias. At iteration 0, RefineNet outputs
   are ~0, so the model starts at baseline PSNR.

### Architecture

```
SR_baseline → Conv(3→32, 3×3) → ReLU → Conv(32→32, 3×3) → ReLU → Conv(32→3, 3×3) → + SR_baseline
                                                                     ↑ init ≈ 0
```

### Parameter Count

| Layer | Shape | Parameters |
|-------|-------|-----------|
| Conv1 | (32, 3, 3, 3) + bias(32) | 896 |
| Conv2 | (32, 32, 3, 3) + bias(32) | 9,248 |
| Conv3 | (3, 32, 3, 3) + bias(3) | 867 |
| **Total** | | **11,011** |

Overhead vs backbone: 11,011 / 306,258 = **3.6%**

## Why Not Modify the Backbone?

We tried several approaches that failed:

| Approach | Result | Why it failed |
|----------|--------|---------------|
| SE blocks in backbone | -0.9 dB regression | Intermediate feature perturbation destroys learned representations |
| Fine-tuning backbone | -0.5 dB after 200 iters | Training pipeline mismatch (ours vs original) causes drift |
| Larger RefineNet (64ch) | Same gain, more params | Diminishing returns — the correction is tiny by nature |

Post-processing refinement works because it operates on the final SR image,
not intermediate features, and cannot break the backbone's learned patterns.
