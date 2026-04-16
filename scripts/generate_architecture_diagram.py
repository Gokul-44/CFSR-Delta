"""
generate_architecture_diagram.py — Programmatic CVPR-grade architecture figure.

Produces a pixel-perfect architecture diagram using matplotlib,
with precise control over layout, fonts, colors, and alignment.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import matplotlib.patheffects as pe
import numpy as np
import os

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

# Colors (muted, academic)
BLUE_FILL   = '#E8EFF8'
BLUE_BORDER = '#8EAACC'
GREEN_FILL  = '#E6F2E6'
GREEN_BORDER= '#7EAA7E'
WHITE       = '#FFFFFF'
BLACK       = '#1A1A1A'
GRAY        = '#777777'
LIGHT_GRAY  = '#BBBBBB'
ARROW_COLOR = '#333333'

# Fonts
TITLE_SIZE  = 11
SUB_SIZE    = 8.5
LABEL_SIZE  = 8
INNER_SIZE  = 7.5
CAP_SIZE    = 8.5
EQ_SIZE     = 10


def draw_block(ax, x, y, w, h, fill, border, title, subtitle, inner_labels,
               lw=0.8):
    """Draw a module block with title, subtitle, and inner sub-blocks."""
    # Outer rectangle
    rect = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="square,pad=0",
        facecolor=fill, edgecolor=border, linewidth=lw
    )
    ax.add_patch(rect)

    # Title
    ax.text(x + w/2, y + h - 0.22, title,
            ha='center', va='top', fontsize=TITLE_SIZE,
            fontweight='bold', color=BLACK, fontfamily='sans-serif')

    # Subtitle
    ax.text(x + w/2, y + h - 0.48, subtitle,
            ha='center', va='top', fontsize=SUB_SIZE,
            color=GRAY, fontfamily='sans-serif', style='italic')

    # Inner sub-blocks
    if inner_labels:
        n = len(inner_labels)
        pad = 0.12
        total_inner_w = w - 2 * pad
        gap = 0.08
        box_w = (total_inner_w - (n - 1) * gap) / n
        box_h = 0.38
        inner_y = y + 0.18

        for i, label in enumerate(inner_labels):
            bx = x + pad + i * (box_w + gap)
            inner_rect = FancyBboxPatch(
                (bx, inner_y), box_w, box_h,
                boxstyle="square,pad=0",
                facecolor=WHITE, edgecolor=border,
                linewidth=0.5, alpha=0.9
            )
            ax.add_patch(inner_rect)
            ax.text(bx + box_w/2, inner_y + box_h/2, label,
                    ha='center', va='center', fontsize=INNER_SIZE,
                    color=BLACK, fontfamily='sans-serif')

            # Arrow between inner blocks
            if i < n - 1:
                ax.annotate('', xy=(bx + box_w + gap * 0.15, inner_y + box_h/2),
                           xytext=(bx + box_w + gap * 0.85, inner_y + box_h/2),
                           arrowprops=dict(arrowstyle='->', color=ARROW_COLOR,
                                          lw=0.6, shrinkA=0, shrinkB=0))


def draw_arrow(ax, x1, y1, x2, y2, style='->', color=ARROW_COLOR, lw=0.8,
               linestyle='-'):
    """Draw a straight arrow."""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color,
                                lw=lw, linestyle=linestyle,
                                shrinkA=1, shrinkB=1))


def draw_image_placeholder(ax, cx, cy, size, quality='low'):
    """Draw a small image representation (colored grid for LR, smooth for HR)."""
    s = size / 2
    if quality == 'low':
        # Pixelated grid
        n = 4
        cell = size / n
        cmap = plt.cm.plasma
        np.random.seed(42)
        for r in range(n):
            for c in range(n):
                val = np.random.rand()
                color = cmap(val * 0.6 + 0.2)
                rect = plt.Rectangle(
                    (cx - s + c * cell, cy - s + r * cell),
                    cell, cell, facecolor=color, edgecolor='#CCCCCC',
                    linewidth=0.3
                )
                ax.add_patch(rect)
        # Border
        ax.add_patch(plt.Rectangle(
            (cx - s, cy - s), size, size,
            fill=False, edgecolor='#999999', linewidth=0.6
        ))
    elif quality == 'medium':
        # Smoother gradient
        n = 8
        cell = size / n
        cmap = plt.cm.plasma
        np.random.seed(42)
        base = np.random.rand(4, 4)
        from scipy.ndimage import zoom
        smooth = zoom(base, 2, order=1)
        for r in range(n):
            for c in range(n):
                val = smooth[r, c]
                color = cmap(val * 0.6 + 0.2)
                rect = plt.Rectangle(
                    (cx - s + c * cell, cy - s + r * cell),
                    cell, cell, facecolor=color, edgecolor='none',
                    linewidth=0
                )
                ax.add_patch(rect)
        ax.add_patch(plt.Rectangle(
            (cx - s, cy - s), size, size,
            fill=False, edgecolor='#888888', linewidth=0.6
        ))
    else:  # high
        # Very smooth
        n = 16
        cell = size / n
        cmap = plt.cm.plasma
        np.random.seed(42)
        base = np.random.rand(4, 4)
        from scipy.ndimage import zoom
        smooth = zoom(base, 4, order=3)
        for r in range(n):
            for c in range(n):
                val = np.clip(smooth[r, c], 0, 1)
                color = cmap(val * 0.6 + 0.2)
                rect = plt.Rectangle(
                    (cx - s + c * cell, cy - s + r * cell),
                    cell, cell, facecolor=color, edgecolor='none',
                    linewidth=0
                )
                ax.add_patch(rect)
        ax.add_patch(plt.Rectangle(
            (cx - s, cy - s), size, size,
            fill=False, edgecolor='#666666', linewidth=0.6
        ))


def generate_diagram(output_path):
    """Generate the full CFSR-Delta architecture diagram."""

    fig, ax = plt.subplots(1, 1, figsize=(16, 4.2))
    ax.set_xlim(-0.5, 15.5)
    ax.set_ylim(-1.2, 3.2)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.patch.set_facecolor(WHITE)

    Y_CENTER = 1.2  # vertical center of pipeline

    # ── 1. LR Input ──
    img_size = 0.55
    lr_cx = 0.3
    draw_image_placeholder(ax, lr_cx, Y_CENTER, img_size, quality='low')
    ax.text(lr_cx, Y_CENTER - 0.55, 'LR Input',
            ha='center', va='top', fontsize=LABEL_SIZE,
            color=BLACK, fontfamily='sans-serif')

    # ── Arrow: LR → Backbone ──
    draw_arrow(ax, lr_cx + 0.4, Y_CENTER, 1.4, Y_CENTER)

    # ── 2. CFSR Backbone ──
    bb_x, bb_w, bb_h = 1.5, 4.8, 1.5
    bb_y = Y_CENTER - bb_h / 2
    draw_block(ax, bb_x, bb_y, bb_w, bb_h,
               fill=BLUE_FILL, border=BLUE_BORDER,
               title='CFSR Backbone',
               subtitle='306K params  ·  Frozen',
               inner_labels=['Conv 3×3', 'Res. Blocks ×12', 'PixelShuffle ×4'])

    # Lock icon (simple text)
    ax.text(bb_x + bb_w - 0.15, bb_y + bb_h - 0.15, '[frozen]',
            ha='right', va='top', fontsize=6.5,
            color=GRAY, fontfamily='sans-serif', style='italic')

    # ── Arrow: Backbone → SR_base ──
    draw_arrow(ax, bb_x + bb_w + 0.05, Y_CENTER, bb_x + bb_w + 0.65, Y_CENTER)

    # ── 3. SR_base ──
    sr_cx = bb_x + bb_w + 0.95
    draw_image_placeholder(ax, sr_cx, Y_CENTER, img_size, quality='medium')
    ax.text(sr_cx, Y_CENTER - 0.55, 'SR$_{base}$',
            ha='center', va='top', fontsize=LABEL_SIZE + 0.5,
            color=BLACK, fontfamily='serif')

    # ── Arrow: SR_base → RefineNet ──
    draw_arrow(ax, sr_cx + 0.4, Y_CENTER, 8.3, Y_CENTER)

    # ── Skip connection (dashed arc) ──
    arc_start_x = sr_cx + 0.15
    arc_end_x = 13.45
    arc_peak_y = Y_CENTER + 1.55

    # Draw the dashed arc as a curved arrow
    arc = FancyArrowPatch(
        (arc_start_x, Y_CENTER + 0.35),
        (arc_end_x, Y_CENTER + 0.45),
        connectionstyle=f"arc3,rad=-0.35",
        arrowstyle='->', color=LIGHT_GRAY,
        linestyle='--', lw=0.9,
        mutation_scale=10
    )
    ax.add_patch(arc)

    # Label on arc
    ax.text((arc_start_x + arc_end_x) / 2, Y_CENTER + 1.55,
            'skip connection',
            ha='center', va='center', fontsize=7,
            color=GRAY, fontfamily='sans-serif', style='italic')

    # ── 4. RefineNet ──
    rn_x, rn_w, rn_h = 8.4, 4.5, 1.3
    rn_y = Y_CENTER - rn_h / 2
    draw_block(ax, rn_x, rn_y, rn_w, rn_h,
               fill=GREEN_FILL, border=GREEN_BORDER,
               title='RefineNet',
               subtitle='11K params  ·  Trainable',
               inner_labels=['Conv(3→32)', 'ReLU', 'Conv(32→32)', 'ReLU', 'Conv(32→3)'])

    # ── Arrow: RefineNet → Delta label → ⊕ ──
    draw_arrow(ax, rn_x + rn_w + 0.05, Y_CENTER, 13.2, Y_CENTER)

    # Delta label
    ax.text(rn_x + rn_w + 0.35, Y_CENTER + 0.2, 'Δ',
            ha='center', va='bottom', fontsize=11,
            fontweight='bold', color=BLACK, fontfamily='serif')

    # ── 5. Sum node ⊕ ──
    sum_cx = 13.45
    circle = Circle((sum_cx, Y_CENTER), 0.22,
                    fill=False, edgecolor=BLACK, linewidth=0.8)
    ax.add_patch(circle)
    ax.text(sum_cx, Y_CENTER, '+',
            ha='center', va='center', fontsize=12,
            fontweight='bold', color=BLACK, fontfamily='sans-serif')

    # ── Arrow: ⊕ → SR_final ──
    draw_arrow(ax, sum_cx + 0.27, Y_CENTER, 14.2, Y_CENTER)

    # ── 6. SR_final ──
    out_cx = 14.55
    draw_image_placeholder(ax, out_cx, Y_CENTER, img_size, quality='high')
    ax.text(out_cx, Y_CENTER - 0.55, 'SR$_{final}$',
            ha='center', va='top', fontsize=LABEL_SIZE + 0.5,
            color=BLACK, fontfamily='serif')

    # ── 7. Equation ──
    ax.text(7.5, -0.35,
            r'$SR_{final} = SR_{base} + \Delta$,    where  $\Delta = \mathrm{RefineNet}(SR_{base})$',
            ha='center', va='top', fontsize=EQ_SIZE,
            color=BLACK, fontfamily='serif', style='italic')

    # ── 8. Figure caption ──
    ax.text(7.5, -0.75,
            'Fig. 1.  Overview of the proposed CFSR-Delta architecture. '
            'A frozen CFSR backbone produces $SR_{base}$, which is refined\n'
            'by a lightweight 3-layer CNN (RefineNet) that predicts '
            'a residual correction $\\Delta$. The final output is '
            '$SR_{base} + \\Delta$.',
            ha='center', va='top', fontsize=CAP_SIZE,
            color='#444444', fontfamily='serif',
            linespacing=1.4)

    # ── Save ──
    fig.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor=WHITE, pad_inches=0.15)
    plt.close(fig)
    print(f'Saved: {output_path}')
    print(f'Size: {os.path.getsize(output_path) / 1024:.0f} KB')


if __name__ == '__main__':
    out = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                       'assets', 'architecture_overview.png')
    generate_diagram(out)
