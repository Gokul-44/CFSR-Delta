"""
generate_architecture_diagram.py — Publication-grade CFSR-Delta architecture figure.

Generates a pixel-perfect, CVPR/ICCV-style architecture diagram using matplotlib.
All layout is grid-aligned with precise spacing for academic publication quality.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import matplotlib.patheffects as pe
import numpy as np
import os

# ═══════════════════════════════════════════════
#  DESIGN TOKENS
# ═══════════════════════════════════════════════

# Colors — slightly stronger than before for better contrast
BLUE_FILL    = '#DEEAF6'    # backbone fill
BLUE_BORDER  = '#6A9FCA'    # backbone border
GREEN_FILL   = '#DFF0DF'    # refinenet fill
GREEN_BORDER = '#6AAA6A'    # refinenet border
INNER_FILL   = '#FAFCFF'    # inner box fill (backbone)
INNER_FILL_G = '#F5FBF5'    # inner box fill (refinenet)
WHITE        = '#FFFFFF'
BLACK        = '#191919'
DARK_GRAY    = '#3A3A3A'
MID_GRAY     = '#666666'
LIGHT_GRAY   = '#AAAAAA'
ARROW_CLR    = '#2A2A2A'
SKIP_CLR     = '#888888'

# Typography sizes
TITLE_SZ   = 12.5
SUBTITLE_SZ= 8.5
LABEL_SZ   = 8.5
INNER_SZ   = 7.0
EQ_SZ      = 10.5
CAP_SZ     = 8.5
DELTA_SZ   = 13


def draw_module(ax, x, y, w, h, fill, border, title, subtitle,
                inner_labels, inner_fill=INNER_FILL):
    """Draw a network module block with title, subtitle, and internal layers."""

    # ── Outer container ──
    outer = FancyBboxPatch(
        (x, y), w, h, boxstyle="square,pad=0",
        facecolor=fill, edgecolor=border, linewidth=1.0
    )
    ax.add_patch(outer)

    # ── Title (bold) ──
    ax.text(x + w / 2, y + h - 0.18, title,
            ha='center', va='top', fontsize=TITLE_SZ,
            fontweight='bold', color=BLACK, fontfamily='sans-serif')

    # ── Subtitle (italic, gray) ──
    ax.text(x + w / 2, y + h - 0.46, subtitle,
            ha='center', va='top', fontsize=SUBTITLE_SZ,
            color=MID_GRAY, fontfamily='sans-serif', style='italic')

    # ── Inner sub-blocks ──
    if not inner_labels:
        return

    n = len(inner_labels)
    margin = 0.15
    usable_w = w - 2 * margin
    gap = 0.10
    box_w = (usable_w - (n - 1) * gap) / n
    box_h = 0.36
    row_y = y + 0.15

    for i, label in enumerate(inner_labels):
        bx = x + margin + i * (box_w + gap)

        # Sub-block rectangle
        sub = FancyBboxPatch(
            (bx, row_y), box_w, box_h, boxstyle="square,pad=0",
            facecolor=inner_fill, edgecolor=border,
            linewidth=0.45, alpha=0.95
        )
        ax.add_patch(sub)

        # Sub-block label
        ax.text(bx + box_w / 2, row_y + box_h / 2, label,
                ha='center', va='center', fontsize=INNER_SZ,
                color=DARK_GRAY, fontfamily='sans-serif')

        # Connector arrow between sub-blocks
        if i < n - 1:
            x_start = bx + box_w + 0.01
            x_end = bx + box_w + gap - 0.01
            ax.annotate(
                '', xy=(x_end, row_y + box_h / 2),
                xytext=(x_start, row_y + box_h / 2),
                arrowprops=dict(arrowstyle='->', color=ARROW_CLR,
                                lw=0.5, shrinkA=0, shrinkB=0)
            )


def draw_arrow(ax, x1, y, x2, lw=0.8):
    """Draw a horizontal arrow at height y."""
    ax.annotate(
        '', xy=(x2, y), xytext=(x1, y),
        arrowprops=dict(arrowstyle='->', color=ARROW_CLR,
                        lw=lw, shrinkA=1, shrinkB=1)
    )


def draw_image_patch(ax, cx, cy, size, resolution):
    """Draw a square image representation with varying quality levels.

    resolution: 'low' (4×4 grid), 'mid' (8×8 smooth), 'high' (16×16 smooth)
    """
    s = size / 2
    np.random.seed(42)
    base_vals = np.random.rand(4, 4)
    cmap = plt.cm.viridis  # more professional than plasma

    if resolution == 'low':
        n = 4
        vals = base_vals
        edge_clr = '#AAAAAA'
        cell_edge = '#CCCCCC'
    elif resolution == 'mid':
        from scipy.ndimage import zoom
        n = 8
        vals = zoom(base_vals, 2, order=1)
        edge_clr = '#888888'
        cell_edge = 'none'
    else:
        from scipy.ndimage import zoom
        n = 16
        vals = np.clip(zoom(base_vals, 4, order=3), 0, 1)
        edge_clr = '#555555'
        cell_edge = 'none'

    cell = size / n
    for r in range(n):
        for c in range(n):
            color = cmap(vals[r % vals.shape[0], c % vals.shape[1]] * 0.65 + 0.15)
            rect = plt.Rectangle(
                (cx - s + c * cell, cy - s + r * cell),
                cell, cell, facecolor=color,
                edgecolor=cell_edge, linewidth=0.2
            )
            ax.add_patch(rect)

    # Border
    ax.add_patch(plt.Rectangle(
        (cx - s, cy - s), size, size,
        fill=False, edgecolor=edge_clr, linewidth=0.7
    ))


def generate_diagram(output_path):
    """Generate the complete CFSR-Delta architecture diagram."""

    # ── Canvas setup ──
    fig, ax = plt.subplots(1, 1, figsize=(16, 4.5))
    ax.set_xlim(-0.3, 15.8)
    ax.set_ylim(-1.35, 3.3)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.patch.set_facecolor(WHITE)

    MID_Y = 1.2   # vertical center of pipeline
    IMG_SZ = 0.58  # image patch size

    # ═════════════════════════════════════════
    #  1.  LR INPUT
    # ═════════════════════════════════════════
    lr_x = 0.3
    draw_image_patch(ax, lr_x, MID_Y, IMG_SZ, 'low')
    ax.text(lr_x, MID_Y - 0.58, 'LR Input',
            ha='center', va='top', fontsize=LABEL_SZ,
            color=BLACK, fontfamily='sans-serif')

    # Arrow: LR → Backbone
    draw_arrow(ax, lr_x + 0.42, MID_Y, 1.45)

    # ═════════════════════════════════════════
    #  2.  CFSR BACKBONE (FROZEN)
    # ═════════════════════════════════════════
    BB_X, BB_W, BB_H = 1.55, 4.9, 1.5
    BB_Y = MID_Y - BB_H / 2
    draw_module(ax, BB_X, BB_Y, BB_W, BB_H,
                fill=BLUE_FILL, border=BLUE_BORDER,
                title='CFSR Backbone',
                subtitle='306K params  \u00b7  Frozen',
                inner_labels=['Conv 3\u00d73', 'Res. Blocks \u00d712', 'PixelShuffle \u00d74'],
                inner_fill=INNER_FILL)

    # Frozen tag (top-right corner)
    ax.text(BB_X + BB_W - 0.12, BB_Y + BB_H - 0.1, '\u2744',
            ha='right', va='top', fontsize=9, color=BLUE_BORDER)

    # Arrow: Backbone → SR_base
    draw_arrow(ax, BB_X + BB_W + 0.05, MID_Y, BB_X + BB_W + 0.68)

    # ═════════════════════════════════════════
    #  3.  SR_base (intermediate output)
    # ═════════════════════════════════════════
    sr_x = BB_X + BB_W + 0.98
    draw_image_patch(ax, sr_x, MID_Y, IMG_SZ, 'mid')
    ax.text(sr_x, MID_Y - 0.58, r'$\mathit{SR}_{base}$',
            ha='center', va='top', fontsize=LABEL_SZ + 1,
            color=BLACK)

    # Arrow: SR_base → RefineNet
    draw_arrow(ax, sr_x + 0.42, MID_Y, 8.55)

    # ═════════════════════════════════════════
    #  4.  SKIP CONNECTION (dashed arc)
    # ═════════════════════════════════════════
    SUM_X = 13.65  # sum node center x (defined early for arc target)

    arc = FancyArrowPatch(
        (sr_x + 0.15, MID_Y + 0.38),
        (SUM_X, MID_Y + 0.38),
        connectionstyle="arc3,rad=-0.32",
        arrowstyle='->', color=SKIP_CLR,
        linestyle='--', lw=0.9, mutation_scale=10
    )
    ax.add_patch(arc)

    # Arc label
    ax.text((sr_x + SUM_X) / 2, MID_Y + 1.65,
            'skip connection',
            ha='center', va='center', fontsize=7.5,
            color=LIGHT_GRAY, fontfamily='sans-serif', style='italic')

    # ═════════════════════════════════════════
    #  5.  REFINENET (TRAINABLE)
    # ═════════════════════════════════════════
    RN_X, RN_W, RN_H = 8.65, 4.7, 1.35
    RN_Y = MID_Y - RN_H / 2
    draw_module(ax, RN_X, RN_Y, RN_W, RN_H,
                fill=GREEN_FILL, border=GREEN_BORDER,
                title='RefineNet',
                subtitle='11K params  \u00b7  Trainable',
                inner_labels=['Conv(3\u219232)', 'ReLU', 'Conv(32\u219232)', 'ReLU', 'Conv(32\u21923)'],
                inner_fill=INNER_FILL_G)

    # Arrow: RefineNet → Δ → ⊕
    draw_arrow(ax, RN_X + RN_W + 0.05, MID_Y, SUM_X - 0.28)

    # Delta label above arrow
    ax.text(RN_X + RN_W + 0.45, MID_Y + 0.22, r'$\mathbf{\Delta}$',
            ha='center', va='bottom', fontsize=DELTA_SZ,
            color=BLACK)

    # ═════════════════════════════════════════
    #  6.  SUM NODE ⊕
    # ═════════════════════════════════════════
    circle = Circle((SUM_X, MID_Y), 0.22,
                    fill=True, facecolor=WHITE,
                    edgecolor=BLACK, linewidth=0.9)
    ax.add_patch(circle)
    # Plus sign with path effects for crispness
    ax.text(SUM_X, MID_Y + 0.01, '+',
            ha='center', va='center', fontsize=13,
            fontweight='bold', color=BLACK, fontfamily='sans-serif')

    # Arrow: ⊕ → SR_final
    draw_arrow(ax, SUM_X + 0.27, MID_Y, 14.45)

    # ═════════════════════════════════════════
    #  7.  SR_final (output)
    # ═════════════════════════════════════════
    out_x = 14.8
    draw_image_patch(ax, out_x, MID_Y, IMG_SZ, 'high')
    ax.text(out_x, MID_Y - 0.58, r'$\mathit{SR}_{final}$',
            ha='center', va='top', fontsize=LABEL_SZ + 1,
            color=BLACK)

    # ═════════════════════════════════════════
    #  8.  EQUATION (bottom center)
    # ═════════════════════════════════════════
    eq_y = -0.30
    ax.text(7.75, eq_y,
            r'$SR_{final} \;=\; SR_{base} \;+\; \Delta$'
            r'$\,,\quad \Delta = \mathrm{RefineNet}(SR_{base})$',
            ha='center', va='top', fontsize=EQ_SZ,
            color=BLACK, fontfamily='serif')

    # ═════════════════════════════════════════
    #  9.  FIGURE CAPTION
    # ═════════════════════════════════════════
    ax.text(7.75, -0.72,
            r'$\bf{Fig.\;1.}$  Overview of the proposed CFSR-Delta architecture. '
            r'A frozen CFSR backbone generates $SR_{base}$, which is refined'
            '\n'
            r'by a lightweight 3-layer CNN (RefineNet) that predicts '
            r'a residual correction $\Delta$. '
            r'The final output is $SR_{base} + \Delta$.',
            ha='center', va='top', fontsize=CAP_SZ,
            color='#404040', fontfamily='serif', linespacing=1.5)

    # ═════════════════════════════════════════
    #  SAVE
    # ═════════════════════════════════════════
    fig.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor=WHITE, pad_inches=0.2)
    plt.close(fig)

    size_kb = os.path.getsize(output_path) / 1024
    print(f'Saved: {output_path}')
    print(f'Size:  {size_kb:.0f} KB  |  300 DPI')


if __name__ == '__main__':
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'assets', 'architecture_overview.png')
    generate_diagram(out)
