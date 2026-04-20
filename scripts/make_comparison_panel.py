"""
Assembles a publication-quality comparison panel from the three reconstruction strips.
Shows how each method restores plates at different (alpha, beta) viewing angles.
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image
import os

OUT_DIR = os.path.join(os.path.dirname(__file__), "figures", "generated")
os.makedirs(OUT_DIR, exist_ok=True)

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "report", "figures")

CASES = [
    {"file": "figure_38_reconstructions_per_model_col3.png", "alpha": 30, "beta": 88,
     "difficulty": "Good recovery",
     "diff_color": "#2ca02c"},
    {"file": "figure_36_reconstructions_per_model_col1.png", "alpha": 85, "beta": 65,
     "difficulty": "Partial recovery",
     "diff_color": "#e67e00"},
    {"file": "figure_37_reconstructions_per_model_col2.png", "alpha": 88, "beta": 82,
     "difficulty": "Fails (beyond limit)",
     "diff_color": "#d62728"},
]

ROW_LABELS = [
    ("Distorted input", True),
    ("Ground truth", True),
    ("U-Net (baseline)", False),
    ("U-Net Conditional", False),
    ("Restormer", False),
    ("GAN pix2pix", False),
    ("Diffusion SR3", False),
]
N_ROWS = len(ROW_LABELS)
N_COLS = len(CASES)

# Each strip image has: header text, then N_ROWS blocks each with: label text + plate image
# We crop out the small label at top of each row block to show only the plate.
LABEL_FRAC = 0.32   # fraction of each row block that is the text label

TARGET_ASPECT = 4.0   # canonical plate aspect (width / height), e.g. 256x64
CANVAS_BG = (255, 255, 255)  # white padding around each plate

def _center_pad_to_aspect(arr, target_aspect=TARGET_ASPECT, bg=CANVAS_BG):
    """Pad a plate image with a neutral background so its (w / h) ratio
    equals target_aspect, without resampling. Result is the image centered
    inside a larger canvas; cropping never occurs."""
    h, w = arr.shape[:2]
    current_aspect = w / max(h, 1)
    if current_aspect < target_aspect:
        # Image too tall relative to target: add left/right padding.
        new_w = int(round(h * target_aspect))
        pad = (new_w - w) // 2
        canvas = np.full((h, new_w, 3), bg, dtype=arr.dtype)
        canvas[:, pad:pad + w] = arr
        return canvas
    else:
        # Image too wide: add top/bottom padding.
        new_h = int(round(w / target_aspect))
        pad = (new_h - h) // 2
        canvas = np.full((new_h, w, 3), bg, dtype=arr.dtype)
        canvas[pad:pad + h, :] = arr
        return canvas


def _tight_crop_plate(row_block):
    """
    Shrink a row block to the tightest bounding box containing the plate
    content (non-white, non-near-black pixels). This removes the text label
    at the top, any padding around the plate, and any bleed from the next
    row, producing a consistent crop across all source images.
    """
    h, w, _ = row_block.shape
    # Consider a pixel "content" if it is neither near-white (background,
    # label area) nor pure padding. White threshold: min RGB > 240.
    gray = row_block.mean(axis=2)
    is_bg = gray > 240
    content = ~is_bg
    # Row/col activity counts
    row_activity = content.sum(axis=1)
    col_activity = content.sum(axis=0)
    # Find row range with enough content (at least 5% of width)
    row_thr = max(1, int(0.05 * w))
    rows = np.where(row_activity > row_thr)[0]
    col_thr = max(1, int(0.02 * h))
    cols = np.where(col_activity > col_thr)[0]
    if rows.size == 0 or cols.size == 0:
        # Fall back to the central 60% vertically
        return row_block[int(0.2 * h): int(0.9 * h), :]
    # Exclude the top text-label band: discard the first contiguous cluster
    # of rows if it is short (a text label typically < 20% of row_h).
    # Simple approach: take the tallest contiguous cluster.
    groups = np.split(rows, np.where(np.diff(rows) > 2)[0] + 1)
    best_group = max(groups, key=len)
    r0, r1 = best_group[0], best_group[-1]
    c0, c1 = cols[0], cols[-1]
    return row_block[r0:r1 + 1, c0:c1 + 1]


def load_strip_plates(img_path):
    img = Image.open(img_path).convert("RGB")
    arr = np.array(img)
    h = arr.shape[0]
    header_px = int(h * 0.08)
    body = arr[header_px:, :]
    row_h = body.shape[0] // N_ROWS
    plates = []
    for i in range(N_ROWS):
        row_block = body[i * row_h: (i + 1) * row_h, :]
        plate = _tight_crop_plate(row_block)
        # Canonicalise aspect so every plate renders centered in its frame
        # at the same scale, with no clipping.
        plate = _center_pad_to_aspect(plate)
        plates.append(plate)
    return plates

strips = [load_strip_plates(os.path.join(FIG_DIR, c["file"])) for c in CASES]

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# Layout constants (inches). Plate aspect ratio is 4:1 (256x64) - row height
# is derived from column width to preserve that aspect without stretching.
LABEL_W = 1.3
COL_W = 2.5
IMG_ROW_H = COL_W / 4.0 + 0.12   # 4:1 plate + padding
HEADER_H = 0.65
GAP = 0.06
N_IMG_ROWS = N_ROWS

fig_w = LABEL_W + N_COLS * COL_W + 0.15
fig_h = HEADER_H + N_IMG_ROWS * IMG_ROW_H + 0.25

fig = plt.figure(figsize=(fig_w, fig_h))

def to_fig(x_inch, y_inch, w_inch, h_inch):
    return [x_inch / fig_w, y_inch / fig_h, w_inch / fig_w, h_inch / fig_h]

# Column headers
for ci, case in enumerate(CASES):
    x = LABEL_W + ci * COL_W
    y = fig_h - HEADER_H
    ax = fig.add_axes(to_fig(x, y, COL_W, HEADER_H))
    ax.axis("off")
    ax.set_facecolor("#f8f8f8")
    angle_str = rf"$\alpha={case['alpha']}^\circ,\;\beta={case['beta']}^\circ$"
    ax.text(0.5, 0.72, angle_str, ha="center", va="center",
            fontsize=9, fontweight="bold", transform=ax.transAxes,
            math_fontfamily="cm")
    ax.text(0.5, 0.28, case["difficulty"], ha="center", va="center",
            fontsize=8, color=case["diff_color"], style="italic",
            transform=ax.transAxes)
    rect = plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor="#cccccc",
                          linewidth=0.8, transform=ax.transAxes)
    ax.add_patch(rect)

# Row labels + images
for ri, (label, is_ref) in enumerate(ROW_LABELS):
    y = fig_h - HEADER_H - (ri + 1) * IMG_ROW_H
    # Row label
    ax_lbl = fig.add_axes(to_fig(0.02, y, LABEL_W - 0.05, IMG_ROW_H))
    ax_lbl.axis("off")
    bg = "#eeeeee" if is_ref else "white"
    ax_lbl.set_facecolor(bg)
    ax_lbl.text(0.97, 0.5, label, ha="right", va="center",
                fontsize=8, fontweight="bold" if is_ref else "normal",
                color="#222222", transform=ax_lbl.transAxes)

    for ci in range(N_COLS):
        x = LABEL_W + ci * COL_W
        ax = fig.add_axes(to_fig(x + GAP, y + GAP * 0.5, COL_W - 2 * GAP, IMG_ROW_H - GAP))
        ax.imshow(strips[ci][ri], aspect="equal", interpolation="lanczos")
        ax.set_xticks([])
        ax.set_yticks([])
        lw = 0.8 if not is_ref else 1.2
        ec = "#888888" if not is_ref else "#555555"
        for spine in ax.spines.values():
            spine.set_linewidth(lw)
            spine.set_edgecolor(ec)

# Dashed divider after ground truth row
div_y = (fig_h - HEADER_H - 2 * IMG_ROW_H) / fig_h
line = plt.Line2D(
    [LABEL_W / fig_w, 1.0], [div_y, div_y],
    transform=fig.transFigure, color="#888888",
    linewidth=0.9, linestyle="--", zorder=10)
fig.add_artist(line)

# Figure caption
fig.text(0.5, 0.004,
         "Examples of plate reconstruction by each model under three angular difficulty levels. "
         "Columns increase in geometric distortion; dashed line separates reference rows (top) "
         "from model outputs.",
         ha="center", va="bottom", fontsize=6.8,
         color="#555555", style="italic", wrap=True)

out_png = os.path.join(OUT_DIR, "fig_comparison_panel.png")
out_pdf = os.path.join(OUT_DIR, "fig_comparison_panel.pdf")
fig.savefig(out_png)
fig.savefig(out_pdf)
plt.close(fig)
print(f"Saved: {out_png}")
