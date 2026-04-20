"""
Synthetic-data construction panel: four pipeline stages in two rows with
labeled arrows (a) -> (b) -> (c) -> (d).
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from PIL import Image
import os

OUT_DIR = os.path.join(os.path.dirname(__file__), "figures", "generated")
REP_FIG = os.path.join(os.path.dirname(__file__), "..", "report", "figures")

mpl.rcParams.update({
    "font.family": "serif",
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# ---- Load source strip images ----
strip_img = np.array(Image.open(os.path.join(REP_FIG,
    "figure_04_synthetic_plate_generation_strip.png")).convert("RGB"))
h, w, _ = strip_img.shape
clean  = strip_img[:, 0       : w // 3]
warped = strip_img[:, w // 3  : 2 * w // 3]
noisy  = strip_img[:, 2 * w // 3:]

_dewarped_raw = Image.open(os.path.join(REP_FIG,
    "figure_18_warped_plate_with_noise.jpeg")).convert("RGB")
# Resize to match the aspect ratio of the other three stages so the
# rendered frame has the same displayed size.
_target_h, _target_w = clean.shape[:2]
dewarped = np.array(_dewarped_raw.resize((_target_w, _target_h),
                                          Image.LANCZOS))

# ---- 2x2 grid of pipeline stages ----
fig, axes = plt.subplots(2, 2, figsize=(10.5, 5.4))

def show_stage(ax, img, title, color):
    ax.imshow(img, aspect="equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor(color); s.set_linewidth(1.4)
    ax.set_title(title, fontsize=12, color=color, fontweight="bold", pad=6)

show_stage(axes[0][0], clean,    "(a) Clean plate",        "#1a5276")
show_stage(axes[0][1], warped,   "(b) 3D warp",             "#e67e22")
show_stage(axes[1][0], noisy,    "(c) + camera artifacts",  "#c0392b")
show_stage(axes[1][1], dewarped, "(d) De-warp + resize",    "#1a7a4a")

plt.tight_layout(rect=(0.04, 0.04, 0.96, 0.96))


# Pipeline stages are shown as an implicit left-to-right, top-to-bottom reading
# order; no arrows or labels between panels (ordering is clear from titles).

out = os.path.join(OUT_DIR, "fig_synth_pipeline_panel.png")
fig.savefig(out)
fig.savefig(out.replace(".png", ".pdf"))
plt.close(fig)
print(f"Saved: {out}")
