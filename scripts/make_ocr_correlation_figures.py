"""
Build PSNR-OCR (Fig 6) and SSIM-OCR (Fig 7) figures using the ACTUAL scatter
+ binned panels from the original report, not synthetic approximations.

Approach: crop the two sub-panels (scatter, binned) from each report figure,
arrange them into a clean 2x2 grid (Restormer | Diffusion-SR3 across columns,
scatter | binned across rows).
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
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

# Each report figure has a global title and two sub-panels side by side.
# Crop: skip title (~8% from top), take full width, split in half horizontally.

def split_report_figure(fname):
    img = np.array(Image.open(os.path.join(REP_FIG, fname)).convert("RGB"))
    h, w, _ = img.shape
    title_px = int(h * 0.085)
    body = img[title_px:, :]
    bh, bw, _ = body.shape
    left  = body[:, : bw // 2]
    right = body[:, bw // 2 :]
    return left, right


def assemble_2x2(top_left_fig, bottom_left_fig, top_right_fig, bottom_right_fig,
                 col_titles, main_title, out_name):
    """
    Layout:
        col_titles[0]     col_titles[1]
        -----------       -----------
        top_left_fig      top_right_fig   (scatter)
        bottom_left_fig   bottom_right_fig (binned)
    """
    top_left, _  = split_report_figure(top_left_fig)
    bot_left, _  = split_report_figure(bottom_left_fig)
    # For the report files, left panel = scatter; right panel = binned
    _, _ = None, None
    top_right, _  = split_report_figure(top_right_fig)
    bot_right, _ = split_report_figure(bottom_right_fig)

    # Actually we need scatter = left half, binned = right half:
    top_left, top_left_binned  = split_report_figure(top_left_fig)
    top_right, top_right_binned = split_report_figure(top_right_fig)

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 6.8))

    axes[0][0].imshow(top_left)
    axes[0][1].imshow(top_right)
    axes[1][0].imshow(top_left_binned)
    axes[1][1].imshow(top_right_binned)

    col_axes_top = axes[0]
    col_axes_top[0].set_title(col_titles[0], fontsize=11, fontweight="bold",
                               color="#08519c", pad=6)
    col_axes_top[1].set_title(col_titles[1], fontsize=11, fontweight="bold",
                               color="#a50f15", pad=6)

    # Row labels on the far left of each row
    axes[0][0].set_ylabel("Scatter (raw)", fontsize=10, fontweight="bold")
    axes[1][0].set_ylabel("Binned mean $\\pm$ s.d.", fontsize=10, fontweight="bold")

    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor("#cccccc"); s.set_linewidth(0.6)

    fig.suptitle(main_title, fontsize=12, fontweight="bold", y=1.00)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, out_name + ".png"))
    fig.savefig(os.path.join(OUT_DIR, out_name + ".pdf"))
    plt.close(fig)
    print(f"Saved: {out_name}")


# PSNR-OCR figure  (uses Restormer fig43 + Diffusion fig44)
# top-left: restormer scatter; top-right: diffusion scatter
# bot-left: restormer binned;  bot-right: diffusion binned
# Here we pass the SAME file twice as "top" and "bot" to extract both halves.
def psnr_ocr():
    # Restormer left panel (scatter), right panel (binned)
    r_scatter, r_binned = split_report_figure("figure_43_restormer_ocr_vs_psnr.png")
    d_scatter, d_binned = split_report_figure("figure_44_diffusion_sr3_ocr_vs_psnr.png")

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 6.8))
    axes[0][0].imshow(r_scatter);  axes[0][0].set_title("Restormer  ($R^2 = 0.991$)",
                                                         fontsize=11, fontweight="bold",
                                                         color="#08519c", pad=6)
    axes[0][1].imshow(d_scatter);  axes[0][1].set_title("Diffusion-SR3  ($R^2 = 0.979$)",
                                                         fontsize=11, fontweight="bold",
                                                         color="#a50f15", pad=6)
    axes[1][0].imshow(r_binned)
    axes[1][1].imshow(d_binned)

    axes[0][0].set_ylabel("Scatter (raw)", fontsize=10, fontweight="bold")
    axes[1][0].set_ylabel("Binned mean $\\pm$ s.d.", fontsize=10, fontweight="bold")

    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor("#cccccc"); s.set_linewidth(0.6)

    fig.suptitle("PSNR vs. Plate-Level OCR Accuracy (Standard Dataset)",
                 fontsize=12, fontweight="bold", y=1.00)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig06_psnr_ocr_correlation.png"))
    fig.savefig(os.path.join(OUT_DIR, "fig06_psnr_ocr_correlation.pdf"))
    plt.close(fig)
    print("Saved: fig06_psnr_ocr_correlation")


def ssim_ocr():
    r_scatter, r_binned = split_report_figure("figure_45_restormer_ocr_vs_ssim.png")
    d_scatter, d_binned = split_report_figure("figure_46_diffusion_sr3_ocr_vs_ssim.png")

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 6.8))
    axes[0][0].imshow(r_scatter); axes[0][0].set_title("Restormer  ($R^2 = 0.777$)",
                                                        fontsize=11, fontweight="bold",
                                                        color="#08519c", pad=6)
    axes[0][1].imshow(d_scatter); axes[0][1].set_title("Diffusion-SR3  ($R^2 = 0.739$)",
                                                        fontsize=11, fontweight="bold",
                                                        color="#a50f15", pad=6)
    axes[1][0].imshow(r_binned)
    axes[1][1].imshow(d_binned)

    axes[0][0].set_ylabel("Scatter (raw)", fontsize=10, fontweight="bold")
    axes[1][0].set_ylabel("Binned mean $\\pm$ s.d.", fontsize=10, fontweight="bold")

    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor("#cccccc"); s.set_linewidth(0.6)

    fig.suptitle("SSIM vs. Plate-Level OCR Accuracy (Standard Dataset): "
                 "threshold-like response",
                 fontsize=11.5, fontweight="bold", y=1.00)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig07_ssim_ocr_threshold.png"))
    fig.savefig(os.path.join(OUT_DIR, "fig07_ssim_ocr_threshold.pdf"))
    plt.close(fig)
    print("Saved: fig07_ssim_ocr_threshold")


if __name__ == "__main__":
    psnr_ocr()
    ssim_ocr()
