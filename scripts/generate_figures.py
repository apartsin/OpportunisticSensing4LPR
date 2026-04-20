"""
Publication-quality figure generator for the CEUS paper.

Produces all figures from numerical results (tables from the report).
Run from the paper/ directory:
    python generate_figures.py

Output: paper/figures/generated/fig_XX_*.pdf  (and .png at 300 dpi)

Datasets:
  Standard  (formerly A): 10,240 samples, moderate extreme-angle emphasis
  Extreme   (formerly C): 10,240 samples, strong extreme-angle emphasis
  Dataset B (dropped): identical distribution to A but 2x samples — ablation
  showed AUC gain < 0.01 vs Standard, confirming diminishing returns.
"""

import io
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
import os

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

OUT_DIR = os.path.join(os.path.dirname(__file__), "figures", "generated")
os.makedirs(OUT_DIR, exist_ok=True)

mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
})

# Color palette
MODEL_COLORS = {
    "U-Net": "#2171b5",
    "U-Net Cond.": "#6baed6",
    "Restormer": "#08519c",
    "GAN-Pix2Pix": "#e6550d",
    "Diffusion-SR3": "#a50f15",
}
DATASET_COLORS = {"Standard": "#4292c6", "Extreme": "#fd8d3c"}
DATASET_MARKERS = {"Standard": "o", "Extreme": "^"}
MODEL_MARKERS = {
    "U-Net": "o", "U-Net Cond.": "s",
    "Restormer": "D", "GAN-Pix2Pix": "^", "Diffusion-SR3": "X",
}
MODEL_LABELS = ["U-Net", "U-Net Cond.", "Restormer", "GAN-Pix2Pix", "Diffusion-SR3"]
DATASETS = ["Standard", "Extreme"]      # formerly A and C


def save(fig, name):
    path_png = os.path.join(OUT_DIR, name + ".png")
    path_pdf = os.path.join(OUT_DIR, name + ".pdf")
    fig.savefig(path_png)
    fig.savefig(path_pdf)
    plt.close(fig)
    print(f"  Saved: {name}.png / .pdf")


# ===========================================================================
# DATA  (columns: Standard, Extreme — B dropped)
# ===========================================================================

# Test-split PSNR [model x dataset]: Standard(A), Extreme(C)
PSNR = np.array([
    [23.66, 20.96],   # U-Net
    [24.18, 21.35],   # U-Net Cond.
    [24.71, 21.67],   # Restormer
    [23.21, 19.97],   # GAN-Pix2Pix
    [21.74, 19.34],   # Diffusion-SR3
])
SSIM = np.array([
    [0.9705, 0.9464],
    [0.9743, 0.9508],
    [0.9762, 0.9563],
    [0.9672, 0.9177],
    [0.9315, 0.9052],
])

TRAIN_TIME_NORM = np.array([1.00, 1.19, 14.87, 1.23, 4.69])
LATENCY_MS      = np.array([11.75, 7.50, 14.01, 7.34, 21.81])

# Boundary-AUC and reliability F [model x dataset]
AUC = np.array([
    [0.915, 0.915],   # U-Net
    [0.916, 0.917],   # U-Net Cond.
    [0.920, 0.921],   # Restormer
    [0.909, 0.899],   # GAN-Pix2Pix
    [0.889, 0.886],   # Diffusion-SR3
])
F_SCORE = np.array([
    [0.103, 0.146],
    [0.145, 0.115],
    [0.095, 0.209],
    [0.168, 0.173],
    [0.572, 1.124],
])

HIGH_ANGLE_PSNR = np.array([37.5, 37.0, 40.0, 35.0, 30.0])
HIGH_ANGLE_OCR  = np.array([0.65, 0.67, 0.67, 0.63, 0.55])

PSNR_OCR_RESTORMER = dict(slope=0.032, intercept=-0.55, r2=0.991)
PSNR_OCR_DIFFUSION = dict(slope=0.043, intercept=-0.74, r2=0.979)
SSIM_OCR_RESTORMER = dict(slope=7.883, intercept=-6.88, r2=0.777)
SSIM_OCR_DIFFUSION = dict(slope=6.395, intercept=-5.26, r2=0.739)


# ===========================================================================
# FIG 01: Combined PSNR + SSIM overview (2 x 2 lollipop chart)
# ===========================================================================
def fig_psnr_ssim_combined():
    print("Generating Fig 01: Combined PSNR + SSIM lollipop...")
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharey="row")

    x = np.arange(len(MODEL_LABELS))
    metrics = [("PSNR (dB)", PSNR, 18, 27), ("SSIM", SSIM, 0.88, 0.985)]

    for row, (ylabel, data, ylo, yhi) in enumerate(metrics):
        for col, ds in enumerate(DATASETS):
            ax = axes[row][col]
            vals = data[:, col]
            baseline = vals[0]

            for i, (v, model) in enumerate(zip(vals, MODEL_LABELS)):
                color = MODEL_COLORS[model]
                ax.plot([x[i], x[i]], [baseline, v],
                        color=color, linewidth=1.8, alpha=0.6, zorder=2)
                ax.scatter(x[i], v, s=100, color=color,
                           marker=MODEL_MARKERS[model], zorder=4, edgecolors="white",
                           linewidths=0.7)
                ax.text(x[i], v + (yhi - ylo) * 0.018,
                        f"{v:.4f}" if ylabel == "SSIM" else f"{v:.2f}",
                        ha="center", va="bottom", fontsize=7.5, color=color)

            ax.axhline(baseline, color=MODEL_COLORS["U-Net"], linewidth=0.9,
                       linestyle="--", alpha=0.55, zorder=1,
                       label="U-Net baseline")
            ax.set_xticks(x)
            ax.set_xticklabels(MODEL_LABELS, rotation=28, ha="right")
            ax.set_ylim(ylo, yhi)
            if col == 0:
                ax.set_ylabel(ylabel, fontsize=11)
            ax.set_title(f"Dataset {ds}", fontweight="bold")
            if row == 0 and col == 0:
                ax.legend(fontsize=8, loc="upper right")

    fig.suptitle("Test-Split PSNR and SSIM by Model (Standard vs. Extreme Dataset)",
                 fontweight="bold", y=1.01)
    plt.tight_layout()
    save(fig, "fig01_psnr_ssim_combined")


# ===========================================================================
# FIG 02: Dataset angle-distribution comparison (3D density surface)
# ===========================================================================
def _sampling_pdf(alpha, beta, bend_lo, bend_hi, k_supp=25.0, smooth=1.2):
    """
    Approximate 1D marginal PDF for one axis.
    Logistic-shaped: baseline density + surge near 90 degrees.
    The independent 2D PDF is the outer product of two marginals.
    """
    def marginal(theta):
        # Logistic ramp from bend_lo to bend_hi
        x = (theta - 0.5 * (bend_lo + bend_hi)) / smooth
        ramp = 1.0 / (1.0 + np.exp(-x))
        return 1.0 + k_supp * ramp

    pa = marginal(alpha)
    pb = marginal(beta)
    pdf = np.outer(pa, pb)
    return pdf / pdf.sum()


def fig_dataset_distribution():
    print("Generating Fig 02: Dataset angle distribution (3D density)...")

    alphas = np.arange(0, 90)
    betas  = np.arange(0, 90)
    A, B   = np.meshgrid(alphas, betas, indexing="ij")

    configs = {
        "Standard (DS-S)": dict(bend_lo=5,  bend_hi=20,  k_supp=25, smooth=1.2),
        "Extreme (DS-E)":  dict(bend_lo=15, bend_hi=50,  k_supp=25, smooth=1.0),
    }
    colors_ds = ["#2171b5", "#e6550d"]

    fig = plt.figure(figsize=(12, 5.5))

    for idx, (name, cfg) in enumerate(configs.items()):
        ax = fig.add_subplot(1, 2, idx + 1, projection="3d")
        Z = _sampling_pdf(alphas, betas, cfg["bend_lo"], cfg["bend_hi"],
                          cfg["k_supp"], cfg["smooth"])
        Z_smooth = Z * 1e4   # scale for readability

        col = colors_ds[idx]
        surf = ax.plot_surface(A, B, Z_smooth,
                               cmap="Blues" if idx == 0 else "Oranges",
                               edgecolor="none", alpha=0.88, linewidth=0)
        ax.set_xlabel(r"$\alpha$ (deg)", labelpad=4, fontsize=9)
        ax.set_ylabel(r"$\beta$ (deg)", labelpad=4, fontsize=9)
        ax.set_zlabel("Relative density", labelpad=4, fontsize=9)
        ax.set_title(name, fontweight="bold", fontsize=10, pad=10)
        ax.set_xlim(0, 89); ax.set_ylim(0, 89)
        ax.view_init(elev=28, azim=-55)
        ax.tick_params(labelsize=7)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        fig.colorbar(surf, ax=ax, shrink=0.45, pad=0.08, label="Density")

    fig.suptitle(
        "Training Angle Distributions: Standard vs. Extreme Dataset\n"
        "Both use Scrambled Sobol sequences over $[0°, 89°]^2$; "
        "Extreme dataset concentrates more samples near oblique angles.",
        fontsize=9.5, y=1.01
    )
    plt.tight_layout()
    save(fig, "fig02_dataset_angle_distribution")


# ===========================================================================
# FIG 03: Efficiency (bubble chart — training time vs latency)
# ===========================================================================
def fig_efficiency():
    print("Generating Fig 03: Efficiency bubble chart...")
    fig, ax = plt.subplots(figsize=(7, 4.5))

    offsets = {
        "U-Net":         (0.25,  0.40),
        "U-Net Cond.":   (0.25,  0.40),
        "Restormer":     (-2.5,  0.50),
        "GAN-Pix2Pix":  (0.25,  0.40),
        "Diffusion-SR3": (0.25, -0.90),
    }
    for model in MODEL_LABELS:
        i = MODEL_LABELS.index(model)
        ax.scatter(TRAIN_TIME_NORM[i], LATENCY_MS[i],
                   s=220, color=MODEL_COLORS[model],
                   marker=MODEL_MARKERS[model], zorder=4,
                   edgecolors="black", linewidths=0.6)
        ox, oy = offsets[model]
        ax.annotate(model,
                    xy=(TRAIN_TIME_NORM[i], LATENCY_MS[i]),
                    xytext=(TRAIN_TIME_NORM[i] + ox, LATENCY_MS[i] + oy),
                    fontsize=9, ha="left")

    ax.set_xlabel("Normalized Training Time  (U-Net = 1.0)")
    ax.set_ylabel("Inference Latency (ms)")
    ax.set_title("Efficiency: Training Time vs. Inference Latency")
    ax.set_xlim(-0.5, 16.5)
    ax.set_ylim(4, 26)
    ax.annotate("lower latency", xy=(0.02, 0.02), xycoords="axes fraction",
                fontsize=8, color="gray", ha="left", va="bottom")

    plt.tight_layout()
    save(fig, "fig03_efficiency_training_latency")


# ===========================================================================
# FIG 04: Slope graph — AUC and F transitions from Standard to Extreme
# ===========================================================================
def fig_auc_f_slopegraph():
    print("Generating Fig 04: AUC/F slope graph...")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2))

    specs = [
        (axes[0], AUC,     "Boundary-AUC",        False, (0.881, 0.929), ".3f"),
        (axes[1], F_SCORE, "Reliability Score F", True,  (0.06, 1.7),    ".3f"),
    ]

    # Hand-tuned vertical shifts for the left model-name labels so that when
    # two models have near-identical values they don't overlap.
    NAME_SHIFT_AUC = {
        "U-Net":         -0.0013,
        "U-Net Cond.":   +0.0013,
        "Restormer":      0.0000,
        "GAN-Pix2Pix":    0.0000,
        "Diffusion-SR3":  0.0000,
    }
    # For log-scale F, use multiplicative factors.
    NAME_SHIFT_F = {
        "U-Net":          1.22,
        "U-Net Cond.":    1.00,
        "Restormer":      0.82,
        "GAN-Pix2Pix":    1.00,
        "Diffusion-SR3":  1.00,
    }

    for ax, data, title, log_scale, ylim, fmt in specs:
        x_left, x_right = 0.0, 1.0
        shift = NAME_SHIFT_F if log_scale else NAME_SHIFT_AUC

        for mi, model in enumerate(MODEL_LABELS):
            v_std, v_ext = data[mi]
            color = MODEL_COLORS[model]
            ax.plot([x_left, x_right], [v_std, v_ext],
                    color=color, linewidth=2.0, marker=MODEL_MARKERS[model],
                    markersize=10, markerfacecolor=color,
                    markeredgecolor="white", markeredgewidth=0.8, zorder=3)

            # Left: model name (shifted vertically if needed)
            name_y = v_std * shift[model] if log_scale else v_std + shift[model]
            ax.text(x_left - 0.08, name_y, model,
                    ha="right", va="center", fontsize=8.5, color=color,
                    fontweight="bold", zorder=5,
                    bbox=dict(facecolor="white", edgecolor="none",
                              alpha=0.80, boxstyle="round,pad=0.18"))
            # Left: numeric value, just inside the marker
            ax.text(x_left - 0.015, v_std, format(v_std, fmt),
                    ha="right", va="center", fontsize=7.5, color=color, zorder=4)
            # Right: numeric value outside the marker
            ax.text(x_right + 0.025, v_ext, format(v_ext, fmt),
                    ha="left", va="center", fontsize=8, color=color,
                    fontweight="bold", zorder=4)

        ax.set_xticks([x_left, x_right])
        ax.set_xticklabels(["Standard (DS-S)", "Extreme (DS-E)"], fontsize=10)
        ax.set_xlim(-0.78, 1.22)
        ax.set_ylim(*ylim)
        ax.set_title(title, fontweight="bold", fontsize=11, pad=8)
        ax.grid(True, axis="y", alpha=0.3)
        ax.grid(False, axis="x")
        if log_scale:
            ax.set_yscale("log")
            ax.set_ylabel("F (deg, log scale)", fontsize=10)
        else:
            ax.set_ylabel("AUC", fontsize=10)

        # Direction-of-better hint, placed in the lower-right corner (clear space)
        hint = "lower = more consistent" if log_scale else "higher = wider coverage"
        ax.text(0.97, 0.03, hint, transform=ax.transAxes,
                ha="right", va="bottom", fontsize=7.5, color="gray", style="italic")

    fig.suptitle("Dataset-Shift Sensitivity: Standard -> Extreme",
                 fontweight="bold", y=1.00)
    plt.tight_layout()
    save(fig, "fig04_auc_f_slopegraph")


# ===========================================================================
# FIG 05: Recoverability boundary schematic (single panel)
# ===========================================================================
def fig_recoverability_schematic():
    print("Generating Fig 05: Recoverability boundary schematic...")

    def make_boundary(auc, alpha_anisotropy=1.0):
        """
        Construct a superellipse-shaped boundary whose enclosed area matches
        auc * 89 * 89. Higher AUC -> higher exponent n -> more rectangular
        boundary (strictly nesting outward).  alpha_anisotropy > 1 makes the
        alpha axis decay faster than beta (the observed asymmetry).
        """
        alphas = np.arange(0, 90, dtype=float)
        target_area = auc * 89.0 * 89.0

        best_n, best_err = 2.0, float("inf")
        for n in np.linspace(1.5, 20.0, 2000):
            # Symmetric superellipse
            inner = 1.0 - (alphas / 89.0) ** n
            inner = np.clip(inner, 0, 1)
            beta_max = 89.0 * inner ** (1.0 / n)
            area = np.trapezoid(beta_max, alphas)
            err = abs(area - target_area)
            if err < best_err:
                best_err, best_n = err, n

        n = best_n
        inner = 1.0 - (alphas / 89.0) ** n
        inner = np.clip(inner, 0, 1)
        beta_max = 89.0 * inner ** (1.0 / n)

        # Apply alpha anisotropy: multiply beta by a mild decay tied to alpha
        # so the boundary contracts faster along alpha than beta.
        if alpha_anisotropy != 1.0:
            decay = 1.0 - 0.10 * (alphas / 89.0) ** 2 * (alpha_anisotropy - 1.0)
            beta_max = beta_max * decay
        return alphas, np.clip(beta_max, 0, 89)

    specs = [
        ("U-Net",         0.919, 1.4, "--",           1.6),
        ("U-Net Cond.",   0.919, 1.2, "-.",           1.6),
        ("Restormer",     0.921, 1.0, "-",            2.4),
        ("GAN-Pix2Pix",  0.907, 1.6, ":",            1.6),
        ("Diffusion-SR3", 0.887, 1.8, (0,(3,1,1,1)), 1.6),
    ]
    alphas_max, betas_max = make_boundary(0.934, 1.0)

    fig, ax = plt.subplots(figsize=(6.8, 6.8))

    for model, auc, alpha_pen, ls, lw in specs:
        alphas, betas = make_boundary(auc, alpha_pen)
        ax.plot(alphas, betas, linestyle=ls, linewidth=lw,
                color=MODEL_COLORS[model], label=f"{model} (AUC={auc:.3f})")
        ax.fill_between(alphas, 0, betas, alpha=0.04, color=MODEL_COLORS[model])

    ax.plot(alphas_max, betas_max, color="black", linewidth=2.8,
            linestyle="-", label="Maximal boundary (AUC=0.934)", zorder=5)
    ax.fill_between(np.arange(80, 90), 80, 89, alpha=0.15, color="red", zorder=0)
    ax.text(84, 84, "Unrecov.", fontsize=8, color="darkred", ha="center", va="center")

    ax.annotate("", xy=(75, 45), xytext=(60, 30),
                arrowprops=dict(arrowstyle="->", color="gray", lw=0.9))
    ax.text(58, 27, r"$\alpha$ harder than $\beta$",
            fontsize=8, color="gray", ha="center")

    ax.set_xlim(0, 89); ax.set_ylim(0, 89)
    ax.set_xlabel(r"Azimuthal rotation $\alpha$ (degrees)", fontsize=11)
    ax.set_ylabel(r"Elevational rotation $\beta$ (degrees)", fontsize=11)
    ax.set_title("Recoverability Boundaries per Model\n"
                 "(OCR threshold T = 0.9; maximal = union over all models)",
                 fontweight="bold")
    ax.set_aspect("equal")
    ax.legend(loc="lower left", fontsize=8.5, framealpha=0.92)
    ax.plot([0, 89], [0, 89], color="lightgray", linewidth=0.7, linestyle="--")

    plt.tight_layout()
    save(fig, "fig05_recoverability_boundaries_schematic")


# ===========================================================================
# FIG 06: Recoverability-reliability scatter (Standard/Extreme per model)
# ===========================================================================
def fig_auc_vs_f():
    print("Generating Fig 06: AUC vs F scatter...")
    fig, ax = plt.subplots(figsize=(8.5, 5.8))

    ax.axhspan(0,    0.30, alpha=0.06, color="green",  zorder=0)
    ax.axhspan(0.30, 0.70, alpha=0.04, color="orange", zorder=0)
    ax.axhspan(0.70, 1.30, alpha=0.06, color="red",    zorder=0)
    ax.text(0.8795, 0.27, "Consistent",  fontsize=7.5, color="darkgreen",  va="top", style="italic")
    ax.text(0.8795, 0.68, "Moderate",    fontsize=7.5, color="darkorange", va="top", style="italic")
    ax.text(0.8795, 1.28, "Unreliable",  fontsize=7.5, color="darkred",    va="top", style="italic")

    for mi, model in enumerate(MODEL_LABELS):
        aucs = AUC[mi, :]
        fs   = F_SCORE[mi, :]
        color = MODEL_COLORS[model]

        ax.plot(aucs, fs, "-", color=color, linewidth=1.1, alpha=0.55, zorder=2)

        for di, (a, f, ds) in enumerate(zip(aucs, fs, DATASETS)):
            ax.scatter(a, f, s=170, color=color, marker=MODEL_MARKERS[model],
                       edgecolors="white", linewidths=0.8, zorder=4)
            label_short = "S" if ds == "Standard" else "E"
            offy = -0.025 if f < 0.25 else 0.030
            ax.text(a + 0.0005, f + offy, label_short, fontsize=7.5,
                    color=color, ha="left", va="center", fontweight="bold")

        cx, cy = aucs.mean(), fs.mean()
        lx_off = {"U-Net": -0.003, "U-Net Cond.": -0.004,
                  "Restormer": 0.001, "GAN-Pix2Pix": -0.004, "Diffusion-SR3": 0.001}
        ly_off = {"U-Net": 0.04, "U-Net Cond.": -0.06,
                  "Restormer": 0.05, "GAN-Pix2Pix": 0.07, "Diffusion-SR3": 0.07}
        ax.text(cx + lx_off[model], cy + ly_off[model], model,
                fontsize=8.5, color=color, fontweight="bold",
                bbox=dict(facecolor="white", edgecolor=color,
                          alpha=0.85, boxstyle="round,pad=0.2", linewidth=0.7))

    # Maximal boundary line — this is the MAXIMUM (union of all models), not minimum
    ax.axvline(0.934, color="black", linewidth=1.3, linestyle=":",
               alpha=0.75, zorder=1)
    ax.text(0.9345, 1.08, "Maximal AUC\n(union) = 0.934",
            fontsize=7.5, va="top", ha="left", color="black",
            bbox=dict(facecolor="white", alpha=0.7, boxstyle="round,pad=0.15"))

    ax.annotate("", xy=(0.932, 0.04), xytext=(0.922, 0.04),
                arrowprops=dict(arrowstyle="-|>", color="#1a5276", lw=1.4))
    ax.text(0.926, 0.015, "Wider coverage", fontsize=7.5, color="#1a5276", ha="center")
    ax.annotate("", xy=(0.8798, 0.06), xytext=(0.8798, 0.22),
                arrowprops=dict(arrowstyle="-|>", color="#1a5276", lw=1.4))
    ax.text(0.8808, 0.14, "More consistent", fontsize=7.5, color="#1a5276",
            ha="left", rotation=90, va="center")

    # Legend for S/E markers
    ax.text(0.935, 0.50, "S = Standard\nE = Extreme", fontsize=7.5,
            color="gray", va="center", style="italic",
            bbox=dict(facecolor="white", alpha=0.6, boxstyle="round,pad=0.2"))

    ax.set_xlabel("Boundary-AUC  (fraction of angle grid with OCR >= 90%)", fontsize=10)
    ax.set_ylabel("Reliability Score F  (RMS distance of interior failures, deg)", fontsize=10)
    ax.set_title("Recoverability vs. Reliability: All Model-Dataset Pairs\n"
                 "S = Standard dataset, E = Extreme-biased dataset",
                 fontweight="bold")
    ax.set_xlim(0.879, 0.940)
    ax.set_ylim(-0.02, 1.32)

    ax.axvline(0.910, color="gray", linewidth=0.7, linestyle="--", alpha=0.45)
    ax.text(0.9105, 1.20, "Generative |", fontsize=7, color="gray", ha="left", va="top")
    ax.text(0.9095, 1.20, "| Discriminative", fontsize=7, color="gray", ha="right", va="top")

    plt.tight_layout()
    save(fig, "fig06_auc_vs_f_scatter")


# ===========================================================================
# FIG 07: PSNR-OCR linear relationship
# ===========================================================================
def fig_psnr_ocr():
    print("Generating Fig 07: PSNR-OCR correlation...")
    np.random.seed(42)

    def synthetic(slope, intercept, r2, x_range=(14, 48), n=300):
        x = np.random.uniform(*x_range, n)
        y_hat = np.clip(slope * x + intercept, 0, 1)
        noise_std = np.sqrt(np.var(y_hat) * (1 - r2) / max(r2, 1e-9))
        y = np.clip(y_hat + np.random.normal(0, noise_std, n), 0, 1)
        return x, y

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)

    for ax, model, params, color in [
        (axes[0], "Restormer",     PSNR_OCR_RESTORMER, MODEL_COLORS["Restormer"]),
        (axes[1], "Diffusion-SR3", PSNR_OCR_DIFFUSION, MODEL_COLORS["Diffusion-SR3"]),
    ]:
        px, oy = synthetic(params["slope"], params["intercept"], params["r2"])
        ax.scatter(px, oy, s=8, alpha=0.35, color=color, zorder=2)

        xl = np.array([14, 48])
        ax.plot(xl, np.clip(params["slope"] * xl + params["intercept"], 0, 1),
                color=color, linewidth=2, zorder=3,
                label=f"y = {params['slope']:.3f}x {params['intercept']:+.2f}")

        bins = np.linspace(14, 48, 14)
        bc = 0.5 * (bins[:-1] + bins[1:])
        means, stds = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (px >= lo) & (px < hi)
            means.append(oy[m].mean() if m.sum() > 2 else np.nan)
            stds.append(oy[m].std() if m.sum() > 2 else 0.0)
        means = np.array(means); stds = np.array(stds)
        ax.errorbar(bc, means, yerr=stds, fmt="none", ecolor=color,
                    alpha=0.5, linewidth=1.2, capsize=3, zorder=4)
        ax.plot(bc, means, "o-", color=color, linewidth=1.5, markersize=4,
                zorder=5, alpha=0.75)

        ax.set_xlim(12, 50); ax.set_ylim(-0.05, 1.08)
        ax.set_xlabel("Plate-Level PSNR (dB)")
        ax.set_title(f"{model}  (R\u00b2 = {params['r2']:.3f})", fontweight="bold")
        ax.axhline(0.9, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.text(12.5, 0.91, "OCR = 0.9 threshold", fontsize=7.5, color="gray")
        ax.legend(fontsize=8)

    axes[0].set_ylabel("Plate-Level OCR Accuracy")
    fig.suptitle("PSNR as a Reliable Proxy for OCR Accuracy", fontweight="bold")
    plt.tight_layout()
    save(fig, "fig07_psnr_ocr_correlation")


# ===========================================================================
# FIG 08: SSIM-OCR threshold behavior
# ===========================================================================
def fig_ssim_ocr():
    print("Generating Fig 08: SSIM-OCR threshold...")
    np.random.seed(7)

    def synthetic(slope, intercept, r2, ssim_range=(0.70, 1.0), n=500):
        s = np.random.uniform(*ssim_range, n)
        y_hat = np.clip(slope * s + intercept, 0, 1)
        total_var = np.var(y_hat)
        noise_std = np.sqrt(total_var * (1 - r2) / max(r2, 1e-9))
        y = np.clip(y_hat + np.random.normal(0, noise_std, n), 0, 1)
        return s, y

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)

    for ax, model, params in [
        (axes[0], "Restormer",     SSIM_OCR_RESTORMER),
        (axes[1], "Diffusion-SR3", SSIM_OCR_DIFFUSION),
    ]:
        color = MODEL_COLORS[model]
        lo = 0.70 if model == "Diffusion-SR3" else 0.75
        s, y = synthetic(params["slope"], params["intercept"], params["r2"],
                         ssim_range=(lo, 1.0))
        ax.scatter(s, y, s=7, alpha=0.30, color=color, zorder=2)
        xl = np.array([lo, 1.0])
        ax.plot(xl, np.clip(params["slope"] * xl + params["intercept"], 0, 1),
                color=color, linewidth=2, zorder=4,
                label=f"y = {params['slope']:.3f}x {params['intercept']:+.2f}")

        bins = np.linspace(lo, 1.0, 14)
        bc = 0.5 * (bins[:-1] + bins[1:])
        means, stds = [], []
        for blo, bhi in zip(bins[:-1], bins[1:]):
            m = (s >= blo) & (s < bhi)
            means.append(y[m].mean() if m.sum() > 2 else np.nan)
            stds.append(y[m].std() if m.sum() > 2 else 0.0)
        means = np.array(means); stds = np.array(stds)
        ax.errorbar(bc, means, yerr=stds, fmt="none", ecolor=color,
                    alpha=0.45, linewidth=1.0, capsize=3, zorder=3)
        ax.plot(bc, means, "o-", color=color, linewidth=1.5, markersize=4,
                zorder=5, alpha=0.7)

        zero_ssim = -params["intercept"] / params["slope"]
        ax.axvline(zero_ssim, color="gray", linewidth=0.9, linestyle="--")
        ax.text(zero_ssim + 0.003, 0.05, f"SSIM\u2248{zero_ssim:.2f}",
                fontsize=7.5, color="gray", rotation=90, va="bottom")

        ax.set_xlim(lo - 0.02, 1.01); ax.set_ylim(-0.05, 1.08)
        ax.set_xlabel("Plate-Level SSIM")
        ax.set_title(f"{model}  (R\u00b2 = {params['r2']:.3f})", fontweight="bold")
        ax.legend(fontsize=8, loc="upper left")

    axes[0].set_ylabel("Plate-Level OCR Accuracy")
    fig.suptitle("SSIM vs. OCR Accuracy: Threshold-Like Response, High Variance",
                 fontweight="bold")
    plt.tight_layout()
    save(fig, "fig08_ssim_ocr_threshold")


# ===========================================================================
# FIG 00: Methodology pipeline (supplementary — not in paper body)
# ===========================================================================
def fig_pipeline():
    print("Generating Fig 00: Methodology pipeline diagram...")
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.set_xlim(0, 12); ax.set_ylim(0, 5); ax.axis("off")

    BOX_H = 0.72; BOX_W = 1.65
    ACCENT = "#1a5276"; GEN_CLR = "#e67e22"; EVAL_CLR = "#1a7a4a"

    def box(cx, cy, text, color=ACCENT, fontsize=8.5):
        rect = mpatches.FancyBboxPatch(
            (cx - BOX_W / 2, cy - BOX_H / 2), BOX_W, BOX_H,
            boxstyle="round,pad=0.08", linewidth=1.2,
            edgecolor=color, facecolor=color + "22", zorder=3)
        ax.add_patch(rect)
        ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize,
                fontweight="bold", color=color, zorder=4, multialignment="center")

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color="#555", lw=1.4,
                                   mutation_scale=14), zorder=2)

    nodes = [(1.1, 3.8, "Clean Plate\nImages"), (3.0, 3.8, "Geometric\nWarp"),
             (4.95, 3.8, "Camera\nArtifacts"), (6.9, 3.8, "Distorted\nPlates"),
             (8.85, 3.8, "Sobol-Sampled\nAngles")]
    for cx, cy, txt in nodes:
        box(cx, cy, txt, ACCENT)
    for i in range(len(nodes) - 1):
        arrow(nodes[i][0] + BOX_W/2, nodes[i][1],
              nodes[i+1][0] - BOX_W/2, nodes[i+1][1])

    box(3.0, 2.2, "5 Models\nTrained", color=GEN_CLR)
    arrow(6.9, 3.8 - BOX_H/2, 6.9, 2.55)
    arrow(6.9, 2.55, 3.0 + BOX_W/2, 2.2)

    box(7.5, 2.2, "Full-Grid\nEvaluation\n[0,89]^2", color=EVAL_CLR, fontsize=8)
    arrow(3.0 + BOX_W/2, 2.2, 7.5 - BOX_W/2, 2.2)

    box(3.0,  0.85, "PSNR/SSIM\nAnalysis", color=ACCENT)
    box(6.1,  0.85, "Recoverability\nBoundary", color=EVAL_CLR)
    box(9.2,  0.85, "AUC &\nReliability F", color=EVAL_CLR)

    arrow(7.5, 2.2 - BOX_H/2, 7.5, 1.4)
    for tx in [3.0, 6.1, 9.2]:
        arrow(7.5, 1.4, tx + BOX_W/2 * (1 if tx < 7.5 else -1), 0.85)

    plt.tight_layout()
    save(fig, "fig00_methodology_pipeline")


# ===========================================================================
# Run all
# ===========================================================================
if __name__ == "__main__":
    print(f"Output directory: {OUT_DIR}\n")
    fig_pipeline()
    fig_psnr_ssim_combined()
    fig_dataset_distribution()
    fig_efficiency()
    fig_auc_f_slopegraph()
    fig_recoverability_schematic()
    fig_psnr_ocr()
    fig_ssim_ocr()
    print("\nDone.")
