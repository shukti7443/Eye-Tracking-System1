from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms


def _confidence_ellipse(x, y, ax, n_std=2.0, **kwargs):
    """Draw a confidence ellipse for (x, y) scatter."""
    if len(x) < 3:
        return
    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1] + 1e-9)
    rx = np.sqrt(1 + pearson)
    ry = np.sqrt(1 - pearson)
    ellipse = Ellipse(
        (0, 0), width=rx * 2, height=ry * 2,
        facecolor="none", **kwargs
    )
    scale_x = np.sqrt(cov[0, 0]) * n_std
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_x, mean_y = np.mean(x), np.mean(y)
    transform = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)
    ellipse.set_transform(transform + ax.transData)
    ax.add_patch(ellipse)


def plot_error_scatter(
    targets: List[Tuple[float, float]],
    gaze_per_target: List[Tuple[List, List]],
    output_path: Optional[str] = None,
    title: str = "Per-Target Gaze Error",
    dpi: int = 150,
) -> plt.Figure:
    """
    Plot gaze samples around each target with error ellipses.

    Args:
        targets: List of (tx, ty) target positions
        gaze_per_target: List of ([gaze_x], [gaze_y]) per target
        output_path: Optional save path
        title: Plot title

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(12, 7), dpi=dpi)
    ax.set_facecolor("#f8f9fa")

    colors = plt.cm.tab10.colors

    for i, ((tx, ty), (gx_list, gy_list)) in enumerate(zip(targets, gaze_per_target)):
        color = colors[i % len(colors)]
        gx = np.array(gx_list)
        gy = np.array(gy_list)

        if len(gx) == 0:
            continue

        ax.scatter(gx, gy, c=[color], alpha=0.25, s=8, zorder=2)
        ax.scatter(np.mean(gx), np.mean(gy), c=[color], s=60,
                   marker="x", linewidths=2, zorder=4)
        ax.scatter(tx, ty, c="black", s=120, marker="o", zorder=5)
        ax.scatter(tx, ty, c=[color], s=60, marker="o", zorder=6)
        ax.plot([tx, np.mean(gx)], [ty, np.mean(gy)],
                color=color, linewidth=1.2, alpha=0.7, zorder=3)

        if len(gx) >= 3:
            _confidence_ellipse(gx, gy, ax, n_std=2.0,
                                edgecolor=color, linewidth=1.5, linestyle="--", alpha=0.6)

        ax.annotate(
            f"T{i+1}", (tx, ty), fontsize=7, ha="center", va="bottom",
            color="black", fontweight="bold",
            xytext=(0, 8), textcoords="offset points"
        )

    target_patch = mpatches.Patch(color="black", label="Target")
    gaze_patch = mpatches.Patch(color="gray", alpha=0.5, label="Gaze samples")
    ax.legend(handles=[target_patch, gaze_patch], loc="lower right")

    ax.set_title(title, fontsize=14, pad=12)
    ax.set_xlabel("Screen X (px)")
    ax.set_ylabel("Screen Y (px)")
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        print(f"[Scatter] Saved → {output_path}")

    return fig


def plot_bland_altman(
    errors_device_a: List[float],
    errors_device_b: List[float],
    label_a: str = "Device A",
    label_b: str = "Device B",
    output_path: Optional[str] = None,
    dpi: int = 150,
) -> plt.Figure:
    """
    Bland-Altman agreement plot between two devices.

    X-axis: mean of both measurements
    Y-axis: difference (A - B)
    Dashed lines: mean +/- 1.96 SD (limits of agreement)
    """
    a = np.array(errors_device_a)
    b = np.array(errors_device_b)
    means = (a + b) / 2
    diffs = a - b
    mean_diff = np.mean(diffs)
    sd_diff = np.std(diffs, ddof=1)
    loa_upper = mean_diff + 1.96 * sd_diff
    loa_lower = mean_diff - 1.96 * sd_diff

    fig, ax = plt.subplots(figsize=(10, 6), dpi=dpi)
    ax.scatter(means, diffs, alpha=0.6, s=40, color="#2196F3", zorder=3)
    ax.axhline(mean_diff, color="#F44336", linewidth=1.5,
               linestyle="-", label=f"Mean diff: {mean_diff:.1f} px")
    ax.axhline(loa_upper, color="#FF9800", linewidth=1.2,
               linestyle="--", label=f"+1.96 SD: {loa_upper:.1f} px")
    ax.axhline(loa_lower, color="#FF9800", linewidth=1.2,
               linestyle="--", label=f"-1.96 SD: {loa_lower:.1f} px")

    ax.fill_between(
        ax.get_xlim(), loa_lower, loa_upper,
        alpha=0.08, color="#FF9800"
    )

    ax.set_title(f"Bland-Altman: {label_a} vs {label_b}", fontsize=14)
    ax.set_xlabel(f"Mean Error — ({label_a} + {label_b}) / 2 (px)")
    ax.set_ylabel(f"Difference — {label_a} - {label_b} (px)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        print(f"[BlandAltman] Saved → {output_path}")

    return fig
