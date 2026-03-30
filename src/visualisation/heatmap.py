from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


def plot_heatmap(
    gaze_x: List[float],
    gaze_y: List[float],
    screen_width: int = 1920,
    screen_height: int = 1080,
    targets: Optional[List[Tuple[float, float]]] = None,
    sigma: float = 30.0,
    output_path: Optional[str] = None,
    title: str = "Gaze Heatmap",
    dpi: int = 150,
) -> plt.Figure:
    """
    Generate a kernel-density heatmap of gaze samples.

    Args:
        gaze_x, gaze_y: Gaze position arrays (pixels)
        screen_width, screen_height: Screen resolution
        targets: Optional list of (x, y) target positions to overlay
        sigma: Gaussian blur radius in pixels
        output_path: If given, save the figure to this path
        title: Plot title
        dpi: Output DPI

    Returns:
        matplotlib Figure
    """
    gaze_x = np.array(gaze_x)
    gaze_y = np.array(gaze_y)

    bins_x = max(int(screen_width / 10), 100)
    bins_y = max(int(screen_height / 10), 80)
    heatmap, xedges, yedges = np.histogram2d(
        gaze_x, gaze_y,
        bins=(bins_x, bins_y),
        range=[[0, screen_width], [0, screen_height]]
    )
    heatmap = gaussian_filter(heatmap.T, sigma=sigma / 10)

    fig, ax = plt.subplots(figsize=(12, 7), dpi=dpi)
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#1a1a2e")

    im = ax.imshow(
        heatmap,
        extent=[0, screen_width, screen_height, 0],
        origin="upper",
        cmap="inferno",
        aspect="auto",
        alpha=0.9,
    )

    if targets:
        tx_list, ty_list = zip(*targets)
        ax.scatter(tx_list, ty_list, c="cyan", s=80, marker="+",
                   linewidths=2, zorder=5, label="Targets")
        ax.legend(facecolor="#2a2a4e", labelcolor="white", loc="upper right")

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Gaze Density", color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    ax.set_title(title, color="white", fontsize=14, pad=12)
    ax.set_xlabel("Screen X (px)", color="white")
    ax.set_ylabel("Screen Y (px)", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444466")

    ax.set_xlim(0, screen_width)
    ax.set_ylim(screen_height, 0)

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"[Heatmap] Saved → {output_path}")

    return fig
