from pathlib import Path
from typing import List, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def plot_timeseries(
    timestamps: List[float],
    gaze_x: List[float],
    gaze_y: List[float],
    valid_flags: List[bool],
    target_x: Optional[float] = None,
    target_y: Optional[float] = None,
    window_s: float = 2.0,
    output_path: Optional[str] = None,
    title: str = "Gaze Time Series",
    dpi: int = 150,
) -> plt.Figure:
    """
    Multi-panel time-series plot:
      1. Gaze X over time
      2. Gaze Y over time
      3. Sample-to-sample error (rolling)
      4. Valid / invalid sample indicator

    Args:
        timestamps: Unix timestamps
        gaze_x, gaze_y: Gaze position arrays
        valid_flags: Boolean validity per sample
        target_x, target_y: Optional static target for error computation
        window_s: Rolling window size for smoothing
        output_path: Save path
        title: Figure title
        dpi: Output DPI

    Returns:
        matplotlib Figure
    """
    ts = np.array(timestamps)
    gx = np.array(gaze_x)
    gy = np.array(gaze_y)
    valid = np.array(valid_flags, dtype=bool)

    t_rel = ts - ts[0]

    fig = plt.figure(figsize=(14, 8), dpi=dpi)
    gs = gridspec.GridSpec(4, 1, hspace=0.45)

    ax1 = fig.add_subplot(gs[0])
    ax1.plot(t_rel, gx, lw=0.8, color="#2196F3", alpha=0.7, label="Gaze X")
    if target_x is not None:
        ax1.axhline(target_x, color="#F44336", lw=1.2, linestyle="--", label="Target X")
    ax1.set_ylabel("X (px)", fontsize=9)
    ax1.set_title(title, fontsize=12, pad=8)
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, alpha=0.25)

   
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.plot(t_rel, gy, lw=0.8, color="#4CAF50", alpha=0.7, label="Gaze Y")
    if target_y is not None:
        ax2.axhline(target_y, color="#FF9800", lw=1.2, linestyle="--", label="Target Y")
    ax2.set_ylabel("Y (px)", fontsize=9)
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, alpha=0.25)

    # --- Panel 3: Rolling error ---
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    if target_x is not None and target_y is not None:
        error = np.sqrt((gx - target_x) ** 2 + (gy - target_y) ** 2)
        error[~valid] = np.nan
        ax3.plot(t_rel, error, lw=0.8, color="#9C27B0", alpha=0.7, label="Error (px)")
        ax3.set_ylabel("Error (px)", fontsize=9)
    else:
        s2s = np.sqrt(np.diff(gx, prepend=gx[0]) ** 2 + np.diff(gy, prepend=gy[0]) ** 2)
        ax3.plot(t_rel, s2s, lw=0.8, color="#9C27B0", alpha=0.7, label="S2S Distance (px)")
        ax3.set_ylabel("S2S (px)", fontsize=9)
    ax3.legend(loc="upper right", fontsize=8)
    ax3.grid(True, alpha=0.25)


    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    ax4.fill_between(t_rel, valid.astype(float), step="mid",
                     alpha=0.6, color="#4CAF50", label="Valid")
    ax4.fill_between(t_rel, (~valid).astype(float), step="mid",
                     alpha=0.6, color="#F44336", label="Invalid")
    ax4.set_ylim(0, 1.05)
    ax4.set_ylabel("Valid", fontsize=9)
    ax4.set_xlabel("Time (s)", fontsize=9)
    ax4.legend(loc="upper right", fontsize=8)

    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.setp(ax3.get_xticklabels(), visible=False)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        print(f"[TimeSeries] Saved → {output_path}")

    return fig
