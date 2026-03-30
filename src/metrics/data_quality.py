from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class DataQualityReport:
    """Data quality metrics for a recording session."""
    total_samples: int
    valid_samples: int
    invalid_samples: int
    data_loss_rate_pct: float
    estimated_blink_count: int
    mean_blink_duration_ms: float
    inter_sample_jitter_ms: float
    out_of_bounds_rate_pct: float
    quality_label: str
    recommendation: str


def detect_blinks(valid_mask: np.ndarray, sampling_rate_hz: float = 30) -> dict:
    """
    Detect blink events as consecutive invalid sample runs.

    Args:
        valid_mask: Boolean array (True = valid sample)
        sampling_rate_hz: Samples per second

    Returns:
        dict with blink_count and mean_blink_duration_ms
    """
    ms_per_sample = 1000.0 / sampling_rate_hz
    invalid = ~valid_mask

    blink_durations = []
    in_blink = False
    blink_len = 0

    for v in invalid:
        if v:
            in_blink = True
            blink_len += 1
        else:
            if in_blink and blink_len >= 2:
                blink_durations.append(blink_len * ms_per_sample)
            in_blink = False
            blink_len = 0

    if in_blink and blink_len >= 2:
        blink_durations.append(blink_len * ms_per_sample)

    return {
        "blink_count": len(blink_durations),
        "mean_blink_duration_ms": float(np.mean(blink_durations)) if blink_durations else 0.0,
    }


def compute_jitter(timestamps: np.ndarray) -> float:
    """
    Compute inter-sample timing jitter (std dev of intervals in ms).
    Low jitter = consistent sampling rate.
    """
    if len(timestamps) < 2:
        return 0.0
    intervals_ms = np.diff(timestamps) * 1000.0
    return float(np.std(intervals_ms))


def classify_quality(
    data_loss_pct: float,
    rms_px: float,
    jitter_ms: float,
) -> tuple:
    """
    Assign a quality label and recommendation based on metric thresholds.

    Returns:
        (label: str, recommendation: str)
    """
    if data_loss_pct > 20:
        return "POOR", "High data loss detected — check camera visibility and lighting."
    if data_loss_pct > 10:
        return "MODERATE", "Elevated data loss — consider improving lighting conditions."
    if rms_px > 80:
        return "POOR", "Tracking very imprecise — recalibrate or reposition the tracker."
    if rms_px > 40:
        return "MODERATE", "Precision is acceptable but could be improved — try recalibrating."
    if jitter_ms > 10:
        return "MODERATE", "Timing jitter detected — ensure no background processes are competing."
    return "GOOD", "Tracking quality is stable."


def compute_data_quality(
    valid_flags: List[bool],
    timestamps: np.ndarray,
    gaze_x: np.ndarray,
    gaze_y: np.ndarray,
    screen_width: int = 1920,
    screen_height: int = 1080,
    sampling_rate_hz: float = 30,
    rms_px: float = 0.0,
) -> DataQualityReport:
    """
    Compute full data quality report.
    """
    valid_mask = np.asarray(valid_flags, dtype=bool)
    total = len(valid_mask)
    valid = int(np.sum(valid_mask))
    invalid = total - valid
    loss_pct = (invalid / total * 100.0) if total > 0 else 0.0

    blink_info = detect_blinks(valid_mask, sampling_rate_hz)
    jitter = compute_jitter(timestamps)

    valid_x = gaze_x[valid_mask]
    valid_y = gaze_y[valid_mask]
    oob = np.sum(
        (valid_x < 0) | (valid_x > screen_width) |
        (valid_y < 0) | (valid_y > screen_height)
    )
    oob_pct = (float(oob) / valid * 100.0) if valid > 0 else 0.0

    label, recommendation = classify_quality(loss_pct, rms_px, jitter)

    return DataQualityReport(
        total_samples=total,
        valid_samples=valid,
        invalid_samples=invalid,
        data_loss_rate_pct=round(loss_pct, 2),
        estimated_blink_count=blink_info["blink_count"],
        mean_blink_duration_ms=round(blink_info["mean_blink_duration_ms"], 1),
        inter_sample_jitter_ms=round(jitter, 2),
        out_of_bounds_rate_pct=round(oob_pct, 2),
        quality_label=label,
        recommendation=recommendation,
    )
