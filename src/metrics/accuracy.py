
import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class TargetAccuracy:
    """Accuracy results for a single target."""
    target_id: int
    target_x: float
    target_y: float
    mean_gaze_x: float
    mean_gaze_y: float
    error_px: float
    error_deg: float
    n_samples: int


@dataclass
class AccuracyReport:
    """Aggregated accuracy across all targets."""
    per_target: List[TargetAccuracy]
    mean_error_px: float
    rmse_px: float
    mean_error_deg: float
    worst_target_error_px: float
    best_target_error_px: float


def euclidean_error(gaze_x: np.ndarray, gaze_y: np.ndarray,
                    target_x: float, target_y: float) -> np.ndarray:
    """Pixel-wise Euclidean distance from gaze samples to a target."""
    return np.sqrt((gaze_x - target_x) ** 2 + (gaze_y - target_y) ** 2)


def pixels_to_degrees(
    error_px: float,
    screen_width_px: int,
    screen_height_px: int,
    screen_width_cm: float,
    viewing_distance_cm: float,
) -> float:
    """Convert pixel error to visual degrees."""
    px_per_cm = screen_width_px / screen_width_cm
    error_cm = error_px / px_per_cm
    return math.degrees(math.atan2(error_cm, viewing_distance_cm))


def compute_accuracy(
    target_gaze_pairs: List[Tuple[Tuple[float, float], np.ndarray, np.ndarray]],
    screen_width_px: int = 1920,
    screen_height_px: int = 1080,
    screen_width_cm: float = 53.1,
    viewing_distance_cm: float = 65.0,
) -> AccuracyReport:
    """
    Compute accuracy for all targets.

    Args:
        target_gaze_pairs: List of ((tx, ty), gaze_x_array, gaze_y_array)
        screen_*: Screen parameters for angular conversion

    Returns:
        AccuracyReport with per-target and aggregated metrics
    """
    per_target = []
    all_errors = []

    for i, ((tx, ty), gaze_x, gaze_y) in enumerate(target_gaze_pairs):
        if len(gaze_x) == 0:
            continue

        errors = euclidean_error(gaze_x, gaze_y, tx, ty)
        mean_error = float(np.mean(errors))
        mean_x = float(np.mean(gaze_x))
        mean_y = float(np.mean(gaze_y))
        mean_deg = pixels_to_degrees(
            mean_error, screen_width_px, screen_height_px,
            screen_width_cm, viewing_distance_cm
        )

        per_target.append(TargetAccuracy(
            target_id=i,
            target_x=tx,
            target_y=ty,
            mean_gaze_x=mean_x,
            mean_gaze_y=mean_y,
            error_px=mean_error,
            error_deg=mean_deg,
            n_samples=len(gaze_x),
        ))
        all_errors.extend(errors.tolist())

    all_errors_arr = np.array(all_errors)

    return AccuracyReport(
        per_target=per_target,
        mean_error_px=float(np.mean(all_errors_arr)),
        rmse_px=float(np.sqrt(np.mean(all_errors_arr ** 2))),
        mean_error_deg=float(np.mean([t.error_deg for t in per_target])),
        worst_target_error_px=float(np.max([t.error_px for t in per_target])),
        best_target_error_px=float(np.min([t.error_px for t in per_target])),
    )
