
import math
from dataclasses import dataclass

import numpy as np


@dataclass
class PrecisionReport:
    """Precision metrics for a gaze recording segment."""
    rms_s2s_px: float
    std_x_px: float
    std_y_px: float
    bcea_px2: float
    bcea_prob: float
    n_samples: int


def rms_sample_to_sample(gaze_x: np.ndarray, gaze_y: np.ndarray) -> float:
    """
    Root mean square of sample-to-sample distances.
    Lower is better — measures temporal stability between consecutive samples.
    """
    if len(gaze_x) < 2:
        return 0.0
    dx = np.diff(gaze_x)
    dy = np.diff(gaze_y)
    distances = np.sqrt(dx ** 2 + dy ** 2)
    return float(np.sqrt(np.mean(distances ** 2)))


def bivariate_contour_ellipse_area(
    gaze_x: np.ndarray, gaze_y: np.ndarray, probability: float = 0.68
) -> float:
    """
    Bivariate Contour Ellipse Area (BCEA).

    Measures the area (in px²) of the ellipse that contains `probability`
    proportion of the gaze distribution.

    Args:
        gaze_x, gaze_y: Arrays of gaze positions
        probability: Coverage probability (0.68 ≈ 1 SD, 0.95 ≈ 2 SD)

    Returns:
        BCEA in pixels²
    """
    if len(gaze_x) < 3:
        return 0.0

    std_x = float(np.std(gaze_x, ddof=1))
    std_y = float(np.std(gaze_y, ddof=1))

    rho = float(np.corrcoef(gaze_x, gaze_y)[0, 1])
    if np.isnan(rho):
        rho = 0.0

    k = -2.0 * math.log(1.0 - probability)
    bcea = 2.0 * math.pi * k * std_x * std_y * math.sqrt(1.0 - rho ** 2)
    return bcea


def compute_precision(
    gaze_x: np.ndarray,
    gaze_y: np.ndarray,
    bcea_probability: float = 0.68,
) -> PrecisionReport:
    """
    Compute all precision metrics for a gaze array.

    Args:
        gaze_x, gaze_y: Numpy arrays of valid gaze positions
        bcea_probability: Coverage probability for BCEA

    Returns:
        PrecisionReport
    """
    gaze_x = np.asarray(gaze_x, dtype=float)
    gaze_y = np.asarray(gaze_y, dtype=float)

    return PrecisionReport(
        rms_s2s_px=rms_sample_to_sample(gaze_x, gaze_y),
        std_x_px=float(np.std(gaze_x, ddof=1)) if len(gaze_x) > 1 else 0.0,
        std_y_px=float(np.std(gaze_y, ddof=1)) if len(gaze_y) > 1 else 0.0,
        bcea_px2=bivariate_contour_ellipse_area(gaze_x, gaze_y, bcea_probability),
        bcea_prob=bcea_probability,
        n_samples=len(gaze_x),
    )
