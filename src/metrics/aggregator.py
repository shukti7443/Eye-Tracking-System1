import json
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .accuracy import AccuracyReport, compute_accuracy
from .precision import PrecisionReport, compute_precision
from .data_quality import DataQualityReport, compute_data_quality


@dataclass
class BenchmarkReport:
    """Full benchmark results for one session."""
    session_id: str
    device_type: str
    task_type: str
    timestamp_utc: str
    duration_s: float
    screen: Dict
    accuracy: AccuracyReport
    precision: PrecisionReport
    data_quality: DataQualityReport

    def summary(self) -> str:
        """Return a human-readable summary string."""
        lines = [
            f"\n{'='*55}",
            f"  EYE-TRACKING BENCHMARK REPORT",
            f"{'='*55}",
            f"  Session   : {self.session_id}",
            f"  Device    : {self.device_type}",
            f"  Task      : {self.task_type}",
            f"  Duration  : {self.duration_s:.1f}s",
            f"  Timestamp : {self.timestamp_utc}",
            f"{'─'*55}",
            f"  ACCURACY",
            f"    Mean Error     : {self.accuracy.mean_error_px:.1f} px  "
            f"({self.accuracy.mean_error_deg:.2f}°)",
            f"    RMSE           : {self.accuracy.rmse_px:.1f} px",
            f"    Best target    : {self.accuracy.best_target_error_px:.1f} px",
            f"    Worst target   : {self.accuracy.worst_target_error_px:.1f} px",
            f"{'─'*55}",
            f"  PRECISION",
            f"    RMS S2S        : {self.precision.rms_s2s_px:.1f} px",
            f"    Std X / Y      : {self.precision.std_x_px:.1f} / {self.precision.std_y_px:.1f} px",
            f"    BCEA ({int(self.precision.bcea_prob*100)}%)   : {self.precision.bcea_px2:.0f} px²",
            f"{'─'*55}",
            f"  DATA QUALITY",
            f"    Data Loss      : {self.data_quality.data_loss_rate_pct:.1f}%",
            f"    Blinks         : {self.data_quality.estimated_blink_count}",
            f"    Jitter
