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
            f"    Jitter         : {self.data_quality.inter_sample_jitter_ms:.1f} ms",
            f"    OOB Rate       : {self.data_quality.out_of_bounds_rate_pct:.1f}%",
            f"{'─'*55}",
            f"  VERDICT        : {self.data_quality.quality_label}",
            f"  NOTE           : {self.data_quality.recommendation}",
            f"{'='*55}\n",
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialize to a nested dictionary (JSON-ready)."""
        return {
            "session_id": self.session_id,
            "device_type": self.device_type,
            "task_type": self.task_type,
            "timestamp_utc": self.timestamp_utc,
            "duration_s": self.duration_s,
            "screen": self.screen,
            "accuracy": asdict(self.accuracy),
            "precision": asdict(self.precision),
            "data_quality": asdict(self.data_quality),
        }

    def export_json(self, path: str) -> None:
        """Save report as JSON."""
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        print(f"[Export] JSON saved → {output}")

    def export_csv(self, path: str) -> None:
        """Save per-target accuracy as CSV."""
        import csv
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["target_id", "target_x", "target_y",
                               "mean_gaze_x", "mean_gaze_y",
                               "error_px", "error_deg", "n_samples"]
            )
            writer.writeheader()
            for t in self.accuracy.per_target:
                writer.writerow(asdict(t))
        print(f"[Export] CSV saved → {output}")


class BenchmarkAggregator:
    """
    Collects raw gaze samples and target hit events during a task run,
    then computes the complete BenchmarkReport on demand.
    """

    def __init__(
        self,
        device_type: str = "unknown",
        task_type: str = "unknown",
        screen_width_px: int = 1920,
        screen_height_px: int = 1080,
        screen_width_cm: float = 53.1,
        viewing_distance_cm: float = 65.0,
        sampling_rate_hz: float = 30,
    ):
        self.device_type = device_type
        self.task_type = task_type
        self.screen = {
            "width_px": screen_width_px,
            "height_px": screen_height_px,
            "width_cm": screen_width_cm,
            "viewing_distance_cm": viewing_distance_cm,
        }
        self.sampling_rate_hz = sampling_rate_hz
        self.session_id = datetime.now().strftime("session_%Y%m%d_%H%M%S")

        self._all_x: List[float] = []
        self._all_y: List[float] = []
        self._all_valid: List[bool] = []
        self._all_ts: List[float] = []
        self._target_data: Dict[int, Tuple] = {}
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None

    def start(self):
        self._start_time = time.time()

    def record_sample(self, ts: float, x: float, y: float, valid: bool):
        self._all_x.append(x)
        self._all_y.append(y)
        self._all_valid.append(valid)
        self._all_ts.append(ts)

    def record_target_sample(
        self, target_id: int, target_x: float, target_y: float,
        gaze_x: float, gaze_y: float
    ):
        if target_id not in self._target_data:
            self._target_data[target_id] = (target_x, target_y, [], [])
        self._target_data[target_id][2].append(gaze_x)
        self._target_data[target_id][3].append(gaze_y)

    def stop(self):
        self._end_time = time.time()

    def compute(self) -> BenchmarkReport:
        """Compute and return the full BenchmarkReport."""
        self._end_time = self._end_time or time.time()
        self._start_time = self._start_time or self._end_time

        gaze_x = np.array(self._all_x)
        gaze_y = np.array(self._all_y)
        timestamps = np.array(self._all_ts)
        valid_flags = self._all_valid

        valid_mask = np.array(valid_flags)
        valid_x = gaze_x[valid_mask]
        valid_y = gaze_y[valid_mask]

        target_pairs = []
        for tid, (tx, ty, gx_list, gy_list) in sorted(self._target_data.items()):
            target_pairs.append(((tx, ty), np.array(gx_list), np.array(gy_list)))

        acc = compute_accuracy(
            target_pairs,
            screen_width_px=self.screen["width_px"],
            screen_height_px=self.screen["height_px"],
            screen_width_cm=self.screen["width_cm"],
            viewing_distance_cm=self.screen["viewing_distance_cm"],
        )
        prec = compute_precision(valid_x, valid_y)
        dq = compute_data_quality(
            valid_flags=valid_flags,
            timestamps=timestamps,
            gaze_x=gaze_x,
            gaze_y=gaze_y,
            screen_width=self.screen["width_px"],
            screen_height=self.screen["height_px"],
            sampling_rate_hz=self.sampling_rate_hz,
            rms_px=prec.rms_s2s_px,
        )

        return BenchmarkReport(
            session_id=self.session_id,
            device_type=self.device_type,
            task_type=self.task_type,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            duration_s=round(self._end_time - self._start_time, 2),
            screen=self.screen,
            accuracy=acc,
            precision=prec,
            data_quality=dq,
        )
