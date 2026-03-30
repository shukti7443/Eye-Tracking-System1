
import math
import time
from typing import List, Tuple

from .base_task import BaseTask, TaskResult


class PursuitTask(BaseTask):
    """
    Smooth pursuit task with configurable motion patterns.

    Supported patterns: 'circle', 'sine', 'linear'
    """

    PATTERNS = ("circle", "sine", "linear")

    def __init__(
        self,
        duration: float = 20.0,
        pattern: str = "circle",
        speed_px: float = 100.0,
        screen_width: int = 1920,
        screen_height: int = 1080,
    ):
        super().__init__(screen_width, screen_height)
        if pattern not in self.PATTERNS:
            raise ValueError(f"pattern must be one of {self.PATTERNS}")
        self.duration = duration
        self.pattern = pattern
        self.speed_px = speed_px

    def _target_position(self, t: float) -> Tuple[float, float]:
        """Compute target (x, y) at time t (seconds from task start)."""
        cx, cy = self.screen_width / 2, self.screen_height / 2
        radius = min(self.screen_width, self.screen_height) * 0.35
        omega = self.speed_px / radius

        if self.pattern == "circle":
            x = cx + radius * math.cos(omega * t)
            y = cy + radius * math.sin(omega * t)

        elif self.pattern == "sine":
            amp = self.screen_height * 0.3
            freq = self.speed_px / (self.screen_width * 0.8)
            x = cx + (self.screen_width * 0.4) * math.sin(freq * t)
            y = cy + amp * math.sin(2 * freq * t)

        elif self.pattern == "linear":
            half_w = self.screen_width * 0.4
            period = 2 * half_w / self.speed_px
            phase = (t % period) / period
            x = cx - half_w + 2 * half_w * (phase if phase < 0.5 else 1 - phase) * 2
            y = cy

        else:
            x, y = cx, cy

        return float(x), float(y)

    def generate_targets(self) -> List[Tuple[float, float]]:
        """Sample target positions at 10 Hz for reference."""
        return [self._target_position(t * 0.1) for t in range(int(self.duration * 10))]

    def run(self, device) -> TaskResult:
        all_timestamps, all_gaze_x, all_gaze_y, all_valid = [], [], [], []
        target_positions: List[Tuple[float, float]] = []
        gaze_x_per_pos, gaze_y_per_pos = [], []

        print(f"[PursuitTask] Running {self.pattern} pursuit for {self.duration}s...")
        t_start = time.time()

        while time.time() - t_start < self.duration:
            elapsed = time.time() - t_start
            tx, ty = self._target_position(elapsed)
            sample = device.get_gaze_sample()

            all_timestamps.append(sample.timestamp)
            all_gaze_x.append(sample.x)
            all_gaze_y.append(sample.y)
            all_valid.append(sample.valid)

            if sample.valid:
                target_positions.append((tx, ty))
                gaze_x_per_pos.append(sample.x)
                gaze_y_per_pos.append(sample.y)

            time.sleep(max(0, 1.0 / device.sampling_rate_hz - 0.002))

        print(f"[PursuitTask] Complete. {len(all_timestamps)} samples collected.")

        gaze_per_target = [([gx], [gy]) for gx, gy in zip(gaze_x_per_pos, gaze_y_per_pos)]

        return TaskResult(
            task_type="pursuit",
            targets=target_positions,
            gaze_per_target=gaze_per_target,
            all_timestamps=all_timestamps,
            all_gaze_x=all_gaze_x,
            all_gaze_y=all_gaze_y,
            all_valid=all_valid,
        )
