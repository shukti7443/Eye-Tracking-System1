"""
Grid Accuracy Task

Presents a configurable N×N grid of static targets one at a time.
The participant fixates each dot while gaze is recorded for `dwell_time` seconds.
This is the primary task for measuring spatial accuracy across the screen.
"""

import math
import time
from typing import List, Tuple

from .base_task import BaseTask, TaskResult


class GridAccuracyTask(BaseTask):
    """
    Static grid accuracy task.

    Targets are evenly spaced across the screen in a √n × √n grid.
    Each target is shown for `dwell_time` seconds; gaze samples collected
    after a brief onset delay to discard saccade transients.
    """

    def __init__(
        self,
        n_targets: int = 9,
        dwell_time: float = 1.5,
        onset_delay: float = 0.3,
        inter_target_gap: float = 0.5,
        screen_width: int = 1920,
        screen_height: int = 1080,
        margin_pct: float = 0.1,
    ):
        super().__init__(screen_width, screen_height)

        grid_side = int(math.sqrt(n_targets))
        if grid_side * grid_side != n_targets:
            raise ValueError(f"n_targets must be a perfect square (e.g. 9, 16, 25). Got {n_targets}.")

        self.n_targets = n_targets
        self.grid_side = grid_side
        self.dwell_time = dwell_time
        self.onset_delay = onset_delay
        self.inter_target_gap = inter_target_gap
        self.margin_pct = margin_pct

    def generate_targets(self) -> List[Tuple[float, float]]:
        """Generate evenly-spaced grid target positions."""
        margin_x = self.screen_width * self.margin_pct
        margin_y = self.screen_height * self.margin_pct
        usable_w = self.screen_width - 2 * margin_x
        usable_h = self.screen_height - 2 * margin_y

        targets = []
        for row in range(self.grid_side):
            for col in range(self.grid_side):
                if self.grid_side == 1:
                    x = self.screen_width / 2
                    y = self.screen_height / 2
                else:
                    x = margin_x + col * usable_w / (self.grid_side - 1)
                    y = margin_y + row * usable_h / (self.grid_side - 1)
                targets.append((float(x), float(y)))
        return targets

    def run(self, device) -> TaskResult:
        """Execute the grid task using the provided device."""
        targets = self.generate_targets()

        all_timestamps: List[float] = []
        all_gaze_x: List[float] = []
        all_gaze_y: List[float] = []
        all_valid: List[bool] = []
        gaze_per_target: List[Tuple[List, List]] = []

        print(f"[GridAccuracyTask] Starting {self.n_targets}-point grid task...")

        for i, (tx, ty) in enumerate(targets):
            print(f"  Target {i+1}/{self.n_targets}  @ ({tx:.0f}, {ty:.0f})")

            # Onset delay — discard samples (saccade in flight)
            t_onset = time.time()
            while time.time() - t_onset < self.onset_delay:
                device.get_gaze_sample()

            # Dwell period — collect samples
            target_x_samples: List[float] = []
            target_y_samples: List[float] = []
            t_start = time.time()

            while time.time() - t_start < self.dwell_time:
                sample = device.get_gaze_sample()
                all_timestamps.append(sample.timestamp)
                all_gaze_x.append(sample.x)
                all_gaze_y.append(sample.y)
                all_valid.append(sample.valid)

                if sample.valid:
                    target_x_samples.append(sample.x)
                    target_y_samples.append(sample.y)

                time.sleep(max(0, 1.0 / device.sampling_rate_hz - 0.002))

            gaze_per_target.append((target_x_samples, target_y_samples))

            # Inter-target gap
            if i < len(targets) - 1:
                time.sleep(self.inter_target_gap)

        print(f"[GridAccuracyTask] Complete. {len(all_timestamps)} total samples collected.")

        return TaskResult(
            task_type="grid_accuracy",
            targets=targets,
            gaze_per_target=gaze_per_target,
            all_timestamps=all_timestamps,
            all_gaze_x=all_gaze_x,
            all_gaze_y=all_gaze_y,
            all_valid=all_valid,
        )
