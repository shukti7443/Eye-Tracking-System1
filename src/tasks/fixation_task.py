

import time
from typing import List, Tuple

from .base_task import BaseTask, TaskResult


class FixationTask(BaseTask):
    """Single central fixation for measuring tracker noise floor and stability."""

    def __init__(
        self,
        duration: float = 30.0,
        screen_width: int = 1920,
        screen_height: int = 1080,
        target_x: float = None,
        target_y: float = None,
    ):
        super().__init__(screen_width, screen_height)
        self.duration = duration
        self.target_x = target_x if target_x is not None else screen_width / 2
        self.target_y = target_y if target_y is not None else screen_height / 2

    def generate_targets(self) -> List[Tuple[float, float]]:
        return [(self.target_x, self.target_y)]

    def run(self, device) -> TaskResult:
        target = (self.target_x, self.target_y)
        all_timestamps, all_gaze_x, all_gaze_y, all_valid = [], [], [], []
        target_x_samples, target_y_samples = [], []

        print(f"[FixationTask] Recording {self.duration}s fixation at center...")
        t_start = time.time()

        while time.time() - t_start < self.duration:
            sample = device.get_gaze_sample()
            all_timestamps.append(sample.timestamp)
            all_gaze_x.append(sample.x)
            all_gaze_y.append(sample.y)
            all_valid.append(sample.valid)

            if sample.valid:
                target_x_samples.append(sample.x)
                target_y_samples.append(sample.y)

            time.sleep(max(0, 1.0 / device.sampling_rate_hz - 0.002))

        print(f"[FixationTask] Complete. {len(all_timestamps)} samples collected.")

        return TaskResult(
            task_type="fixation",
            targets=[target],
            gaze_per_target=[(target_x_samples, target_y_samples)],
            all_timestamps=all_timestamps,
            all_gaze_x=all_gaze_x,
            all_gaze_y=all_gaze_y,
            all_valid=all_valid,
        )
