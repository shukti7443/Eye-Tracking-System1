

import time
from typing import List, Tuple

from .base_task import BaseTask, TaskResult


class SaccadeTask(BaseTask):
   
    def __init__(
        self,
        amplitude_px: float = 500.0,
        frequency_hz: float = 0.5,
        duration: float = 30.0,
        onset_delay: float = 0.15,
        screen_width: int = 1920,
        screen_height: int = 1080,
    ):
        super().__init__(screen_width, screen_height)
        self.amplitude_px = amplitude_px
        self.frequency_hz = frequency_hz
        self.duration = duration
        self.onset_delay = onset_delay

        cx, cy = screen_width / 2, screen_height / 2
        half = amplitude_px / 2
        self._pos_a = (max(0.0, cx - half), float(cy))
        self._pos_b = (min(float(screen_width), cx + half), float(cy))

    def generate_targets(self) -> List[Tuple[float, float]]:
        return [self._pos_a, self._pos_b]

    def run(self, device) -> TaskResult:
        period = 1.0 / self.frequency_hz
        dwell = period / 2.0 - self.onset_delay

        all_timestamps, all_gaze_x, all_gaze_y, all_valid = [], [], [], []
        target_list: List[Tuple[float, float]] = []
        positions = [self._pos_a, self._pos_b]
        current_idx = 0

        print(f"[SaccadeTask] Running {self.duration}s saccade task "
              f"(amplitude={self.amplitude_px:.0f}px, {self.frequency_hz}Hz)...")

        t_start = time.time()
        while time.time() - t_start < self.duration:
            tx, ty = positions[current_idx]

            # Onset delay — wait for saccade to land
            t_onset = time.time()
            while time.time() - t_onset < self.onset_delay:
                device.get_gaze_sample()

            # Dwell — collect post-saccade fixation samples
            t_dwell = time.time()
            while time.time() - t_dwell < max(dwell, 0.1):
                sample = device.get_gaze_sample()
                all_timestamps.append(sample.timestamp)
                all_gaze_x.append(sample.x)
                all_gaze_y.append(sample.y)
                all_valid.append(sample.valid)
                target_list.append((tx, ty))
                time.sleep(max(0, 1.0 / device.sampling_rate_hz - 0.002))

            current_idx = 1 - current_idx

        # Split gaze samples by target
        a_x, a_y, b_x, b_y = [], [], [], []
        for (tx, ty), gx, gy, v in zip(target_list, all_gaze_x, all_gaze_y, all_valid):
            if not v:
                continue
            if (tx, ty) == self._pos_a:
                a_x.append(gx)
                a_y.append(gy)
            else:
                b_x.append(gx)
                b_y.append(gy)

        print(f"[SaccadeTask] Complete. {len(all_timestamps)} total samples.")

        return TaskResult(
            task_type="saccade",
            targets=[self._pos_a, self._pos_b],
            gaze_per_target=[(a_x, a_y), (b_x, b_y)],
            all_timestamps=all_timestamps,
            all_gaze_x=all_gaze_x,
            all_gaze_y=all_gaze_y,
            all_valid=all_valid,
        )
