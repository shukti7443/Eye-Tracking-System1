"""
Tobii Device Adapter (stub)

Wraps the Tobii Pro Python SDK to provide gaze samples via the standard
BaseDevice interface.

Installation:
    pip install tobii-research
    See docs/device_setup.md for full setup instructions.

Usage:
    device = TobiiDevice(sampling_rate_hz=60)
    with device:
        sample = device.get_gaze_sample()
"""

import time
import threading
from typing import Optional

from .base_device import BaseDevice, GazeSample


class TobiiDevice(BaseDevice):
    """
    Tobii Pro eye tracker adapter using the tobii_research SDK.

    Subscribes to the Tobii gaze data stream and buffers the latest sample
    for retrieval via get_gaze_sample().
    """

    def __init__(
        self,
        tracker_address: Optional[str] = None,
        sampling_rate_hz: float = 60.0,
        **kwargs,
    ):
        super().__init__(sampling_rate_hz)
        self.tracker_address = tracker_address
        self._tracker = None
        self._latest_sample: Optional[GazeSample] = None
        self._lock = threading.Lock()

    def connect(self) -> None:
        try:
            import tobii_research as tr
        except ImportError:
            raise ImportError(
                "TobiiDevice requires tobii-research SDK. "
                "Install with: pip install tobii-research\n"
                "See docs/device_setup.md for further instructions."
            )

        if self.tracker_address:
            self._tracker = tr.EyeTracker(self.tracker_address)
        else:
            trackers = tr.find_all_eyetrackers()
            if not trackers:
                raise RuntimeError("No Tobii eye trackers found. Check USB connection.")
            self._tracker = trackers[0]
            print(f"[TobiiDevice] Found tracker: {self._tracker.model} @ {self._tracker.address}")

        self._tracker.subscribe_to(
            tr.EYETRACKER_GAZE_DATA,
            self._gaze_callback,
            as_dictionary=True,
        )
        self._connected = True
        print(f"[TobiiDevice] Connected and streaming.")

    def _gaze_callback(self, gaze_data: dict) -> None:
        """Tobii SDK callback — fires on each new gaze sample."""
        left = gaze_data.get("left_gaze_point_on_display_area", (0.5, 0.5))
        right = gaze_data.get("right_gaze_point_on_display_area", (0.5, 0.5))
        left_valid = gaze_data.get("left_gaze_point_validity", 0)
        right_valid = gaze_data.get("right_gaze_point_validity", 0)

        if left_valid and right_valid:
            nx = (left[0] + right[0]) / 2
            ny = (left[1] + right[1]) / 2
            valid = True
        elif left_valid:
            nx, ny = left
            valid = True
        elif right_valid:
            nx, ny = right
            valid = True
        else:
            nx, ny = 0.5, 0.5
            valid = False

        ts = gaze_data.get("system_time_stamp", time.time() * 1e6) / 1e6
        left_pupil = gaze_data.get("left_pupil_diameter")
        right_pupil = gaze_data.get("right_pupil_diameter")

        sample = GazeSample(
            timestamp=ts,
            x=nx,
            y=ny,
            valid=valid,
            left_pupil_size=left_pupil,
            right_pupil_size=right_pupil,
            confidence=1.0 if valid else 0.0,
        )
        with self._lock:
            self._latest_sample = sample

    def get_gaze_sample(self) -> GazeSample:
        with self._lock:
            if self._latest_sample is not None:
                return self._latest_sample
        return GazeSample(timestamp=time.time(), x=0.5, y=0.5, valid=False, confidence=0.0)

    def disconnect(self) -> None:
        if self._tracker is
