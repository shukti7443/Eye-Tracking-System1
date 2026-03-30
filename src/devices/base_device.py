"""
Base class for all eye-tracking devices.

Any new device (like webcam, Tobii, etc.) should inherit this class
and implement the required methods.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class GazeSample:
    """Represents one gaze data point."""
    timestamp: float
    x: float
    y: float
    valid: bool
    left_pupil_size: Optional[float] = None
    right_pupil_size: Optional[float] = None
    confidence: float = 1.0


class BaseDevice(ABC):
    """Abstract base class for all eye-tracking devices."""

    def __init__(self, sampling_rate_hz: float = 30):
        self.sampling_rate_hz = sampling_rate_hz
        self._connected = False

    @abstractmethod
    def connect(self) -> None:
        """Connect to the device."""
        pass

    @abstractmethod
    def get_gaze_sample(self) -> GazeSample:
        """Get one gaze sample."""
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect the device."""
        pass

    @property
    def is_connected(self) -> bool:
        return self._connected

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
        return False
