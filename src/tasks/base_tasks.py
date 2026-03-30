"""
Abstract Base Task

All benchmark tasks must subclass BaseTask and implement:
  - generate_targets() -> List of (x, y) target positions
  - run(device) -> dict of collected gaze data per target
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class TaskResult:
    """Result container from a completed benchmark task."""
    task_type: str
    targets: List[Tuple[float, float]]          
    gaze_per_target: List[Tuple[List, List]]     
    all_timestamps: List[float]
    all_gaze_x: List[float]
    all_gaze_y: List[float]
    all_valid: List[bool]


class BaseTask(ABC):
    """Abstract base for all eye-tracking benchmark tasks."""

    def __init__(self, screen_width: int = 1920, screen_height: int = 1080):
        self.screen_width = screen_width
        self.screen_height = screen_height

    @abstractmethod
    def generate_targets(self) -> List[Tuple[float, float]]:
        """Return list of (x, y) target positions in screen pixels."""
        ...

    @abstractmethod
    def run(self, device) -> TaskResult:
        """
        Execute the task, collect gaze data, and return a TaskResult.

        Args:
            device: A connected BaseDevice instance

        Returns:
            TaskResult with all collected data
        """
        ...
