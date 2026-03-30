"""
CSV Replay Device — replays gaze data from a CSV file.

Expected CSV columns:
    timestamp, x, y, valid[, confidence]

This is useful for:
  - Offline re-analysis of previously recorded sessions
  - Running the full benchmark pipeline without a live device
  - Unit testing (deterministic data)
"""

import csv
import time
from pathlib import Path
from typing import Iterator, List

from .base_device import BaseDevice, GazeSample


class CSVReplayDevice(BaseDevice):
    """
    Simulates a live eye-tracking device by replaying samples from a CSV file.
    Preserves original timing gaps between samples.
    """

    REQUIRED_COLUMNS = {"timestamp", "x", "y", "valid"}

    def __init__(self, csv_path: str, sampling_rate_hz: float = 30, realtime: bool = True):
        super().__init__(sampling_rate_hz)
        self.csv_path = Path(csv_path)
        self.realtime = realtime  # If False, replay as fast as possible
        self._samples: List[GazeSample] = []
        self._iterator: Iterator[GazeSample] = iter([])

    def connect(self) -> None:
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")

        self._samples = self._load_csv()
        self._iterator = iter(self._samples)
        self._connected = True
        print(f"[CSVReplayDevice] Loaded {len(self._samples)} samples from {self.csv_path.name}")

    def _load_csv(self) -> List[GazeSample]:
        samples = []
        with open(self.csv_path, newline="") as f:
            reader = csv.DictReader(f)
            headers = set(reader.fieldnames or [])
            missing = self.REQUIRED_COLUMNS - headers
            if missing:
                raise ValueError(f"CSV missing required columns: {missing}")

            for row in reader:
                samples.append(GazeSample(
                    timestamp=float(row["timestamp"]),
                    x=float(row["x"]),
                    y=float(row["y"]),
                    valid=row["valid"].strip().lower() in ("1", "true", "yes"),
                    confidence=float(row.get("confidence", 1.0)),
                ))

        return samples

    def get_gaze_sample(self) -> GazeSample:
        try:
            sample = next(self._iterator)
            return sample
        except StopIteration:
            # Loop back to the beginning for continuous replay
            self._iterator = iter(self._samples)
            return next(self._iterator)

    def disconnect(self) -> None:
        self._connected = False
        self._samples = []

    def __len__(self):
        return len(self._samples)
