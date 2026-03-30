"""
Webcam Device — gaze estimation via MediaPipe Face Mesh.

Uses iris landmark positions to estimate on-screen gaze position.
Accuracy is lower than dedicated hardware, but requires no special equipment.

Setup: pip install mediapipe opencv-python
"""

import time
from typing import Optional

import numpy as np

from .base_device import BaseDevice, GazeSample


class WebcamDevice(BaseDevice):
    """
    Eye-tracking device using webcam + MediaPipe FaceMesh iris landmarks.

    Gaze is estimated from the relative position of iris landmarks within
    the eye socket bounding box, mapped linearly to screen coordinates.

    Note: This is an approximation. Accuracy improves with:
      - Good, consistent lighting
      - Head relatively still
      - Camera at eye level
      - Run a calibration step before benchmarking
    """

    def __init__(
        self,
        screen_width: int = 1920,
        screen_height: int = 1080,
        camera_index: int = 0,
        sampling_rate_hz: float = 30,
    ):
        super().__init__(sampling_rate_hz)
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.camera_index = camera_index
        self._cap = None
        self._face_mesh = None

    def connect(self) -> None:
        try:
            import cv2
            import mediapipe as mp

            self._cv2 = cv2
            self._mp_face_mesh = mp.solutions.face_mesh

            self._cap = cv2.VideoCapture(self.camera_index)
            if not self._cap.isOpened():
                raise RuntimeError(f"Cannot open camera index {self.camera_index}")

            self._face_mesh = self._mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,  
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )

            self._connected = True
            print(f"[WebcamDevice] Camera {self.camera_index} opened successfully.")

        except ImportError as e:
            raise ImportError(
                "WebcamDevice requires mediapipe and opencv-python. "
                "Install with: pip install mediapipe opencv-python"
            ) from e

    def get_gaze_sample(self) -> GazeSample:
        if not self._connected or self._cap is None:
            raise RuntimeError("Device not connected. Call connect() first.")

        ret, frame = self._cap.read()
        if not ret:
            return GazeSample(
                timestamp=time.time(), x=0.0, y=0.0, valid=False, confidence=0.0
            )

        rgb = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)
        results = self._face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return GazeSample(
                timestamp=time.time(), x=0.0, y=0.0, valid=False, confidence=0.3
            )

        landmarks = results.multi_face_landmarks[0].landmark
        h, w = frame.shape[:2]
n
        right_iris = landmarks[473]
        left_iris = landmarks[468]

        iris_x = (right_iris.x + left_iris.x) / 2
        iris_y = (right_iris.y + left_iris.y) / 2

        screen_x = (1.0 - iris_x) * self.screen_width
        screen_y = iris_y * self.screen_height

        return GazeSample(
            timestamp=time.time(),
            x=float(screen_x),
            y=float(screen_y),
            valid=True,
            confidence=0.8,
        )

    def disconnect(self) -> None:
        if self._cap is not None:
            self._cap.release()
        if self._face_mesh is not None:
            self._face_mesh.close()
        self._connected = False
        print("[WebcamDevice] Disconnected.")
