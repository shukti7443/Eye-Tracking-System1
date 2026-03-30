"""
Mouse Device — uses the system mouse cursor as a gaze proxy.

Useful for:
  - UI testing and development without a camera
  - Demonstrating the benchmark pipeline interactively
  - Quick sanity checks of metric computations

Requires: no extra dependencies (uses ctypes on Windows, Xlib or AppKit on Linux/macOS)
Falls back to a pure-Python approach using tkinter if available.
"""

import time
import sys

from .base_device import BaseDevice, GazeSample


def _get_mouse_position():
    """Cross-platform mouse position reader. Returns (x, y) in screen pixels."""
    if sys.platform == "win32":
        import ctypes
        class POINT(ctypes.Structure):
            _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]
        pt = POINT()
        ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
        return float(pt.x), float(pt.y)

    elif sys.platform == "darwin":
        try:
            from AppKit import NSEvent
            loc = NSEvent.mouseLocation()
            import Quartz
            screen_h = Quartz.CGDisplayPixelsHigh(Quartz.CGMainDisplayID())
            return float(loc.x), float(screen_h - loc.y)
        except ImportError:
            pass

    # Fallback: use tkinter (works on most platforms)
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        x = root.winfo_pointerx()
        y = root.winfo_pointery()
        root.destroy()
        return float(x), float(y)
    except Exception:
        return 0.0, 0.0


class MouseDevice(BaseDevice):
    """
    Eye-tracking device that reads the system mouse cursor position.
    Ask the participant (or tester) to move the mouse to simulate gaze.
    """

    def __init__(self, sampling_rate_hz: float = 60, **kwargs):
        super().__init__(sampling_rate_hz)

    def connect(self) -> None:
        self._connected = True
        print("[MouseDevice] Connected — using mouse cursor as gaze proxy.")

    def get_gaze_sample(self) -> GazeSample:
        x, y = _get_mouse_position()
        return GazeSample(
            timestamp=time.time(),
            x=x,
            y=y,
            valid=True,
            confidence=1.0,
        )

    def disconnect(self) -> None:
        self._connected = False
        print("[MouseDevice] Disconnected.")
