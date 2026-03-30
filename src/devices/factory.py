"""
Device Factory — creates the correct device adapter from a config string.
"""

from .base_device import BaseDevice


class DeviceFactory:
    """Factory for creating eye-tracking device adapters."""

    @staticmethod
    def create(device_type: str, **kwargs) -> BaseDevice:
        """
        Instantiate a device by type name.

        Args:
            device_type: One of 'webcam', 'csv', 'tobii', 'mouse'
            **kwargs: Forwarded to the device constructor

        Returns:
            A BaseDevice subclass instance (not yet connected)
        """
        device_type = device_type.lower().strip()

        if device_type == "webcam":
            from .webcam_device import WebcamDevice
            return WebcamDevice(**kwargs)

        elif device_type == "csv":
            from .csv_replay import CSVReplayDevice
            csv_path = kwargs.pop("csv_path", kwargs.pop("input", None))
            if not csv_path:
                raise ValueError("CSV device requires 'csv_path' or 'input' argument")
            return CSVReplayDevice(csv_path=csv_path, **kwargs)

        elif device_type == "tobii":
            try:
                from .tobii_device import TobiiDevice
                return TobiiDevice(**kwargs)
            except ImportError:
                raise ImportError(
                    "Tobii device requires the Tobii SDK. "
                    "See docs/device_setup.md for installation instructions."
                )

        elif device_type == "mouse":
            from .mouse_device import MouseDevice
            return MouseDevice(**kwargs)

        else:
            raise ValueError(
                f"Unknown device type: '{device_type}'. "
                f"Supported: webcam, csv, tobii, mouse"
            )
