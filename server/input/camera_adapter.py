from __future__ import annotations

import logging
from collections.abc import Callable

import cv2  # type: ignore[import-untyped]
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class LocalCameraAdapter:
    """Local webcam adapter implementing CameraSource protocol."""

    def __init__(self, camera_index: int = 0) -> None:
        self._capture: cv2.VideoCapture = cv2.VideoCapture(camera_index)
        self._subscribers: list[Callable[[NDArray[np.uint8]], None]] = []

    def read_frame(self) -> NDArray[np.uint8] | None:
        ok, frame = self._capture.read()
        if not ok:
            return None
        frame_array: NDArray[np.uint8] = np.asarray(frame, dtype=np.uint8)
        for callback in self._subscribers:
            try:
                callback(frame_array)
            except Exception:
                logger.exception("Camera subscriber raised an exception")
        return frame_array

    def release(self) -> None:
        self._capture.release()

    def is_opened(self) -> bool:
        return bool(self._capture.isOpened())

    @property
    def frame_width(self) -> int:
        return int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def frame_height(self) -> int:
        return int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def subscribe(self, callback: Callable[[NDArray[np.uint8]], None]) -> None:
        """Register a callback invoked on each successful read_frame()."""
        self._subscribers.append(callback)
