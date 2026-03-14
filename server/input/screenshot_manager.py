from __future__ import annotations

import numpy as np
from config.third_party import load_cv2
from numpy.typing import NDArray

from models.protocols import CameraSource
cv2 = load_cv2()


class ScreenshotManager:
    """Centralised frame capture shared by pipeline and state tracker."""

    def __init__(self, camera: CameraSource) -> None:
        self._camera = camera
        self._bgr_frame: NDArray[np.uint8] | None = None
        self._rgb_frame: NDArray[np.uint8] | None = None

    def tick(self) -> None:
        """Capture a new frame. Call once per pipeline tick."""
        self._bgr_frame = self._camera.read_frame()
        self._rgb_frame = None  # invalidate cached RGB

    @property
    def bgr_frame(self) -> NDArray[np.uint8] | None:
        return self._bgr_frame

    @property
    def rgb_frame(self) -> NDArray[np.uint8] | None:
        if self._rgb_frame is None and self._bgr_frame is not None:
            self._rgb_frame = np.asarray(
                cv2.cvtColor(self._bgr_frame, cv2.COLOR_BGR2RGB),
                dtype=np.uint8,
            )
        return self._rgb_frame

    def encode_jpeg(self, quality: int = 80) -> bytes:
        """JPEG-encode the current BGR frame."""
        if self._bgr_frame is None:
            return b""
        ok, buf = cv2.imencode(
            ".jpg",
            self._bgr_frame,
            [cv2.IMWRITE_JPEG_QUALITY, quality],
        )
        if not ok:
            return b""
        return bytes(buf)

    def is_opened(self) -> bool:
        return self._camera.is_opened()

    def release(self) -> None:
        self._camera.release()
