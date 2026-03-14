from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import replace
from typing import Protocol

from .ring_buffer import LatestValueBuffer
from .types import CaptureStatus, ClientType, VideoFrame

try:
    import cv2  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover - exercised only when OpenCV is missing.
    cv2 = None


class _UnchangedType:
    pass


_UNCHANGED = _UnchangedType()


class CameraFrame(Protocol):
    width: int
    height: int

    def tobytes(self) -> bytes: ...


class CameraDevice(Protocol):
    def isOpened(self) -> bool: ...

    def read(self) -> tuple[bool, CameraFrame | None]: ...

    def release(self) -> None: ...


class _BufferedCameraFrame:
    def __init__(self, width: int, height: int, frame_bytes: bytes) -> None:
        self.width = width
        self.height = height
        self._frame_bytes = frame_bytes

    def tobytes(self) -> bytes:
        return self._frame_bytes


class _OpenCVCameraDevice:
    def __init__(self, camera_index: int) -> None:
        if cv2 is None:  # pragma: no cover - guarded by _build_capture.
            raise RuntimeError("OpenCV unavailable")
        self._capture = cv2.VideoCapture(camera_index)

    def isOpened(self) -> bool:
        return bool(self._capture.isOpened())

    def read(self) -> tuple[bool, CameraFrame | None]:
        ok, frame = self._capture.read()
        if not ok:
            return False, None
        return True, _BufferedCameraFrame(
            width=int(frame.shape[1]),
            height=int(frame.shape[0]),
            frame_bytes=frame.tobytes(),
        )

    def release(self) -> None:
        self._capture.release()


class CameraCapture:
    """Buffered webcam capture for the desktop client."""

    def __init__(
        self,
        client_id: str,
        camera_index: int = 0,
        buffer_size: int = 2,
        pixel_format: str = "bgr24",
        capture_factory: Callable[[int], CameraDevice | None] | None = None,
        time_fn: Callable[[], float] | None = None,
        retry_interval_seconds: float = 0.1,
    ) -> None:
        self._client_id = client_id
        self._camera_index = camera_index
        self._pixel_format = pixel_format
        self._capture_factory = capture_factory
        self._time_fn = time_fn or time.time
        self._retry_interval_seconds = retry_interval_seconds
        self._buffer = LatestValueBuffer[VideoFrame](buffer_size)
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._capture: CameraDevice | None = None
        self._status_lock = threading.Lock()
        self._status = CaptureStatus(
            source_name="camera",
            healthy=False,
            opened=False,
            running=False,
        )

    @property
    def is_running(self) -> bool:
        with self._status_lock:
            return self._status.running

    def start(self) -> None:
        if self.is_running:
            return
        self._stop_event.clear()
        try:
            self._capture = self._build_capture()
        except Exception as exc:
            self._set_status(
                healthy=False,
                opened=False,
                running=False,
                error_message=str(exc),
            )
            self._capture = None
            return
        capture = self._capture
        if capture is None or not capture.isOpened():
            self._set_status(
                healthy=False,
                opened=False,
                running=False,
                error_message="camera unavailable",
            )
            self._release_capture()
            return

        self._set_status(healthy=True, opened=True, running=True, error_message=None)
        self._thread = threading.Thread(
            target=self._capture_loop,
            name="desktop-camera-capture",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._thread = None
        self._release_capture()
        self._set_status(running=False, opened=False, healthy=False)

    def get_latest_frame(self) -> VideoFrame | None:
        frame = self._buffer.get_latest()
        if frame is None:
            return None
        self._sync_dropped_count()
        return frame

    def get_status(self) -> CaptureStatus:
        self._sync_dropped_count()
        with self._status_lock:
            return replace(self._status)

    def _build_capture(self) -> CameraDevice | None:
        if self._capture_factory is not None:
            return self._capture_factory(self._camera_index)
        if cv2 is None:
            return None
        return _OpenCVCameraDevice(self._camera_index)

    def _capture_loop(self) -> None:
        while not self._stop_event.is_set():
            capture = self._capture
            if capture is None:
                break

            ok, frame = capture.read()
            if not ok or frame is None:
                self._mark_failure("camera read failed")
                time.sleep(self._retry_interval_seconds)
                continue

            timestamp = self._time_fn()
            video_frame = VideoFrame(
                client_id=self._client_id,
                client_type=ClientType.DESKTOP,
                timestamp=timestamp,
                width=frame.width,
                height=frame.height,
                pixel_format=self._pixel_format,
                frame_bytes=self._frame_to_bytes(frame),
            )
            self._buffer.push(video_frame)
            self._set_status(
                healthy=True,
                opened=True,
                running=True,
                last_timestamp=timestamp,
                error_message=None,
            )

        self._set_status(running=False, healthy=False)

    def _frame_to_bytes(self, frame: CameraFrame) -> bytes:
        return frame.tobytes()

    def _mark_failure(self, message: str) -> None:
        with self._status_lock:
            self._status.failure_count += 1
            self._status.healthy = False
            self._status.error_message = message

    def _sync_dropped_count(self) -> None:
        stats = self._buffer.stats()
        with self._status_lock:
            self._status.dropped_count = stats.dropped_count

    def _set_status(
        self,
        *,
        healthy: bool | None = None,
        opened: bool | None = None,
        running: bool | None = None,
        last_timestamp: float | None = None,
        error_message: str | None | _UnchangedType = _UNCHANGED,
    ) -> None:
        with self._status_lock:
            if healthy is not None:
                self._status.healthy = healthy
            if opened is not None:
                self._status.opened = opened
            if running is not None:
                self._status.running = running
            if last_timestamp is not None:
                self._status.last_timestamp = last_timestamp
            if not isinstance(error_message, _UnchangedType):
                self._status.error_message = error_message

    def _release_capture(self) -> None:
        if self._capture is None:
            return
        self._capture.release()
        self._capture = None
