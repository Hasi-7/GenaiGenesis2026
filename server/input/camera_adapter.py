from __future__ import annotations

import logging
import socket
import struct
import threading
from collections.abc import Callable

import numpy as np
from config.third_party import OpenCVVideoCaptureProtocol, load_cv2
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

_FRAME_HEADER = struct.Struct("!4sIIIQ")
_FRAME_MAGIC = b"CSJ1"
cv2 = load_cv2()


class LocalCameraAdapter:
    """Local webcam adapter implementing CameraSource protocol."""

    def __init__(self, camera_index: int = 0) -> None:
        self._capture: OpenCVVideoCaptureProtocol = cv2.VideoCapture(camera_index)
        self._subscribers: list[Callable[[NDArray[np.uint8]], None]] = []

    def read_frame(self) -> NDArray[np.uint8] | None:
        ok, frame = self._capture.read()
        if not ok:
            logger.info("read_frame: capture read failed")
            return None
        frame_array: NDArray[np.uint8] = np.asarray(frame, dtype=np.uint8)
        logger.info(
            "read_frame: %dx%d frame, notifying %d subscribers",
            frame_array.shape[1],
            frame_array.shape[0],
            len(self._subscribers),
        )
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


class NetworkCameraAdapter:
    """TCP camera adapter for frames streamed from the Raspberry Pi."""

    def __init__(self, host: str = "0.0.0.0", port: int = 9000) -> None:
        self._host = host
        self._port = port
        self._server_socket: socket.socket | None = None
        self._client_socket: socket.socket | None = None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._latest_frame: NDArray[np.uint8] | None = None
        self._latest_sequence = 0
        self._last_read_sequence = -1
        self._frame_width = 0
        self._frame_height = 0
        self._logged_first_frame = False

        try:
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_socket.bind((host, port))
            server_socket.listen(1)
            server_socket.settimeout(0.5)
        except OSError:
            logger.exception(
                "Failed to bind mirror receiver on %s:%d",
                host,
                port,
            )
            return

        self._server_socket = server_socket
        self._thread = threading.Thread(
            target=self._accept_loop,
            name="mirror-camera-receiver",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "Listening for mirror frames on %s:%d",
            host,
            port,
        )

    def read_frame(self) -> NDArray[np.uint8] | None:
        with self._lock:
            if (
                self._latest_frame is None
                or self._latest_sequence == self._last_read_sequence
            ):
                return None
            self._last_read_sequence = self._latest_sequence
            return self._latest_frame.copy()

    def release(self) -> None:
        self._stop_event.set()
        if self._client_socket is not None:
            try:
                self._client_socket.close()
            except OSError:
                pass
            self._client_socket = None
        if self._server_socket is not None:
            try:
                self._server_socket.close()
            except OSError:
                pass
            self._server_socket = None
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def is_opened(self) -> bool:
        return self._server_socket is not None

    @property
    def frame_width(self) -> int:
        return self._frame_width

    @property
    def frame_height(self) -> int:
        return self._frame_height

    def _accept_loop(self) -> None:
        while not self._stop_event.is_set():
            server_socket = self._server_socket
            if server_socket is None:
                return
            try:
                client_socket, address = server_socket.accept()
            except TimeoutError:
                continue
            except OSError:
                if self._stop_event.is_set():
                    return
                logger.exception("Mirror receiver accept failed")
                continue

            logger.info(
                "Mirror client connected from %s:%d",
                address[0],
                address[1],
            )
            self._logged_first_frame = False
            self._client_socket = client_socket
            client_socket.settimeout(1.0)
            try:
                self._read_client_loop(client_socket)
            except OSError:
                if not self._stop_event.is_set():
                    logger.exception("Mirror client read loop failed")
            finally:
                try:
                    client_socket.close()
                except OSError:
                    pass
                self._client_socket = None
                logger.info("Mirror client disconnected")

    def _read_client_loop(self, client_socket: socket.socket) -> None:
        while not self._stop_event.is_set():
            header = self._recv_exact(client_socket, _FRAME_HEADER.size)
            if header is None:
                return

            (
                magic,
                width,
                height,
                payload_size,
                _timestamp_ns,
            ) = _FRAME_HEADER.unpack(header)
            if magic != _FRAME_MAGIC:
                logger.warning("Dropping frame with invalid magic: %r", magic)
                return

            payload = self._recv_exact(client_socket, payload_size)
            if payload is None:
                return

            jpeg_array = np.frombuffer(payload, dtype=np.uint8)
            decoded = cv2.imdecode(jpeg_array, cv2.IMREAD_COLOR)
            if decoded is None:
                logger.warning("Failed to decode streamed JPEG frame")
                continue

            frame_array: NDArray[np.uint8] = np.asarray(decoded, dtype=np.uint8)
            with self._lock:
                self._latest_frame = frame_array
                self._latest_sequence += 1
                self._frame_width = width if width > 0 else int(frame_array.shape[1])
                self._frame_height = height if height > 0 else int(frame_array.shape[0])
            if not self._logged_first_frame:
                logger.info(
                    "Received first mirror frame: %dx%d (%d bytes JPEG)",
                    self._frame_width,
                    self._frame_height,
                    payload_size,
                )
                self._logged_first_frame = True

    def _recv_exact(
        self,
        client_socket: socket.socket,
        size: int,
    ) -> bytes | None:
        buffer = bytearray()
        while len(buffer) < size and not self._stop_event.is_set():
            try:
                chunk = client_socket.recv(size - len(buffer))
            except TimeoutError:
                continue
            if not chunk:
                return None
            buffer.extend(chunk)
        if len(buffer) != size:
            return None
        return bytes(buffer)
