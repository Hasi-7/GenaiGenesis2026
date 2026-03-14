from __future__ import annotations

import json
import logging
import socket
import struct
import threading
import time

import cv2  # type: ignore[import-untyped]
import numpy as np
from models.protocols import MicSource
from models.types import CognitiveState, FrameAnalysis, LLMResponse
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

_HEADER = struct.Struct("!4sIIIQ")
_FRAME_MAGIC = b"CSJ1"
_AUDIO_MAGIC = b"CSA1"
_EVENT_MAGIC = b"CSM1"


class RemoteMediaServer:
    """Bidirectional TCP media bridge for frontend clients."""

    def __init__(self, host: str = "0.0.0.0", port: int = 9100) -> None:
        self._host = host
        self._port = port
        self._server_socket: socket.socket | None = None
        self._client_socket: socket.socket | None = None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._send_lock = threading.Lock()

        self._latest_frame: NDArray[np.uint8] | None = None
        self._latest_frame_sequence = 0
        self._last_frame_read_sequence = -1
        self._frame_width = 0
        self._frame_height = 0
        self._last_frame_timestamp = 0.0

        self._latest_audio: NDArray[np.float32] | None = None
        self._latest_audio_sequence = 0
        self._last_audio_read_sequence = -1
        self._audio_sample_rate = 16_000
        self._audio_channels = 1
        self._last_audio_timestamp = 0.0

        self._last_state_emit = 0.0
        self._client_connected_at = 0.0
        self._warned_no_media = False
        self._frame_packets_received = 0
        self._audio_packets_received = 0

        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            server_socket.bind((host, port))
        except OSError as exc:
            server_socket.close()
            if exc.errno == 98:
                msg = (
                    f"Remote media server port {port} is already in use. "
                    "Stop the previous server process or set "
                    "COGNITIVESENSE_SERVER_PORT to a different port."
                )
                raise RuntimeError(msg) from exc
            raise
        server_socket.listen(1)
        server_socket.settimeout(0.5)
        self._server_socket = server_socket

        self._thread = threading.Thread(
            target=self._accept_loop,
            name="frontend-media-server",
            daemon=True,
        )
        self._thread.start()
        logger.info("Listening for frontend media on %s:%d", host, port)

    def make_camera_source(self) -> RemoteCameraAdapter:
        return RemoteCameraAdapter(self)

    def make_mic_source(self) -> RemoteMicAdapter:
        return RemoteMicAdapter(self)

    def read_frame(self) -> NDArray[np.uint8] | None:
        with self._lock:
            if (
                self._latest_frame is None
                or self._latest_frame_sequence == self._last_frame_read_sequence
            ):
                return None
            self._last_frame_read_sequence = self._latest_frame_sequence
            return self._latest_frame.copy()

    def get_latest_chunk(self) -> NDArray[np.float32] | None:
        with self._lock:
            if (
                self._latest_audio is None
                or self._latest_audio_sequence == self._last_audio_read_sequence
            ):
                return None
            self._last_audio_read_sequence = self._latest_audio_sequence
            return self._latest_audio.copy()

    @property
    def frame_width(self) -> int:
        return self._frame_width

    @property
    def frame_height(self) -> int:
        return self._frame_height

    @property
    def audio_sample_rate(self) -> int:
        return self._audio_sample_rate

    def is_opened(self) -> bool:
        return self._server_socket is not None and not self._stop_event.is_set()

    @property
    def has_client(self) -> bool:
        return self._client_socket is not None

    def publish_state(self, state: CognitiveState, analysis: FrameAnalysis) -> None:
        now = time.time()
        if now - self._last_state_emit < 0.25:
            return
        self._last_state_emit = now
        self._send_event(
            {
                "type": "state",
                "label": state.label.name,
                "confidence": state.confidence,
                "signals": [
                    {"label": signal.label, "confidence": signal.confidence}
                    for signal in state.contributing_signals[:6]
                ],
                "transport": {
                    "video": self._is_recent(self._last_frame_timestamp),
                    "audio": self._is_recent(self._last_audio_timestamp),
                },
                "updatedAt": analysis.timestamp,
            }
        )

    def publish_feedback(
        self,
        response: LLMResponse,
        state: CognitiveState,
    ) -> None:
        self._send_event(
            {
                "type": "feedback",
                "text": response.feedback_text,
                "label": state.label.name,
                "timestamp": response.timestamp,
            }
        )

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
                logger.exception("Frontend media accept failed")
                continue

            logger.info(
                "Frontend media client connected from %s:%d; waiting for frames/audio",
                address[0],
                address[1],
            )
            client_socket.settimeout(1.0)
            with self._send_lock:
                if self._client_socket is not None:
                    try:
                        self._client_socket.close()
                    except OSError:
                        pass
                self._client_socket = client_socket
            self._client_connected_at = time.time()
            self._warned_no_media = False
            self._frame_packets_received = 0
            self._audio_packets_received = 0

            try:
                self._read_client_loop(client_socket)
            except OSError:
                if not self._stop_event.is_set():
                    logger.exception("Frontend media client loop failed")
            finally:
                with self._send_lock:
                    if self._client_socket is client_socket:
                        self._client_socket = None
                try:
                    client_socket.close()
                except OSError:
                    pass
                connection_seconds = max(0.0, time.time() - self._client_connected_at)
                logger.info(
                    (
                        "Frontend media client disconnected after %.1fs; "
                        "frames=%d audio=%d"
                    ),
                    connection_seconds,
                    self._frame_packets_received,
                    self._audio_packets_received,
                )
                self._client_connected_at = 0.0

    def _read_client_loop(self, client_socket: socket.socket) -> None:
        while not self._stop_event.is_set():
            header = self._recv_exact(client_socket, _HEADER.size)
            if header is None:
                return

            magic, meta1, meta2, payload_size, timestamp_ns = _HEADER.unpack(header)
            payload = self._recv_exact(client_socket, payload_size)
            if payload is None:
                return

            if magic == _FRAME_MAGIC:
                self._handle_frame(payload, meta1, meta2, timestamp_ns)
            elif magic == _AUDIO_MAGIC:
                self._handle_audio(payload, meta1, meta2, timestamp_ns)
            elif magic == _EVENT_MAGIC:
                logger.debug("Ignoring frontend control event: %s", payload[:64])
            else:
                logger.warning("Unknown frontend media packet: %r", magic)

    def _handle_frame(
        self,
        payload: bytes,
        width: int,
        height: int,
        timestamp_ns: int,
    ) -> None:
        jpeg_array = np.frombuffer(payload, dtype=np.uint8)
        decoded = cv2.imdecode(jpeg_array, cv2.IMREAD_COLOR)
        if decoded is None:
            logger.warning("Failed to decode frontend JPEG frame")
            return

        frame_array: NDArray[np.uint8] = np.asarray(decoded, dtype=np.uint8)
        with self._lock:
            self._latest_frame = frame_array
            self._latest_frame_sequence += 1
            self._frame_packets_received += 1
            self._frame_width = width if width > 0 else int(frame_array.shape[1])
            self._frame_height = height if height > 0 else int(frame_array.shape[0])
            self._last_frame_timestamp = (
                timestamp_ns / 1_000_000_000 if timestamp_ns else time.time()
            )
        if self._frame_packets_received == 1:
            logger.info(
                "Received first frontend video frame: %dx%d (%d bytes JPEG)",
                self._frame_width,
                self._frame_height,
                len(payload),
            )
        elif self._frame_packets_received % 60 == 0:
            logger.info(
                "Frontend video active: frames=%d latest=%dx%d",
                self._frame_packets_received,
                self._frame_width,
                self._frame_height,
            )

    def _handle_audio(
        self,
        payload: bytes,
        sample_rate: int,
        channels: int,
        timestamp_ns: int,
    ) -> None:
        if len(payload) % 4 != 0:
            logger.warning("Dropping frontend audio payload with invalid length")
            return
        audio = np.frombuffer(payload, dtype=np.float32)
        if channels > 1:
            audio = audio.reshape(-1, channels).mean(axis=1)
        audio_array: NDArray[np.float32] = np.asarray(audio, dtype=np.float32)
        with self._lock:
            self._latest_audio = audio_array
            self._latest_audio_sequence += 1
            self._audio_packets_received += 1
            self._audio_sample_rate = sample_rate if sample_rate > 0 else 16_000
            self._audio_channels = max(channels, 1)
            self._last_audio_timestamp = (
                timestamp_ns / 1_000_000_000 if timestamp_ns else time.time()
            )
        if self._audio_packets_received == 1:
            logger.info(
                "Received first frontend audio chunk: rate=%d channels=%d samples=%d",
                self._audio_sample_rate,
                self._audio_channels,
                len(audio_array),
            )
        elif self._audio_packets_received % 40 == 0:
            logger.info(
                "Frontend audio active: chunks=%d rate=%d",
                self._audio_packets_received,
                self._audio_sample_rate,
            )

    def _send_event(self, payload: dict[str, object]) -> None:
        encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        packet = (
            _HEADER.pack(_EVENT_MAGIC, 0, 0, len(encoded), time.time_ns()) + encoded
        )
        with self._send_lock:
            client_socket = self._client_socket
            if client_socket is None:
                return
            try:
                client_socket.sendall(packet)
            except OSError:
                logger.exception("Failed to send frontend event")

    def _recv_exact(self, client_socket: socket.socket, size: int) -> bytes | None:
        buffer = bytearray()
        while len(buffer) < size and not self._stop_event.is_set():
            try:
                chunk = client_socket.recv(size - len(buffer))
            except TimeoutError:
                if (
                    not self._warned_no_media
                    and self._client_connected_at
                    and time.time() - self._client_connected_at > 5.0
                    and self._frame_packets_received == 0
                    and self._audio_packets_received == 0
                ):
                    self._warned_no_media = True
                    logger.warning(
                        "Frontend TCP client connected but no media packets "
                        "received after 5 seconds"
                    )
                continue
            if not chunk:
                return None
            buffer.extend(chunk)
        if len(buffer) != size:
            return None
        return bytes(buffer)

    def _is_recent(self, timestamp: float) -> bool:
        return bool(timestamp) and time.time() - timestamp < 3.0


class RemoteCameraAdapter:
    def __init__(self, server: RemoteMediaServer) -> None:
        self._server = server

    def read_frame(self) -> NDArray[np.uint8] | None:
        return self._server.read_frame()

    def release(self) -> None:
        self._server.release()

    def is_opened(self) -> bool:
        return self._server.is_opened()

    @property
    def frame_width(self) -> int:
        return self._server.frame_width

    @property
    def frame_height(self) -> int:
        return self._server.frame_height


class RemoteMicAdapter(MicSource):
    def __init__(self, server: RemoteMediaServer) -> None:
        self._server = server

    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None

    def get_latest_chunk(self) -> NDArray[np.float32] | None:
        return self._server.get_latest_chunk()

    @property
    def is_recording(self) -> bool:
        return self._server.has_client
