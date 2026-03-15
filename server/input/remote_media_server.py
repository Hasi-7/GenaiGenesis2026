from __future__ import annotations

import logging
import socket
import struct
import threading
import time
from collections.abc import Callable

import cv2  # type: ignore[import-untyped]
import numpy as np
from core.feedback_codec import (
    MAX_INDICATORS,
    MAX_RECOMMENDATIONS,
    padded_codes,
    recommendation_codes,
    top_indicator_codes,
)
from models.protocols import CameraSource, MicSource
from models.types import CognitiveState, CognitiveStateLabel, FrameAnalysis, LLMResponse
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

_HEADER = struct.Struct("!4sIIIQ")
_FRAME_MAGIC = b"CSJ1"
_AUDIO_MAGIC = b"CSA1"
_EVENT_MAGIC = b"CSM1"

_EVENT_HELLO = 1
_EVENT_STATE = 2
_EVENT_FEEDBACK = 3

_HELLO_PAYLOAD = struct.Struct("!BBH")
_STATE_PAYLOAD = struct.Struct(f"!BBBBI{MAX_INDICATORS}B{MAX_RECOMMENDATIONS}B")
_FEEDBACK_HEADER = struct.Struct("!BBBB")

_HELLO_VERSION = 1

_SOURCE_UNKNOWN = 0
_SOURCE_DESKTOP = 1
_SOURCE_MIRROR = 2

_CAPABILITY_SEND_VIDEO = 1 << 0
_CAPABILITY_SEND_AUDIO = 1 << 1
_CAPABILITY_RECEIVE_STATE = 1 << 2
_CAPABILITY_BINARY_CONTROL = 1 << 4

_STREAM_VIDEO_RECENT = 1 << 0
_STREAM_AUDIO_RECENT = 1 << 1

_STATE_UNKNOWN = 0
_STATE_FOCUSED = 1
_STATE_FATIGUED = 2
_STATE_STRESSED = 3
_STATE_DISTRACTED = 4

_TRIGGER_TRANSITION = 1
_TRIGGER_SUSTAINED_ALERT = 2

_SEVERITY_SOFT = 1
_SEVERITY_WARNING = 2
_SEVERITY_URGENT = 3

_STATE_LABEL_TO_ID = {
    CognitiveStateLabel.UNKNOWN: _STATE_UNKNOWN,
    CognitiveStateLabel.FOCUSED: _STATE_FOCUSED,
    CognitiveStateLabel.FATIGUED: _STATE_FATIGUED,
    CognitiveStateLabel.STRESSED: _STATE_STRESSED,
    CognitiveStateLabel.DISTRACTED: _STATE_DISTRACTED,
}

_TRIGGER_KIND_TO_ID = {
    "transition": _TRIGGER_TRANSITION,
    "sustained_alert": _TRIGGER_SUSTAINED_ALERT,
}

_SEVERITY_TO_ID = {
    "soft": _SEVERITY_SOFT,
    "warning": _SEVERITY_WARNING,
    "urgent": _SEVERITY_URGENT,
}


class RemoteClientSession:
    def __init__(
        self,
        session_id: int,
        client_socket: socket.socket,
        address: tuple[str, int],
        *,
        on_disconnect: Callable[[RemoteClientSession], None],
    ) -> None:
        self._session_id = session_id
        self._address = address
        self._client_socket = client_socket
        self._on_disconnect = on_disconnect
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._send_lock = threading.Lock()
        self._disconnect_notified = False

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
        self._state_sequence = 0
        self._connected_at = time.time()
        self._warned_no_media = False
        self._frame_packets_received = 0
        self._audio_packets_received = 0
        self._hello_received = False
        self._source_kind = _SOURCE_UNKNOWN
        self._capabilities = 0
        self._device_key = f"session-{session_id}-{address[0]}"
        self._display_name = self._device_key

        self._client_socket.settimeout(1.0)

    @property
    def session_id(self) -> int:
        return self._session_id

    @property
    def session_label(self) -> str:
        return (
            f"{self.source_name}:{self._session_id}@"
            f"{self._address[0]}:{self._address[1]}"
        )

    @property
    def source_name(self) -> str:
        if self._source_kind == _SOURCE_DESKTOP:
            return "desktop"
        if self._source_kind == _SOURCE_MIRROR:
            return "mirror"
        return "unknown"

    @property
    def device_key(self) -> str:
        return self._device_key

    @property
    def display_name(self) -> str:
        return self._display_name

    @property
    def remote_ip(self) -> str:
        return self._address[0]

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run,
            name=f"media-session-{self._session_id}",
            daemon=True,
        )
        self._thread.start()

    def join(self, timeout: float | None = None) -> None:
        if self._thread is None:
            return
        self._thread.join(timeout=timeout)

    def close(self, *, wait: bool = True) -> None:
        self._stop_event.set()
        try:
            self._client_socket.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            self._client_socket.close()
        except OSError:
            pass
        if (
            wait
            and self._thread is not None
            and threading.current_thread() is not self._thread
        ):
            self._thread.join(timeout=1.0)

    def is_opened(self) -> bool:
        return not self._stop_event.is_set()

    @property
    def frame_width(self) -> int:
        return self._frame_width

    @property
    def frame_height(self) -> int:
        return self._frame_height

    @property
    def has_state_telemetry(self) -> bool:
        return bool(self._capabilities & _CAPABILITY_RECEIVE_STATE)

    def make_camera_source(self) -> CameraSource:
        return SessionCameraSource(self)

    def make_mic_source(self) -> MicSource:
        return SessionMicSource(self)

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

    def publish_state(self, state: CognitiveState, analysis: FrameAnalysis) -> None:
        if not self.has_state_telemetry:
            return

        now = time.time()
        if now - self._last_state_emit < 0.25:
            return

        stream_flags = 0
        if self._is_recent(self._last_frame_timestamp):
            stream_flags |= _STREAM_VIDEO_RECENT
        if self._is_recent(self._last_audio_timestamp):
            stream_flags |= _STREAM_AUDIO_RECENT

        self._last_state_emit = now
        self._state_sequence += 1
        indicator_codes = padded_codes(top_indicator_codes(state), MAX_INDICATORS)
        recommendation_values = padded_codes(
            recommendation_codes(state),
            MAX_RECOMMENDATIONS,
        )
        payload = _STATE_PAYLOAD.pack(
            _HELLO_VERSION,
            _STATE_LABEL_TO_ID.get(state.label, _STATE_UNKNOWN),
            max(0, min(255, round(state.confidence * 255))),
            stream_flags,
            self._state_sequence,
            *indicator_codes,
            *recommendation_values,
        )
        self._send_packet(
            _EVENT_MAGIC,
            _EVENT_STATE,
            0,
            payload,
            time.time_ns(),
        )

    def publish_feedback(self, response: LLMResponse, state: CognitiveState) -> None:
        del state
        if not self.has_state_telemetry:
            return
        payload = _FEEDBACK_HEADER.pack(
            _HELLO_VERSION,
            _TRIGGER_KIND_TO_ID.get(response.trigger_kind, _TRIGGER_TRANSITION),
            _SEVERITY_TO_ID.get(response.severity, _SEVERITY_SOFT),
            1 if response.should_notify else 0,
        ) + response.feedback_text.encode("utf-8")
        self._send_packet(
            _EVENT_MAGIC,
            _EVENT_FEEDBACK,
            0,
            payload,
            time.time_ns(),
        )

    def _run(self) -> None:
        logger.debug("Remote session %s connected", self.session_label)
        try:
            self._read_client_loop()
        except OSError:
            if not self._stop_event.is_set():
                logger.exception("Remote session %s failed", self.session_label)
        finally:
            self._stop_event.set()
            try:
                self._client_socket.close()
            except OSError:
                pass
            logger.debug(
                "Remote session %s disconnected after %.1fs; frames=%d audio=%d",
                self.session_label,
                max(0.0, time.time() - self._connected_at),
                self._frame_packets_received,
                self._audio_packets_received,
            )
            self._notify_disconnect()

    def _read_client_loop(self) -> None:
        while not self._stop_event.is_set():
            header = self._recv_exact(_HEADER.size)
            if header is None:
                return

            magic, meta1, meta2, payload_size, timestamp_ns = _HEADER.unpack(header)
            payload = self._recv_exact(payload_size)
            if payload is None:
                return

            if magic == _FRAME_MAGIC:
                self._infer_legacy_source(_SOURCE_MIRROR, _CAPABILITY_SEND_VIDEO)
                self._handle_frame(payload, meta1, meta2, timestamp_ns)
            elif magic == _AUDIO_MAGIC:
                self._infer_legacy_source(
                    _SOURCE_DESKTOP,
                    _CAPABILITY_SEND_VIDEO | _CAPABILITY_SEND_AUDIO,
                )
                self._handle_audio(payload, meta1, meta2, timestamp_ns)
            elif magic == _EVENT_MAGIC:
                self._handle_event(meta1, meta2, payload)
            else:
                logger.warning(
                    "Dropping unknown packet %r from session %s",
                    magic,
                    self.session_label,
                )

    def _handle_event(self, message_type: int, flags: int, payload: bytes) -> None:
        if message_type != _EVENT_HELLO:
            return
        if len(payload) < _HELLO_PAYLOAD.size:
            logger.warning(
                "Ignoring malformed hello packet from %s", self.session_label
            )
            return

        version, source_kind, _reserved = _HELLO_PAYLOAD.unpack(
            payload[: _HELLO_PAYLOAD.size]
        )
        if version != _HELLO_VERSION:
            logger.warning(
                "Ignoring unsupported hello version %d from %s",
                version,
                self.session_label,
            )
            return

        self._hello_received = True
        self._source_kind = source_kind
        self._capabilities = flags
        self._parse_identity_payload(payload[_HELLO_PAYLOAD.size :])
        logger.debug(
            "Session %s hello: source=%s device=%s capabilities=0x%08x",
            self.session_label,
            self.source_name,
            self._device_key,
            self._capabilities,
        )

    def _parse_identity_payload(self, payload: bytes) -> None:
        if not payload:
            return
        try:
            decoded = payload.decode("utf-8", errors="ignore").strip()
        except Exception:
            return
        if not decoded:
            return
        device_key, separator, display_name = decoded.partition("|")
        normalized_key = device_key.strip()
        normalized_name = display_name.strip() if separator else ""
        if normalized_key:
            self._device_key = normalized_key[:96]
        if normalized_name:
            self._display_name = normalized_name[:96]
        elif normalized_key:
            self._display_name = normalized_key[:96]

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
            logger.warning(
                "Failed to decode JPEG frame from session %s", self.session_label
            )
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
            logger.debug(
                "Received first video frame for %s: %dx%d (%d bytes JPEG)",
                self.session_label,
                self._frame_width,
                self._frame_height,
                len(payload),
            )

    def _handle_audio(
        self,
        payload: bytes,
        sample_rate: int,
        channels: int,
        timestamp_ns: int,
    ) -> None:
        if len(payload) % 4 != 0:
            logger.warning(
                "Dropping audio payload with invalid length from %s",
                self.session_label,
            )
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
            logger.debug(
                "Received first audio chunk for %s: rate=%d channels=%d samples=%d",
                self.session_label,
                self._audio_sample_rate,
                self._audio_channels,
                len(audio_array),
            )

    def _infer_legacy_source(self, source_kind: int, capability_bits: int) -> None:
        if self._hello_received:
            return
        if self._source_kind == _SOURCE_UNKNOWN:
            self._source_kind = source_kind
        self._capabilities |= capability_bits

    def _recv_exact(self, size: int) -> bytes | None:
        buffer = bytearray()
        while len(buffer) < size and not self._stop_event.is_set():
            try:
                chunk = self._client_socket.recv(size - len(buffer))
            except TimeoutError:
                if (
                    not self._warned_no_media
                    and time.time() - self._connected_at > 5.0
                    and self._frame_packets_received == 0
                    and self._audio_packets_received == 0
                ):
                    self._warned_no_media = True
                    logger.warning(
                        "Remote session %s connected but no media received "
                        "after 5 seconds",
                        self.session_label,
                    )
                continue
            if not chunk:
                return None
            buffer.extend(chunk)
        if len(buffer) != size:
            return None
        return bytes(buffer)

    def _send_packet(
        self,
        magic: bytes,
        meta1: int,
        meta2: int,
        payload: bytes,
        timestamp_ns: int,
    ) -> None:
        packet = _HEADER.pack(magic, meta1, meta2, len(payload), timestamp_ns) + payload
        with self._send_lock:
            if self._stop_event.is_set():
                return
            try:
                self._client_socket.sendall(packet)
            except OSError:
                logger.exception(
                    "Failed to send telemetry to session %s", self.session_label
                )
                self.close(wait=False)

    def _notify_disconnect(self) -> None:
        if self._disconnect_notified:
            return
        self._disconnect_notified = True
        self._on_disconnect(self)

    def _is_recent(self, timestamp: float) -> bool:
        return bool(timestamp) and time.time() - timestamp < 3.0


class SessionCameraSource:
    def __init__(self, session: RemoteClientSession) -> None:
        self._session = session

    def read_frame(self) -> NDArray[np.uint8] | None:
        return self._session.read_frame()

    def release(self) -> None:
        return None

    def is_opened(self) -> bool:
        return self._session.is_opened()

    @property
    def frame_width(self) -> int:
        return self._session.frame_width

    @property
    def frame_height(self) -> int:
        return self._session.frame_height


class SessionMicSource(MicSource):
    def __init__(self, session: RemoteClientSession) -> None:
        self._session = session

    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None

    def get_latest_chunk(self) -> NDArray[np.float32] | None:
        return self._session.get_latest_chunk()

    @property
    def is_recording(self) -> bool:
        return self._session.is_opened()


class RemoteMediaServer:
    """Unified TCP ingress for Electron and Raspberry Pi clients."""

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 9000,
        *,
        on_session_connected: Callable[[RemoteClientSession], None] | None = None,
        on_session_disconnected: Callable[[RemoteClientSession], None] | None = None,
    ) -> None:
        self._host = host
        self._port = port
        self._on_session_connected = on_session_connected
        self._on_session_disconnected = on_session_disconnected
        self._server_socket: socket.socket | None = None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._sessions: dict[int, RemoteClientSession] = {}
        self._sessions_lock = threading.Lock()
        self._next_session_id = 1

        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            server_socket.bind((host, port))
        except OSError as exc:
            server_socket.close()
            if exc.errno == 98:
                raise RuntimeError(
                    f"Network ingress port {port} is already in use. "
                    "Stop the previous server process or set "
                    "COGNITIVESENSE_SERVER_PORT to a different port."
                ) from exc
            raise
        server_socket.listen()
        server_socket.settimeout(0.5)
        self._server_socket = server_socket

        self._thread = threading.Thread(
            target=self._accept_loop,
            name="media-ingress-server",
            daemon=True,
        )
        self._thread.start()
        logger.debug("Listening for remote media on %s:%d", host, port)

    def is_opened(self) -> bool:
        return self._server_socket is not None and not self._stop_event.is_set()

    def release(self) -> None:
        self._stop_event.set()
        if self._server_socket is not None:
            try:
                self._server_socket.close()
            except OSError:
                pass
            self._server_socket = None

        with self._sessions_lock:
            sessions = list(self._sessions.values())

        for session in sessions:
            session.close(wait=False)
        for session in sessions:
            session.join(timeout=1.0)

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
                logger.exception("Remote media accept failed")
                continue

            session = RemoteClientSession(
                self._next_session_id,
                client_socket,
                address,
                on_disconnect=self._handle_session_disconnect,
            )
            self._next_session_id += 1

            with self._sessions_lock:
                self._sessions[session.session_id] = session

            try:
                if self._on_session_connected is not None:
                    self._on_session_connected(session)
                session.start()
            except Exception:
                logger.exception(
                    "Failed to initialize remote session %s", session.session_label
                )
                session.close(wait=False)
                self._handle_session_disconnect(session)

    def _handle_session_disconnect(self, session: RemoteClientSession) -> None:
        with self._sessions_lock:
            self._sessions.pop(session.session_id, None)
        if self._on_session_disconnected is not None:
            self._on_session_disconnected(session)
