from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import replace
from typing import Protocol

from .ring_buffer import LatestValueBuffer
from .types import AudioChunk, CaptureStatus, ClientType

try:
    import sounddevice as sd  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover - exercised only when sounddevice is missing.
    sd = None


class _UnchangedType:
    pass


_UNCHANGED = _UnchangedType()


class AudioInputBuffer(Protocol):
    def tobytes(self) -> bytes: ...


class AudioStreamCallback(Protocol):
    def __call__(
        self,
        indata: AudioInputBuffer,
        frames: int,
        time_info: object,
        status: object,
    ) -> None: ...


class AudioStream(Protocol):
    def start(self) -> None: ...

    def stop(self) -> None: ...

    def close(self) -> None: ...


class AudioStreamFactory(Protocol):
    def __call__(
        self,
        *,
        samplerate: int,
        channels: int,
        blocksize: int,
        dtype: str,
        callback: AudioStreamCallback,
    ) -> AudioStream: ...


class MicCapture:
    """Buffered microphone capture for the desktop client."""

    def __init__(
        self,
        client_id: str,
        sample_rate: int = 16_000,
        channels: int = 1,
        blocksize: int = 1_600,
        buffer_size: int = 4,
        stream_factory: AudioStreamFactory | None = None,
        time_fn: Callable[[], float] | None = None,
    ) -> None:
        self._client_id = client_id
        self._sample_rate = sample_rate
        self._channels = channels
        self._blocksize = blocksize
        self._stream_factory = stream_factory
        self._time_fn = time_fn or time.time
        self._buffer = LatestValueBuffer[AudioChunk](buffer_size)
        self._stream: AudioStream | None = None
        self._status_lock = threading.Lock()
        self._status = CaptureStatus(
            source_name="microphone",
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
        try:
            self._stream = self._build_stream()
        except Exception as exc:
            self._set_status(
                healthy=False,
                opened=False,
                running=False,
                error_message=str(exc),
            )
            self._stream = None
            return
        if self._stream is None:
            self._set_status(
                healthy=False,
                opened=False,
                running=False,
                error_message="microphone unavailable",
            )
            return

        try:
            self._stream.start()
        except Exception as exc:  # pragma: no cover - hardware dependent.
            self._set_status(
                healthy=False,
                opened=False,
                running=False,
                error_message=str(exc),
            )
            self._stream = None
            return

        self._set_status(healthy=True, opened=True, running=True, error_message=None)

    def stop(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:  # pragma: no cover - hardware dependent cleanup.
                pass
        self._stream = None
        self._set_status(running=False, opened=False, healthy=False)

    def get_latest_audio_chunk(self) -> AudioChunk | None:
        chunk = self._buffer.get_latest()
        if chunk is None:
            return None
        self._sync_dropped_count()
        return chunk

    def get_status(self) -> CaptureStatus:
        self._sync_dropped_count()
        with self._status_lock:
            return replace(self._status)

    def _build_stream(self) -> AudioStream | None:
        if self._stream_factory is not None:
            return self._stream_factory(
                samplerate=self._sample_rate,
                channels=self._channels,
                blocksize=self._blocksize,
                dtype="float32",
                callback=self._audio_callback,
            )
        if sd is None:
            return None
        return sd.InputStream(
            samplerate=self._sample_rate,
            channels=self._channels,
            blocksize=self._blocksize,
            dtype="float32",
            callback=self._audio_callback,
        )

    def _audio_callback(
        self,
        indata: AudioInputBuffer,
        frames: int,
        time_info: object,
        status: object,
    ) -> None:
        del time_info
        if status:
            self._mark_failure(str(status))
        timestamp = self._time_fn()
        pcm_bytes = indata.tobytes()

        chunk = AudioChunk(
            client_id=self._client_id,
            client_type=ClientType.DESKTOP,
            timestamp=timestamp,
            sample_rate=self._sample_rate,
            channels=self._channels,
            sample_count=frames,
            pcm_bytes=pcm_bytes,
        )
        self._buffer.push(chunk)
        self._set_status(
            healthy=not bool(status),
            opened=True,
            running=True,
            last_timestamp=timestamp,
            error_message=str(status) if status else None,
        )

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
