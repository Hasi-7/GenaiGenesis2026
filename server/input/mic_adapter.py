from __future__ import annotations

import importlib
import logging
import queue
from collections.abc import Callable
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@runtime_checkable
class _InputStreamProtocol(Protocol):
    def start(self) -> None: ...

    def stop(self) -> None: ...

    def close(self) -> None: ...


class _InputStreamFactory(Protocol):
    def __call__(
        self,
        *,
        samplerate: int,
        channels: int,
        blocksize: int,
        dtype: str,
        callback: Callable[[NDArray[np.float32], int, object, object], None],
    ) -> _InputStreamProtocol: ...


@runtime_checkable
class _SoundDeviceModule(Protocol):
    InputStream: _InputStreamFactory


class LocalMicAdapter:
    """Local microphone adapter implementing MicSource protocol."""

    def __init__(
        self,
        sample_rate: int = 16_000,
        channels: int = 1,
        blocksize: int = 1_600,
    ) -> None:
        self._sample_rate = sample_rate
        self._channels = channels
        self._blocksize = blocksize
        self._queue: queue.Queue[NDArray[np.float32]] = queue.Queue(maxsize=2)
        self._stream: _InputStreamProtocol | None = None
        self._recording = False
        self._subscribers: list[Callable[[NDArray[np.float32]], None]] = []

    def start(self) -> None:
        if self._recording:
            return
        sd = self._load_sounddevice()
        self._stream = sd.InputStream(
            samplerate=self._sample_rate,
            channels=self._channels,
            blocksize=self._blocksize,
            dtype="float32",
            callback=self._audio_callback,
        )
        self._stream.start()
        self._recording = True

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self._recording = False

    def get_latest_chunk(self) -> NDArray[np.float32] | None:
        latest: NDArray[np.float32] | None = None
        try:
            while True:
                latest = self._queue.get_nowait()
        except queue.Empty:
            pass
        if latest is not None:
            logger.info(
                "get_latest_chunk: %d samples", latest.shape[0]
            )
        return latest

    @property
    def is_recording(self) -> bool:
        return self._recording

    def subscribe(self, callback: Callable[[NDArray[np.float32]], None]) -> None:
        """Register a callback invoked on each new audio chunk."""
        self._subscribers.append(callback)

    def _audio_callback(
        self,
        indata: NDArray[np.float32],
        frames: int,
        time_info: object,
        status: object,
    ) -> None:
        del frames, time_info
        if status:
            logger.warning("Audio stream status: %s", status)
        chunk: NDArray[np.float32] = indata.copy()
        logger.info(
            "_audio_callback: %d samples, notifying %d subscribers",
            chunk.shape[0],
            len(self._subscribers),
        )
        for callback in self._subscribers:
            try:
                callback(chunk)
            except Exception:
                logger.exception("Mic subscriber raised an exception")
        try:
            self._queue.put_nowait(chunk)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(chunk)
            except queue.Full:
                pass

    @staticmethod
    def _load_sounddevice() -> _SoundDeviceModule:
        try:
            module = importlib.import_module("sounddevice")
        except OSError as exc:
            raise RuntimeError(
                "sounddevice is installed, but the PortAudio system library is missing. "
                "Install libportaudio2 (and optionally portaudio19-dev) for desktop audio."
            ) from exc
        if not isinstance(module, _SoundDeviceModule):
            raise RuntimeError(
                "sounddevice imported successfully, but InputStream is unavailable."
            )
        return module
