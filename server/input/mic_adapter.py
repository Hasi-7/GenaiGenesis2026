from __future__ import annotations

import logging
import queue
from collections.abc import Callable

import numpy as np
import sounddevice as sd  # type: ignore[import-untyped]
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


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
        self._stream: sd.InputStream | None = None
        self._recording = False
        self._subscribers: list[Callable[[NDArray[np.float32]], None]] = []

    def start(self) -> None:
        if self._recording:
            return
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
        status: sd.CallbackFlags,
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
