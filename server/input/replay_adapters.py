"""Mock camera and mic adapters that replay recorded sessions."""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from pathlib import Path

import numpy as np
from config.third_party import load_cv2
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

cv2 = load_cv2()


class ReplayCameraAdapter:
    """Replays video.avi as a CameraSource."""

    def __init__(self, session_dir: str, *, loop: bool = True) -> None:
        self._path = Path(session_dir) / "video.avi"
        self._loop = loop
        self._cap = cv2.VideoCapture(str(self._path))
        if not self._cap.isOpened():
            logger.error("ReplayCameraAdapter: cannot open %s", self._path)

        meta_path = Path(session_dir) / "meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                self._meta = json.load(f)
        else:
            self._meta = {}

    def read_frame(self) -> NDArray[np.uint8] | None:
        ok, frame = self._cap.read()
        if not ok:
            if self._loop:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self._cap.read()
                if not ok:
                    return None
            else:
                return None
        return np.asarray(frame, dtype=np.uint8)

    def release(self) -> None:
        self._cap.release()

    def is_opened(self) -> bool:
        return bool(self._cap.isOpened())

    @property
    def frame_width(self) -> int:
        return int(self._meta.get("width", self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)))

    @property
    def frame_height(self) -> int:
        return int(self._meta.get("height", self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))


class ReplayMicAdapter:
    """Replays audio.wav as a MicSource, pushing chunks at real-time pace."""

    def __init__(
        self,
        session_dir: str,
        *,
        loop: bool = True,
        chunk_size: int = 1_600,
        sample_rate: int = 16_000,
    ) -> None:
        self._path = Path(session_dir) / "audio.wav"
        self._loop = loop
        self._chunk_size = chunk_size
        self._sample_rate = sample_rate
        self._queue: queue.Queue[NDArray[np.float32]] = queue.Queue(maxsize=2)
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._recording = False
        self._audio_data: NDArray[np.float32] | None = None

        meta_path = Path(session_dir) / "meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            self._chunk_size = meta.get("chunk_size", self._chunk_size)
            self._sample_rate = meta.get("sample_rate", self._sample_rate)

    def start(self) -> None:
        if self._recording:
            return
        import soundfile as sf

        data, sr = sf.read(str(self._path), dtype="float32")
        self._audio_data = np.asarray(data, dtype=np.float32)
        if self._audio_data.ndim > 1:
            self._audio_data = self._audio_data[:, 0]
        self._sample_rate = sr
        self._stop_event.clear()
        self._recording = True
        self._thread = threading.Thread(
            target=self._push_loop, name="replay-mic", daemon=True
        )
        self._thread.start()
        logger.debug(
            "ReplayMicAdapter: loaded %d samples from %s (sr=%d)",
            len(self._audio_data),
            self._path,
            self._sample_rate,
        )

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._recording = False

    def get_latest_chunk(self) -> NDArray[np.float32] | None:
        latest: NDArray[np.float32] | None = None
        try:
            while True:
                latest = self._queue.get_nowait()
        except queue.Empty:
            pass
        return latest

    @property
    def is_recording(self) -> bool:
        return self._recording

    def _push_loop(self) -> None:
        assert self._audio_data is not None
        chunk_duration = self._chunk_size / self._sample_rate
        offset = 0
        while not self._stop_event.is_set():
            end = offset + self._chunk_size
            if end > len(self._audio_data):
                if self._loop:
                    offset = 0
                    continue
                else:
                    break
            chunk = self._audio_data[offset:end].reshape(-1, 1)
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
            offset = end
            self._stop_event.wait(chunk_duration)
