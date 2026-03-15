from __future__ import annotations

import logging

import numpy as np
from models.protocols import CameraSource
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class CombinedCameraSource:
    def __init__(self, sources: list[tuple[str, CameraSource]]) -> None:
        self._sources = sources
        self._next_index = 0
        self._active_source_name: str | None = None

    def read_frame(self) -> NDArray[np.uint8] | None:
        source_count = len(self._sources)
        if source_count == 0:
            return None

        for offset in range(source_count):
            index = (self._next_index + offset) % source_count
            source_name, source = self._sources[index]
            frame = source.read_frame()
            if frame is None:
                continue

            self._next_index = (index + 1) % source_count
            if self._active_source_name != source_name:
                logger.debug("Frame source switched to %s", source_name)
                self._active_source_name = source_name
            return frame
        return None

    def release(self) -> None:
        for source_name, source in self._sources:
            try:
                source.release()
            except Exception:
                logger.exception("Failed to release camera source %s", source_name)

    def is_opened(self) -> bool:
        return bool(self._sources) and all(
            source.is_opened() for _, source in self._sources
        )

    @property
    def frame_width(self) -> int:
        return self._active_dimension("width")

    @property
    def frame_height(self) -> int:
        return self._active_dimension("height")

    def _active_dimension(self, dimension: str) -> int:
        if self._active_source_name is not None:
            for source_name, source in self._sources:
                if source_name != self._active_source_name:
                    continue
                value = self._source_dimension(source, dimension)
                if value > 0:
                    return value

        for _, source in self._sources:
            value = self._source_dimension(source, dimension)
            if value > 0:
                return value
        return 0

    def _source_dimension(self, source: CameraSource, dimension: str) -> int:
        if dimension == "width":
            return source.frame_width
        return source.frame_height
