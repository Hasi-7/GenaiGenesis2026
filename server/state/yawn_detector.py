from __future__ import annotations

import time

from models.types import ClassifierResult, YawnData

_JAW_OPEN = "jawOpen"
_MOUTH_CLOSE = "mouthClose"
_MOUTH_SMILE_LEFT = "mouthSmileLeft"
_MOUTH_SMILE_RIGHT = "mouthSmileRight"

_JAW_OPEN_THRESHOLD = 0.52
_MOUTH_CLOSE_MAX = 0.35
_SMILE_MAX = 0.35
_YAWN_DWELL_SECONDS = 1.0


class YawnDetector:
    """Detect sustained yawn-like mouth opening from face blendshapes."""

    def __init__(self) -> None:
        self._started_at: float | None = None

    def detect(self, blendshapes: dict[str, float]) -> YawnData:
        now = time.time()
        jaw_open = float(blendshapes.get(_JAW_OPEN, 0.0))
        mouth_close = float(blendshapes.get(_MOUTH_CLOSE, 0.0))
        smile = (
            float(blendshapes.get(_MOUTH_SMILE_LEFT, 0.0))
            + float(blendshapes.get(_MOUTH_SMILE_RIGHT, 0.0))
        ) / 2.0

        is_yawn_shape = (
            jaw_open >= _JAW_OPEN_THRESHOLD
            and mouth_close <= _MOUTH_CLOSE_MAX
            and smile <= _SMILE_MAX
        )

        if is_yawn_shape:
            if self._started_at is None:
                self._started_at = now
            duration = now - self._started_at
        else:
            self._started_at = None
            duration = 0.0

        return YawnData(
            jaw_open=round(jaw_open, 4),
            mouth_close=round(mouth_close, 4),
            yawn_duration_seconds=round(duration, 4),
            is_yawn_shape=is_yawn_shape,
        )

    def classify(self, data: YawnData) -> ClassifierResult | None:
        if data.yawn_duration_seconds < _YAWN_DWELL_SECONDS:
            return None
        confidence = min(
            1.0,
            0.65 + (data.yawn_duration_seconds - _YAWN_DWELL_SECONDS) / 2.0,
        )
        return ClassifierResult(label="yawning", confidence=round(confidence, 3))

    def reset(self) -> None:
        self._started_at = None
