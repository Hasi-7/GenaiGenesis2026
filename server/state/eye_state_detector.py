from __future__ import annotations

import time
from collections import deque

from models.types import ClassifierResult, EyeStateData

_EYE_BLINK_LEFT = "eyeBlinkLeft"
_EYE_BLINK_RIGHT = "eyeBlinkRight"

_CLOSED_THRESHOLD = 0.62
_PROLONGED_CLOSURE_SECONDS = 1.2
_PERCLOS_WINDOW_SECONDS = 30.0
_PERCLOS_FATIGUE_THRESHOLD = 0.35


class EyeStateDetector:
    """Tracks eye openness, closure duration, and PERCLOS from blendshapes."""

    def __init__(self) -> None:
        self._closed_started_at: float | None = None
        self._closure_samples: deque[tuple[float, bool]] = deque()

    def detect(self, blendshapes: dict[str, float]) -> EyeStateData:
        now = time.time()
        blink_left = float(blendshapes.get(_EYE_BLINK_LEFT, 0.0))
        blink_right = float(blendshapes.get(_EYE_BLINK_RIGHT, 0.0))
        openness_left = 1.0 - blink_left
        openness_right = 1.0 - blink_right
        openness_average = (openness_left + openness_right) / 2.0
        is_closed = blink_left >= _CLOSED_THRESHOLD and blink_right >= _CLOSED_THRESHOLD

        if is_closed:
            if self._closed_started_at is None:
                self._closed_started_at = now
            closure_duration = now - self._closed_started_at
        else:
            self._closed_started_at = None
            closure_duration = 0.0

        self._closure_samples.append((now, is_closed))
        cutoff = now - _PERCLOS_WINDOW_SECONDS
        while self._closure_samples and self._closure_samples[0][0] < cutoff:
            self._closure_samples.popleft()

        if self._closure_samples:
            closed_count = sum(1 for _, closed in self._closure_samples if closed)
            perclos = closed_count / len(self._closure_samples)
        else:
            perclos = 0.0

        return EyeStateData(
            openness_left=round(openness_left, 4),
            openness_right=round(openness_right, 4),
            openness_average=round(openness_average, 4),
            is_closed=is_closed,
            closure_duration_seconds=round(closure_duration, 4),
            perclos=round(perclos, 4),
        )

    def classify(self, eye_state: EyeStateData) -> ClassifierResult | None:
        if eye_state.closure_duration_seconds >= _PROLONGED_CLOSURE_SECONDS:
            confidence = min(
                1.0,
                0.7
                + (eye_state.closure_duration_seconds - _PROLONGED_CLOSURE_SECONDS)
                / 2.0,
            )
            return ClassifierResult(
                label="eyes_closed", confidence=round(confidence, 3)
            )

        if eye_state.perclos >= _PERCLOS_FATIGUE_THRESHOLD:
            confidence = min(
                1.0,
                0.55 + (eye_state.perclos - _PERCLOS_FATIGUE_THRESHOLD),
            )
            return ClassifierResult(label="fatigued", confidence=round(confidence, 3))

        return None

    def reset(self) -> None:
        self._closed_started_at = None
        self._closure_samples.clear()
