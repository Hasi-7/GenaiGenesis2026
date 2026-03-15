"""Eye Aspect Ratio (EAR) based blink detector.

Expected input:
    landmarks: np.ndarray of shape (478, 3) float32 — normalized face mesh
    landmarks from MediaPipeFaceLandmarkSource for a single face.

Expected output:
    detect() -> BlinkData with per-eye EAR, average EAR, blink flag,
                and a rolling blinks-per-minute count.
    classify() -> ClassifierResult with label in {"normal", "fatigued",
                  "stressed", "elevated"} and a confidence in [0, 1].

Usage example::

    detector = EarBlinkDetector()
    data = detector.detect(landmarks_478x3)
    result = detector.classify(data)
    print(result.label, result.confidence)

Single-thread assumption: detect() and classify() are NOT thread-safe.
Call them exclusively from the main pipeline loop.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Protocol

import numpy as np
from models.types import BlinkData, ClassifierResult


class BlinkDetectorProtocol(Protocol):
    """Protocol for blink detection via Eye Aspect Ratio."""

    def detect(self, landmarks: np.ndarray) -> BlinkData:
        """Compute EAR values and detect blinks from face landmarks."""
        ...

    def classify(self, blink_data: BlinkData) -> ClassifierResult:
        """
        Classify fatigue based on blink rate.

        Labels: "normal" (15-20/min), "fatigued" (>25/min), "stressed" (<10/min).
        """
        ...

    def reset(self) -> None:
        """Reset blink counter and frame history."""
        ...


# MediaPipe Face Mesh landmark indices for EAR computation.
# Order matches the 6-point EAR formula: P1(outer), P2(upper-outer),
# P3(upper-inner), P4(inner), P5(lower-inner), P6(lower-outer).
_RIGHT_EYE_IDX: tuple[int, ...] = (33, 160, 158, 133, 153, 144)
_LEFT_EYE_IDX: tuple[int, ...] = (362, 385, 387, 263, 380, 373)

_WINDOW_SECONDS = 60.0  # rolling window for blink rate


def _ear(landmarks: np.ndarray, idx: tuple[int, ...]) -> float:
    """Compute Eye Aspect Ratio for a 6-point eye landmark set.

    EAR = (||P2-P6|| + ||P3-P5||) / (2 * ||P1-P4||)

    Only x, y coordinates are used; z is ignored as it adds noise.
    """
    p = landmarks[list(idx), :2]  # shape (6, 2)
    vertical_a = float(np.linalg.norm(p[1] - p[5]))
    vertical_b = float(np.linalg.norm(p[2] - p[4]))
    horizontal = float(np.linalg.norm(p[0] - p[3]))
    if horizontal < 1e-6:
        return 0.0
    return (vertical_a + vertical_b) / (2.0 * horizontal)


class EarBlinkDetector:
    """Detects blinks using Eye Aspect Ratio and tracks blink rate.

    A blink is recorded when EAR stays below *ear_threshold* for at
    least *consecutive_frames* frames.  The detector is stateful and
    designed to be called once per frame from a single thread.
    """

    def __init__(
        self,
        ear_threshold: float = 0.21,
        consecutive_frames: int = 3,
    ) -> None:
        self._threshold = ear_threshold
        self._consec_required = consecutive_frames

        self._consec_count: int = 0  # frames eye has been below threshold
        self._in_blink: bool = False  # True while eye is currently closed
        self._blink_times: deque[float] = deque()  # timestamps of completed blinks

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def detect(self, landmarks: np.ndarray) -> BlinkData:
        """Compute EAR and update blink state.

        Args:
            landmarks: float32 array shape (478, 3) for a single face.

        Returns:
            BlinkData populated with EAR values and blink state.
        """
        ear_r = _ear(landmarks, _RIGHT_EYE_IDX)
        ear_l = _ear(landmarks, _LEFT_EYE_IDX)
        ear_avg = (ear_r + ear_l) / 2.0

        blink_detected = self._update_blink_state(ear_avg)
        bpm = self._blinks_per_minute()

        return BlinkData(
            ear_left=ear_l,
            ear_right=ear_r,
            ear_average=ear_avg,
            blink_detected=blink_detected,
            blinks_per_minute=bpm,
        )

    def classify(self, data: BlinkData) -> ClassifierResult:
        """Map blink rate to a cognitive-state label.

        Thresholds:
            <10 bpm  -> "stressed"   (suppressed blinking under stress)
            15-20 bpm-> "normal"
            >25 bpm  -> "fatigued"   (fatigue-induced excessive blinking)
            10-15 or 20-25 bpm -> "elevated" (transitional zone)

        Confidence is a simple linear interpolation within each band.
        """
        bpm = data.blinks_per_minute
        # No blinks recorded yet — not enough data to classify
        if bpm == 0.0:
            return ClassifierResult(label="unknown", confidence=0.0)
        if bpm < 10.0:
            label = "stressed"
            conf = max(0.5, 1.0 - bpm / 20.0)
        elif bpm <= 15.0:
            label = "elevated"
            conf = 0.55
        elif bpm <= 20.0:
            label = "normal"
            conf = 0.5 + (1.0 - abs(bpm - 17.5) / 5.0) * 0.5
        elif bpm <= 25.0:
            label = "elevated"
            conf = 0.55
        else:
            label = "fatigued"
            conf = min(1.0, 0.5 + (bpm - 25.0) / 20.0)

        return ClassifierResult(label=label, confidence=round(conf, 3))

    def reset(self) -> None:
        """Clear all state (e.g. when camera switches or subject changes)."""
        self._consec_count = 0
        self._in_blink = False
        self._blink_times.clear()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _update_blink_state(self, ear_avg: float) -> bool:
        """State machine: returns True on the frame a blink is *completed*."""
        if ear_avg < self._threshold:
            self._consec_count += 1
            self._in_blink = True
            return False
        else:
            if self._in_blink and self._consec_count >= self._consec_required:
                self._record_blink()
                self._in_blink = False
                self._consec_count = 0
                return True
            self._consec_count = 0
            self._in_blink = False
            return False

    def _record_blink(self) -> None:
        now = time.time()
        self._blink_times.append(now)
        # Prune timestamps outside the rolling window
        cutoff = now - _WINDOW_SECONDS
        while self._blink_times and self._blink_times[0] < cutoff:
            self._blink_times.popleft()

    def _blinks_per_minute(self) -> float:
        if not self._blink_times:
            return 0.0
        now = time.time()
        cutoff = now - _WINDOW_SECONDS
        # Count only timestamps inside window
        count = sum(1 for t in self._blink_times if t >= cutoff)
        elapsed = min(_WINDOW_SECONDS, now - self._blink_times[0] + 1e-3)
        return count / elapsed * 60.0


# ----------------------------------------------------------------------
# Maintainer notes
# ----------------------------------------------------------------------
# EAR threshold = 0.21:
#   Empirically derived from Soukupova & Cech (2016). Values below ~0.21
#   reliably indicate eye closure for most subjects. Setting it higher
#   (e.g. 0.25) creates false positives from squinting; lower (e.g. 0.17)
#   misses partial blinks. Expose via PipelineConfig.ear_blink_threshold
#   if per-user calibration is desired.
#
# consecutive_frames = 3:
#   At 15 FPS, 3 frames ≈ 200 ms — the minimum physiological blink
#   duration. This filters out single-frame EAR dips caused by motion
#   blur or landmark jitter without noticeably delaying detection.
#
# Blink rate bands (bpm):
#   Sourced from ophthalmology literature (mean ~15-20 bpm at rest).
#   Stress-induced suppression typically drops rate below 10 bpm;
#   fatigue causes incomplete lid closure and elevated compensatory
#   blink rate (>25 bpm). Adjust thresholds after collecting labelled
#   data from real users.
