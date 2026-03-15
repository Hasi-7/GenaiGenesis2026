"""Iris-based gaze ratio detector using MediaPipe Face Mesh iris landmarks.

Expected input:
    landmarks: np.ndarray of shape (478, 3) float32 — single-face normalized
    face mesh output from MediaPipeFaceLandmarkSource (refine_landmarks=True).
    Iris landmarks at indices 468 (left iris center) and 473 (right iris center)
    must be present (requires refine_landmarks=True).

Expected output:
    detect() -> GazeData with horizontal_ratio in [-1, 1] (negative=left),
                vertical_ratio in [-1, 1] (negative=up), and a direction string.
    classify() -> ClassifierResult with label "focused" or "distracted".

Usage example::

    detector = IrisGazeDetector()
    gaze = detector.detect(landmarks_478x3)
    result = detector.classify(gaze)
    print(gaze.direction, result.label)

Single-thread assumption: all methods are NOT thread-safe.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Protocol

import numpy as np
from models.types import ClassifierResult, GazeData


class EyeMovementDetectorProtocol(Protocol):
    """Protocol for gaze direction estimation from iris landmarks."""

    def detect(self, landmarks: np.ndarray) -> GazeData:
        """
        Compute gaze direction from iris position relative to eye contours.

        Uses iris landmarks 468 (left center) and 473 (right center).
        """
        ...

    def classify(self, gaze_data: GazeData) -> ClassifierResult:
        """
        Classify attention based on gaze stability.

        Labels: "focused" (sustained center), "distracted" (frequent off-center).
        """
        ...


# -----------------------------------------------------------------------
# Landmark index constants (MediaPipe Face Mesh with refine_landmarks=True)
# -----------------------------------------------------------------------

# Iris centers
# MediaPipe assigns 468 to the RIGHT iris and 473 to the LEFT iris.
_LEFT_IRIS = 473
_RIGHT_IRIS = 468

# Eye corners used to define the gaze bounding box.
# Left eye: outer=263, inner=362  |  Right eye: outer=33, inner=133
_LEFT_EYE_OUTER = 263
_LEFT_EYE_INNER = 362
_LEFT_EYE_TOP = 386
_LEFT_EYE_BOTTOM = 374

_RIGHT_EYE_OUTER = 33
_RIGHT_EYE_INNER = 133
_RIGHT_EYE_TOP = 159
_RIGHT_EYE_BOTTOM = 145


def _gaze_ratio(
    iris: np.ndarray,
    corner_a: np.ndarray,
    corner_b: np.ndarray,
) -> float:
    """Return iris position as ratio in [-1, 1] along the corner_a→corner_b axis.

    0.0 means iris is perfectly centred; +1.0 means fully at corner_b.
    """
    axis = corner_b - corner_a
    length = float(np.linalg.norm(axis))
    if length < 1e-6:
        return 0.0
    proj = float(np.dot(iris - corner_a, axis)) / length
    # Normalise to [-1, 1] where 0 is midpoint
    ratio = (proj / length) * 2.0 - 1.0
    return float(np.clip(ratio, -1.0, 1.0))


class IrisGazeDetector:
    """Computes normalised gaze ratios from iris landmark positions.

    Maintains a short rolling window to classify sustained attention
    (focused) vs. frequent off-centre gaze (distracted).
    """

    def __init__(
        self,
        center_threshold: float = 0.2,
        window: int = 15,
        focused_dwell_seconds: float = 0.8,
        distracted_dwell_seconds: float = 0.6,
    ) -> None:
        """
        Args:
            center_threshold: |ratio| below this value is considered "centre".
            window: number of recent frames used for classify() decision.
        """
        self._center_threshold = center_threshold
        self._window = window
        self._focused_dwell_seconds = focused_dwell_seconds
        self._distracted_dwell_seconds = distracted_dwell_seconds
        self._h_history: deque[float] = deque(maxlen=window)
        self._v_history: deque[float] = deque(maxlen=window)
        self._focused_started_at: float | None = None
        self._distracted_started_at: float | None = None

    def detect(self, landmarks: np.ndarray) -> GazeData:
        """Compute gaze ratios and direction for a single face.

        Args:
            landmarks: float32 array shape (478, 3) for one face.

        Returns:
            GazeData with horizontal_ratio, vertical_ratio, and direction.
        """
        # Average left/right horizontal ratios.
        # Left eye: outer(263) -> inner(362) axis points toward the nose.
        # Right eye: use inner(133) -> outer(33) to keep the same axis
        # direction after the horizontally flipped debug_webcam path.
        h_left = _gaze_ratio(
            landmarks[_LEFT_IRIS, :2],
            landmarks[_LEFT_EYE_OUTER, :2],
            landmarks[_LEFT_EYE_INNER, :2],
        )
        h_right = _gaze_ratio(
            landmarks[_RIGHT_IRIS, :2],
            landmarks[_RIGHT_EYE_INNER, :2],
            landmarks[_RIGHT_EYE_OUTER, :2],
        )
        h = float(np.mean([h_left, h_right]))

        # Vertical: use left eye top/bottom (mirrored between eyes so one suffices)
        v_left = _gaze_ratio(
            landmarks[_LEFT_IRIS, :2],
            landmarks[_LEFT_EYE_TOP, :2],
            landmarks[_LEFT_EYE_BOTTOM, :2],
        )
        v_right = _gaze_ratio(
            landmarks[_RIGHT_IRIS, :2],
            landmarks[_RIGHT_EYE_TOP, :2],
            landmarks[_RIGHT_EYE_BOTTOM, :2],
        )
        v = float(np.mean([v_left, v_right]))

        self._h_history.append(h)
        self._v_history.append(v)

        direction = self._direction(h, v)
        return GazeData(
            horizontal_ratio=round(h, 4),
            vertical_ratio=round(v, 4),
            direction=direction,
        )

    def classify(self, gaze: GazeData) -> ClassifierResult:
        """Classify focus level from the rolling gaze history.

        Sustained centre gaze over the window -> "focused".
        Frequent large deviations -> "distracted".
        """
        if len(self._h_history) < self._window // 2:
            # Not enough history yet
            return ClassifierResult(label="unknown", confidence=0.0)

        now = time.time()
        h_arr = np.array(self._h_history)
        v_arr = np.array(self._v_history)
        centred = (np.abs(h_arr) < self._center_threshold) & (
            np.abs(v_arr) < self._center_threshold
        )
        centre_ratio = float(centred.mean())
        centered_now = (
            abs(gaze.horizontal_ratio) < self._center_threshold
            and abs(gaze.vertical_ratio) < self._center_threshold
        )

        if centered_now:
            if self._focused_started_at is None:
                self._focused_started_at = now
            self._distracted_started_at = None
        else:
            if self._distracted_started_at is None:
                self._distracted_started_at = now
            self._focused_started_at = None

        focused_dwell = (
            0.0 if self._focused_started_at is None else now - self._focused_started_at
        )
        distracted_dwell = (
            0.0
            if self._distracted_started_at is None
            else now - self._distracted_started_at
        )

        if centre_ratio >= 0.8 and focused_dwell >= self._focused_dwell_seconds:
            label = "focused"
            conf = 0.5 + centre_ratio * 0.5
        elif (
            centre_ratio <= 0.45 and distracted_dwell >= self._distracted_dwell_seconds
        ):
            label = "distracted"
            conf = 0.5 + (1.0 - centre_ratio) * 0.5
        else:
            label = "unknown"
            conf = 0.0

        return ClassifierResult(label=label, confidence=round(conf, 3))

    def reset(self) -> None:
        """Clear gaze history (e.g. on subject change)."""
        self._h_history.clear()
        self._v_history.clear()
        self._focused_started_at = None
        self._distracted_started_at = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _direction(self, h: float, v: float) -> str:
        thr = self._center_threshold
        if abs(h) < thr and abs(v) < thr:
            return "center"
        if abs(h) >= abs(v):
            return "right" if h > 0 else "left"
        return "down" if v > 0 else "up"


# ----------------------------------------------------------------------
# Maintainer notes
# ----------------------------------------------------------------------
# center_threshold = 0.2:
#   Gaze ratios of ±0.2 represent ~20 % displacement from the centre of
#   the eye bounding box. Smaller values are too sensitive to natural
#   micro-saccades; larger values mask real distractions. Tune by
#   recording gaze data while users perform a known "focused" task and
#   inspecting the ratio distribution.
#
# window = 15 frames:
#   At 15 FPS this covers 1 second of history — long enough to smooth
#   saccades but short enough to react to genuine attention shifts.
#   Increase to 30 (2 s) for less noisy environments.
#
# Iris landmark indices 468 / 473:
#   Available only when refine_landmarks=True is passed to Face Mesh.
#   If iris landmarks are missing (e.g. older MediaPipe versions), fall
#   back to pupil approximation via the inner eye corner centroid.
