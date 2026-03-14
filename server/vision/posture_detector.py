"""MediaPipe Pose-based posture detector.

Expected input:
    frame_rgb: np.ndarray of shape (H, W, 3) uint8 in RGB color order.

Expected output:
    detect() -> PostureData with shoulder_angle (deg), head_tilt (deg),
                is_slouching flag, or None if no person detected.
    classify() -> ClassifierResult with label in {"upright", "slouched", "leaning"}.

Usage example::

    detector = MediaPipePostureDetector()
    data = detector.detect(frame_rgb)
    if data:
        result = detector.classify(data)
        print(result.label, result.confidence)
    detector.close()

Single-thread assumption: not thread-safe. Call from main pipeline loop only.
"""

from __future__ import annotations

import math

import numpy as np
import mediapipe as mp  # type: ignore[import-untyped]

from models.types import ClassifierResult, PostureData


# MediaPipe Pose landmark indices used for posture estimation
_LEFT_SHOULDER = 11
_RIGHT_SHOULDER = 12
_LEFT_EAR = 7
_RIGHT_EAR = 8
_NOSE = 0

# Thresholds
_SLOUCH_ANGLE_DEG = 8.0    # shoulder line deviation from horizontal
_SLOUCH_RATIO_MIN = 0.55   # min ear-to-shoulder vertical ratio (below = slouching)
_LEAN_TILT_DEG = 20.0      # head tilt from vertical axis


def _angle_from_horizontal(p1: np.ndarray, p2: np.ndarray) -> float:
    """Return the deviation (degrees) of the p1→p2 vector from horizontal.

    Always returns a value in [0, 90] — direction doesn't matter, only tilt.
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    angle = abs(math.degrees(math.atan2(dy, dx)))
    # Fold into [0, 90]: a perfectly horizontal line is 0°, vertical is 90°
    if angle > 90.0:
        angle = 180.0 - angle
    return angle


def _angle_from_vertical(p1: np.ndarray, p2: np.ndarray) -> float:
    """Return the angle (degrees) between the p1→p2 vector and vertical."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    # Angle from vertical = 90 - angle from horizontal
    return abs(90.0 - abs(math.degrees(math.atan2(dy, dx))))


class MediaPipePostureDetector:
    """Estimates body posture from a full-body or upper-body RGB frame.

    Uses MediaPipe Pose to locate shoulder, ear, and nose landmarks,
    then derives shoulder roll angle and head tilt.
    """

    def __init__(self) -> None:
        self._pose = mp.solutions.pose.Pose(  # type: ignore[attr-defined]
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def detect(self, frame_rgb: np.ndarray) -> PostureData | None:
        """Extract posture metrics from an RGB frame.

        Args:
            frame_rgb: uint8 array of shape (H, W, 3).

        Returns:
            PostureData or None if no person is detected.
        """
        results = self._pose.process(frame_rgb)
        if not results.pose_landmarks:
            return None

        lm = results.pose_landmarks.landmark

        def pt(idx: int) -> np.ndarray:
            return np.array([lm[idx].x, lm[idx].y], dtype=np.float32)

        left_shoulder = pt(_LEFT_SHOULDER)
        right_shoulder = pt(_RIGHT_SHOULDER)
        left_ear = pt(_LEFT_EAR)
        right_ear = pt(_RIGHT_EAR)
        nose = pt(_NOSE)

        shoulder_angle = _angle_from_horizontal(left_shoulder, right_shoulder)
        # Head tilt: angle of the ear-to-ear line from horizontal.
        # When upright, both ears are at the same height → ~0°.
        # When the head/body leans sideways, one ear drops → angle increases.
        head_tilt = _angle_from_horizontal(left_ear, right_ear)

        # Forward-slouch detection: as the head drops toward the shoulders,
        # the vertical ear-to-shoulder distance shrinks relative to shoulder width.
        # A front-facing camera can't detect the depth (z) change of forward slouch
        # via shoulder angle alone, so we use this ratio as a second signal.
        shoulder_mid = (left_shoulder + right_shoulder) / 2.0
        ear_mid = (left_ear + right_ear) / 2.0
        shoulder_width = float(np.linalg.norm(left_shoulder - right_shoulder))
        # In image coords y increases downward, so shoulder_mid[1] > ear_mid[1] when upright
        vertical_ratio = (shoulder_mid[1] - ear_mid[1]) / (shoulder_width + 1e-6)

        is_slouching = (
            shoulder_angle > _SLOUCH_ANGLE_DEG
            or vertical_ratio < _SLOUCH_RATIO_MIN
        )

        return PostureData(
            shoulder_angle=round(float(shoulder_angle), 2),
            head_tilt=round(float(head_tilt), 2),
            is_slouching=is_slouching,
        )

    def classify(self, data: PostureData) -> ClassifierResult:
        """Map PostureData to a posture label with confidence.

        Labels:
            "slouched"  – shoulder angle > 15° (forward head / rounded back)
            "leaning"   – head tilt > 20° while not slouching
            "upright"   – otherwise

        Confidence is proportional to how far the measurement deviates
        from (or stays within) the threshold.
        """
        if data.is_slouching:
            excess = data.shoulder_angle - _SLOUCH_ANGLE_DEG
            conf = min(1.0, 0.55 + excess / 30.0)
            return ClassifierResult(label="slouched", confidence=round(conf, 3))

        if data.head_tilt > _LEAN_TILT_DEG:
            excess = data.head_tilt - _LEAN_TILT_DEG
            conf = min(1.0, 0.55 + excess / 40.0)
            return ClassifierResult(label="leaning", confidence=round(conf, 3))

        # Upright: confidence grows as angles stay well within thresholds
        margin_s = (_SLOUCH_ANGLE_DEG - data.shoulder_angle) / _SLOUCH_ANGLE_DEG
        margin_h = (_LEAN_TILT_DEG - data.head_tilt) / _LEAN_TILT_DEG
        conf = 0.5 + float(np.clip((margin_s + margin_h) / 4.0, 0.0, 0.5))
        return ClassifierResult(label="upright", confidence=round(conf, 3))

    def close(self) -> None:
        """Release MediaPipe Pose resources."""
        self._pose.close()


# ----------------------------------------------------------------------
# Maintainer notes
# ----------------------------------------------------------------------
# _SLOUCH_ANGLE_DEG = 15°:
#   The shoulder line should be nearly horizontal when sitting upright.
#   Deviations > 15° correlate with forward head posture or one-shoulder
#   raise. Collected empirically; tighten to 10° for stricter detection.
#
# _LEAN_TILT_DEG = 20°:
#   Head tilts < 20° are natural and transient. > 20° sustained suggests
#   leaning to one side or resting head on hand (fatigue indicator).
#
# model_complexity=1:
#   Balances accuracy and speed. Use 0 for Raspberry Pi (Mirror env),
#   2 for best accuracy on desktop with a dedicated GPU.
#
# smooth_landmarks=True:
#   Applies temporal smoothing inside MediaPipe, reducing per-frame noise
#   at the cost of slightly higher latency (~1-2 frames). Safe at 15 FPS.
