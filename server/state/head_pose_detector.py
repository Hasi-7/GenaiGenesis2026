from __future__ import annotations

import math
import time

import cv2  # type: ignore[import-untyped]
import numpy as np
from models.types import ClassifierResult, HeadPoseData
from numpy.typing import NDArray

_NOSE_TIP = 1
_CHIN = 152
_LEFT_EYE_OUTER = 263
_RIGHT_EYE_OUTER = 33
_LEFT_MOUTH = 61
_RIGHT_MOUTH = 291

_MODEL_POINTS = np.array(
    [
        (0.0, 0.0, 0.0),
        (0.0, -63.6, -12.5),
        (-43.3, 32.7, -26.0),
        (43.3, 32.7, -26.0),
        (-28.9, -28.9, -24.1),
        (28.9, -28.9, -24.1),
    ],
    dtype=np.float64,
)

_AWAY_YAW_DEG = 22.0
_DOWN_PITCH_DEG = 18.0
_AWAY_DWELL_SECONDS = 0.8
_DOWN_DWELL_SECONDS = 1.2


class HeadPoseDetector:
    """Estimate coarse head pose from existing face landmarks."""

    def __init__(self) -> None:
        self._away_started_at: float | None = None
        self._down_started_at: float | None = None

    def detect(
        self,
        landmarks: NDArray[np.float32],
        frame_width: int,
        frame_height: int,
    ) -> HeadPoseData | None:
        if frame_width <= 0 or frame_height <= 0:
            return None

        image_points = np.array(
            [
                self._point(landmarks, _NOSE_TIP, frame_width, frame_height),
                self._point(landmarks, _CHIN, frame_width, frame_height),
                self._point(landmarks, _LEFT_EYE_OUTER, frame_width, frame_height),
                self._point(landmarks, _RIGHT_EYE_OUTER, frame_width, frame_height),
                self._point(landmarks, _LEFT_MOUTH, frame_width, frame_height),
                self._point(landmarks, _RIGHT_MOUTH, frame_width, frame_height),
            ],
            dtype=np.float64,
        )

        focal_length = float(frame_width)
        center = (frame_width / 2.0, frame_height / 2.0)
        camera_matrix = np.array(
            [
                [focal_length, 0.0, center[0]],
                [0.0, focal_length, center[1]],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        distortion = np.zeros((4, 1), dtype=np.float64)

        success, rotation_vector, _translation_vector = cv2.solvePnP(
            _MODEL_POINTS,
            image_points,
            camera_matrix,
            distortion,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            return None

        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        rotation_matrix_array = np.asarray(rotation_matrix, dtype=np.float64)
        yaw, pitch, roll = self._euler_degrees(rotation_matrix_array)
        return HeadPoseData(
            yaw=round(yaw, 2),
            pitch=round(pitch, 2),
            roll=round(roll, 2),
        )

    def classify(self, head_pose: HeadPoseData) -> ClassifierResult | None:
        now = time.time()
        yaw_abs = abs(head_pose.yaw)
        pitch = head_pose.pitch
        if yaw_abs >= _AWAY_YAW_DEG:
            if self._away_started_at is None:
                self._away_started_at = now
        else:
            self._away_started_at = None

        if pitch >= _DOWN_PITCH_DEG:
            if self._down_started_at is None:
                self._down_started_at = now
        else:
            self._down_started_at = None

        away_duration = (
            0.0 if self._away_started_at is None else now - self._away_started_at
        )
        down_duration = (
            0.0 if self._down_started_at is None else now - self._down_started_at
        )

        if away_duration >= _AWAY_DWELL_SECONDS:
            confidence = min(
                1.0,
                0.6 + (yaw_abs - _AWAY_YAW_DEG) / 35.0 + away_duration / 4.0,
            )
            return ClassifierResult(label="head_away", confidence=round(confidence, 3))
        if down_duration >= _DOWN_DWELL_SECONDS:
            confidence = min(
                1.0,
                0.6 + (pitch - _DOWN_PITCH_DEG) / 25.0 + down_duration / 4.0,
            )
            return ClassifierResult(label="head_down", confidence=round(confidence, 3))
        return None

    def reset(self) -> None:
        self._away_started_at = None
        self._down_started_at = None

    def _point(
        self,
        landmarks: NDArray[np.float32],
        index: int,
        frame_width: int,
        frame_height: int,
    ) -> tuple[float, float]:
        return (
            float(landmarks[index, 0] * frame_width),
            float(landmarks[index, 1] * frame_height),
        )

    def _euler_degrees(
        self,
        rotation_matrix: NDArray[np.float64],
    ) -> tuple[float, float, float]:
        sy = math.sqrt(
            rotation_matrix[0, 0] * rotation_matrix[0, 0]
            + rotation_matrix[1, 0] * rotation_matrix[1, 0]
        )
        singular = sy < 1e-6

        if not singular:
            pitch = math.degrees(
                math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
            )
            yaw = math.degrees(math.atan2(-rotation_matrix[2, 0], sy))
            roll = math.degrees(
                math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
            )
        else:
            pitch = math.degrees(
                math.atan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
            )
            yaw = math.degrees(math.atan2(-rotation_matrix[2, 0], sy))
            roll = 0.0

        return yaw, pitch, roll
