"""MediaPipe face landmark sources producing normalized landmark arrays.

Two implementations are provided:

MediaPipeFaceLandmarkSource
    Legacy Face Mesh (Solutions API). Returns (478, 3) landmark arrays.
    Still used by blink and gaze detectors which only need landmarks.

FaceLandmarkerTask
    Tasks API backed by face_landmarker_v2_with_blendshapes.task.
    Returns (478, 3) landmark arrays via detect() — same interface as
    the legacy class — AND caches the 52 ARKit blendshape coefficients
    from the last detection in last_blendshapes for use by the
    expression classifier.

Expected input:
    RGB frame as np.ndarray of shape (H, W, 3), dtype uint8.

Expected output (both classes):
    list[np.ndarray] where each array is shape (478, 3) float32,
    values normalized to [0, 1] representing (x, y, z).
    Returns None when no faces are detected.

Usage example::

    source = FaceLandmarkerTask()
    frame_rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    faces = source.detect(frame_rgb)
    if faces:
        landmarks_478x3 = faces[0]
        blendshapes = source.last_blendshapes[0]   # dict[str, float]
    source.close()
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Protocol

import numpy as np
from config.third_party import (
    FaceLandmarkerProtocol,
    LandmarkListProtocol,
    LandmarkProtocol,
    MediaPipeTasksPythonProtocol,
    MediaPipeVisionProtocol,
    load_mediapipe,
    load_mediapipe_tasks_python,
    load_mediapipe_tasks_vision,
)
from numpy.typing import NDArray

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "face_landmarker_v2_with_blendshapes.task")
mp = load_mediapipe()
mp_tasks: MediaPipeTasksPythonProtocol = load_mediapipe_tasks_python()
mp_vision: MediaPipeVisionProtocol = load_mediapipe_tasks_vision()


class FaceLandmarkSource(Protocol):
    """Protocol for face landmark detection (MediaPipe Face Mesh)."""

    def detect(self, frame_rgb: NDArray[np.uint8]) -> list[NDArray[np.float32]] | None:
        """
        Detect faces and return landmark arrays.

        Each array has shape (478, 3) with normalized (x, y, z) coords.
        Returns None if no face is detected.
        """
        ...

    def close(self) -> None:
        """Release detector resources."""
        ...


class BlendshapeLandmarkSource(Protocol):
    """Protocol for face landmark detection with blendshape output."""

    def detect(self, frame_rgb: NDArray[np.uint8]) -> list[NDArray[np.float32]] | None:
        """
        Detect faces and return landmark arrays.

        Each array has shape (478, 3) with normalized (x, y, z) coords.
        Returns None if no face is detected.
        Side-effect: populates last_blendshapes.
        """
        ...

    @property
    def last_blendshapes(self) -> list[dict[str, float]] | None:
        """Blendshapes from the most recent detect() call, or None if no face."""
        ...

    def close(self) -> None:
        """Release detector resources."""
        ...


class MediaPipeFaceLandmarkSource:
    """Wraps MediaPipe Face Mesh (legacy Solutions API) to return per-face landmark arrays.

    Call close() when done to release MediaPipe resources.
    Designed for single-threaded use at ~15 FPS from a main loop.
    """

    _NUM_LANDMARKS = 478  # 468 base + 10 iris landmarks (refine_landmarks=True)

    def __init__(self, refine_landmarks: bool = True) -> None:
        self._refine_landmarks = refine_landmarks
        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def detect(self, frame_rgb: NDArray[np.uint8]) -> list[NDArray[np.float32]] | None:
        """Detect face landmarks in an RGB frame.

        Args:
            frame_rgb: uint8 array of shape (H, W, 3) in RGB color order.

        Returns:
            List of float32 arrays each shaped (478, 3) with normalized
            (x, y, z) coordinates, or None if no faces detected.
        """
        results = self._face_mesh.process(frame_rgb)
        if not results.multi_face_landmarks:
            return None
        return [
            self._landmarks_to_numpy(face_lm)
            for face_lm in results.multi_face_landmarks
        ]

    def close(self) -> None:
        """Release MediaPipe Face Mesh resources."""
        self._face_mesh.close()

    def _landmarks_to_numpy(
        self,
        face_landmarks: LandmarkListProtocol,
    ) -> NDArray[np.float32]:
        arr = np.empty((self._NUM_LANDMARKS, 3), dtype=np.float32)
        for i, lm in enumerate(face_landmarks.landmark):
            arr[i, 0] = lm.x
            arr[i, 1] = lm.y
            arr[i, 2] = lm.z
        return arr


class FaceLandmarkerTask:
    """MediaPipe Tasks FaceLandmarker with blendshape output.

    Wraps mediapipe.tasks.vision.FaceLandmarker using the
    face_landmarker_v2_with_blendshapes.task model which provides
    52 ARKit blendshape coefficients per face in addition to the
    standard 478 face mesh landmarks.

    detect() returns the same landmark format as MediaPipeFaceLandmarkSource
    so it is a drop-in replacement for blink/gaze detectors.

    After each detect() call, last_blendshapes holds a list of
    dict[str, float] — one dict per detected face — mapping blendshape
    category names to their score in [0, 1].

    Requires vision/face_landmarker_v2_with_blendshapes.task to be present.
    """

    def __init__(self) -> None:
        opts = mp_vision.FaceLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=_MODEL_PATH),
            output_face_blendshapes=True,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker: FaceLandmarkerProtocol = (
            mp_vision.FaceLandmarker.create_from_options(opts)
        )
        self._last_blendshapes: list[dict[str, float]] | None = None

    @property
    def last_blendshapes(self) -> list[dict[str, float]] | None:
        """Blendshapes from the most recent detect() call, or None if no face."""
        return self._last_blendshapes

    def detect(self, frame_rgb: NDArray[np.uint8]) -> list[NDArray[np.float32]] | None:
        """Detect face landmarks and blendshapes in an RGB frame.

        Args:
            frame_rgb: uint8 array of shape (H, W, 3) in RGB color order.

        Returns:
            List of float32 arrays each shaped (478, 3) with normalized
            (x, y, z) coordinates, or None if no faces detected.
            Side-effect: populates last_blendshapes.
        """
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self._landmarker.detect(mp_image)

        if not result.face_landmarks:
            self._last_blendshapes = None
            return None

        landmark_arrays = [
            self._landmarks_to_numpy(face_lm)
            for face_lm in result.face_landmarks
        ]

        if result.face_blendshapes:
            self._last_blendshapes = [
                {cat.category_name: cat.score for cat in face_bs}
                for face_bs in result.face_blendshapes
            ]
        else:
            self._last_blendshapes = None

        return landmark_arrays

    def close(self) -> None:
        """Release FaceLandmarker resources."""
        self._landmarker.close()

    def _landmarks_to_numpy(
        self,
        face_landmarks: Sequence[LandmarkProtocol],
    ) -> NDArray[np.float32]:
        arr = np.empty((len(face_landmarks), 3), dtype=np.float32)
        for i, lm in enumerate(face_landmarks):
            arr[i, 0] = lm.x
            arr[i, 1] = lm.y
            arr[i, 2] = lm.z
        return arr


# ----------------------------------------------------------------------
# Maintainer notes
# ----------------------------------------------------------------------
# FaceLandmarkerTask vs MediaPipeFaceLandmarkSource:
#   Use FaceLandmarkerTask in debug_webcam and pipeline_controller so that
#   blendshapes are available for the expression classifier.
#   MediaPipeFaceLandmarkSource is retained as a fallback for environments
#   where the .task model file is unavailable.
#
# Blendshape names (52 total, ARKit standard):
#   browDownLeft, browDownRight, browInnerUp, browOuterUpLeft, browOuterUpRight,
#   cheekPuff, cheekSquintLeft, cheekSquintRight,
#   eyeBlinkLeft, eyeBlinkRight, eyeSquintLeft, eyeSquintRight,
#   eyeWideLeft, eyeWideRight,
#   jawForward, jawLeft, jawOpen, jawRight,
#   mouthClose, mouthFrownLeft, mouthFrownRight, mouthPressLeft, mouthPressRight,
#   mouthSmileLeft, mouthSmileRight, noseSneerLeft, noseSneerRight, ...
#
# Iris landmarks (468-477) are present in the Tasks API output when the model
# supports them; the base face_landmarker model includes iris detection.
