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
import numpy as np
import mediapipe as mp  # type: ignore[import-untyped]
from mediapipe.tasks import python as mp_tasks  # type: ignore[import-untyped]
from mediapipe.tasks.python import vision as mp_vision  # type: ignore[import-untyped]

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "face_landmarker_v2_with_blendshapes.task")


class MediaPipeFaceLandmarkSource:
    """Wraps MediaPipe Face Mesh (legacy Solutions API) to return per-face landmark arrays.

    Call close() when done to release MediaPipe resources.
    Designed for single-threaded use at ~15 FPS from a main loop.
    """

    _NUM_LANDMARKS = 478  # 468 base + 10 iris landmarks (refine_landmarks=True)

    def __init__(self, refine_landmarks: bool = True) -> None:
        self._refine_landmarks = refine_landmarks
        self._face_mesh = mp.solutions.face_mesh.FaceMesh(  # type: ignore[attr-defined]
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def detect(self, frame_rgb: np.ndarray) -> list[np.ndarray] | None:
        """Detect face landmarks in an RGB frame.

        Args:
            frame_rgb: uint8 array of shape (H, W, 3) in RGB color order.

        Returns:
            List of float32 arrays each shaped (478, 3) with normalized
            (x, y, z) coordinates, or None if no faces detected.
        """
        results = self._face_mesh.process(frame_rgb)  # type: ignore[union-attr]
        if not results.multi_face_landmarks:  # type: ignore[union-attr]
            return None
        return [
            self._landmarks_to_numpy(face_lm)  # type: ignore[arg-type]
            for face_lm in results.multi_face_landmarks  # type: ignore[union-attr]
        ]

    def close(self) -> None:
        """Release MediaPipe Face Mesh resources."""
        self._face_mesh.close()  # type: ignore[union-attr]

    def _landmarks_to_numpy(self, face_landmarks: object) -> np.ndarray:
        arr = np.empty((self._NUM_LANDMARKS, 3), dtype=np.float32)
        for i, lm in enumerate(face_landmarks.landmark):  # type: ignore[union-attr]
            arr[i, 0] = lm.x  # type: ignore[union-attr]
            arr[i, 1] = lm.y  # type: ignore[union-attr]
            arr[i, 2] = lm.z  # type: ignore[union-attr]
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
        opts = mp_vision.FaceLandmarkerOptions(  # type: ignore[misc]
            base_options=mp_tasks.BaseOptions(model_asset_path=_MODEL_PATH),  # type: ignore[misc]
            output_face_blendshapes=True,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = mp_vision.FaceLandmarker.create_from_options(opts)  # type: ignore[misc]
        self._last_blendshapes: list[dict[str, float]] | None = None

    @property
    def last_blendshapes(self) -> list[dict[str, float]] | None:
        """Blendshapes from the most recent detect() call, or None if no face."""
        return self._last_blendshapes

    def detect(self, frame_rgb: np.ndarray) -> list[np.ndarray] | None:
        """Detect face landmarks and blendshapes in an RGB frame.

        Args:
            frame_rgb: uint8 array of shape (H, W, 3) in RGB color order.

        Returns:
            List of float32 arrays each shaped (478, 3) with normalized
            (x, y, z) coordinates, or None if no faces detected.
            Side-effect: populates last_blendshapes.
        """
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)  # type: ignore[misc]
        result = self._landmarker.detect(mp_image)  # type: ignore[union-attr]

        if not result.face_landmarks:  # type: ignore[union-attr]
            self._last_blendshapes = None
            return None

        landmark_arrays = [
            self._landmarks_to_numpy(face_lm)  # type: ignore[arg-type]
            for face_lm in result.face_landmarks  # type: ignore[union-attr]
        ]

        if result.face_blendshapes:  # type: ignore[union-attr]
            self._last_blendshapes = [
                {cat.category_name: cat.score for cat in face_bs}  # type: ignore[union-attr]
                for face_bs in result.face_blendshapes  # type: ignore[union-attr]
            ]
        else:
            self._last_blendshapes = None

        return landmark_arrays

    def close(self) -> None:
        """Release FaceLandmarker resources."""
        self._landmarker.close()  # type: ignore[union-attr]

    def _landmarks_to_numpy(self, face_landmarks: object) -> np.ndarray:
        lms = list(face_landmarks)  # type: ignore[call-overload, arg-type]
        arr = np.empty((len(lms), 3), dtype=np.float32)  # type: ignore[arg-type]
        for i, lm in enumerate(lms):  # type: ignore[misc]
            arr[i, 0] = lm.x  # type: ignore[union-attr]
            arr[i, 1] = lm.y  # type: ignore[union-attr]
            arr[i, 2] = lm.z  # type: ignore[union-attr]
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
