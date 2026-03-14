"""MediaPipe Face Mesh wrapper producing normalized landmark arrays.

Expected input:
    RGB frame as np.ndarray of shape (H, W, 3), dtype uint8.

Expected output:
    list[np.ndarray] where each array is shape (478, 3) float32,
    values normalized to [0, 1] representing (x, y, z).
    Returns None when no faces are detected.

Usage example::

    source = MediaPipeFaceLandmarkSource()
    frame_rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    faces = source.detect(frame_rgb)
    if faces:
        landmarks_478x3 = faces[0]
    source.close()
"""

from __future__ import annotations

import numpy as np
import mediapipe as mp  # type: ignore[import-untyped]


class MediaPipeFaceLandmarkSource:
    """Wraps MediaPipe Face Mesh to return per-face landmark arrays.

    Lazy-initializes the Face Mesh solution on first use.
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

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _landmarks_to_numpy(self, face_landmarks: object) -> np.ndarray:
        """Convert a MediaPipe NormalizedLandmarkList to a (478, 3) float32 array."""
        arr = np.empty((self._NUM_LANDMARKS, 3), dtype=np.float32)
        for i, lm in enumerate(face_landmarks.landmark):  # type: ignore[union-attr]
            arr[i, 0] = lm.x
            arr[i, 1] = lm.y
            arr[i, 2] = lm.z
        return arr


# ----------------------------------------------------------------------
# Maintainer notes
# ----------------------------------------------------------------------
# - refine_landmarks=True enables the 10 iris landmarks (indices 468-477)
#   required by IrisGazeDetector. Disable only on very low-end hardware.
# - max_num_faces=1 keeps latency minimal for the single-person use case.
#   Raise to 2+ if multi-person support is needed.
# - min_detection_confidence / min_tracking_confidence = 0.5 is the
#   MediaPipe default; lower values detect more faces but increase FP rate.
