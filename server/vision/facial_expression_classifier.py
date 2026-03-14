"""Landmark-geometry heuristic facial expression classifier.

Expected input:
    landmarks: np.ndarray of shape (478, 3) float32 — normalized face mesh
    from MediaPipeFaceLandmarkSource for a single face.

Expected output:
    classify() -> ClassifierResult with label in {"neutral", "tense", "relaxed"}
                  and a confidence in [0, 1].
    raw_scores() -> dict[str, float] mapping each label to its raw heuristic score.

Usage example::

    clf = LandmarkExpressionClassifier()
    result = clf.classify(landmarks_478x3)
    scores = clf.raw_scores(landmarks_478x3)
    print(result.label, result.confidence, scores)

TODO: Replace heuristic geometry with a FER CNN (e.g. trained on FER2013).
      See the "FER CNN integration" section at the bottom of this file.

Single-thread assumption: stateless — safe to call from any context, but
designed for the single-threaded 15 FPS main loop.
"""

from __future__ import annotations

import numpy as np

from models.types import ClassifierResult


# -----------------------------------------------------------------------
# MediaPipe Face Mesh landmark indices used for expression heuristics
# -----------------------------------------------------------------------

# Mouth openness: upper lip (13), lower lip (14)
_UPPER_LIP = 13
_LOWER_LIP = 14

# Jaw open: chin (152) vs nose tip (4)
_CHIN = 152
_NOSE_TIP = 4

# Brow landmarks: left inner brow (65), left outer brow (105)
#                right inner brow (295), right outer brow (334)
_LEFT_BROW_INNER = 65
_LEFT_BROW_OUTER = 105
_RIGHT_BROW_INNER = 295
_RIGHT_BROW_OUTER = 334

# Eye landmarks for squint detection: left upper (386), left lower (374)
#                                      right upper (159), right lower (145)
_LEFT_EYE_TOP = 386
_LEFT_EYE_BOTTOM = 374
_RIGHT_EYE_TOP = 159
_RIGHT_EYE_BOTTOM = 145

# Reference distance: inter-pupil / inter-eye-corner for normalization
# Using outer eye corners: left=263, right=33
_LEFT_EYE_OUTER = 263
_RIGHT_EYE_OUTER = 33


def _norm_dist(a: np.ndarray, b: np.ndarray, scale: float) -> float:
    """Euclidean distance between 2-D points, divided by scale."""
    d = float(np.linalg.norm(a[:2] - b[:2]))
    return d / scale if scale > 1e-6 else 0.0


class LandmarkExpressionClassifier:
    """Classifies facial expressions using geometric heuristics.

    Three heuristic features are computed on each call to classify():
      1. mouth_openness – normalised lip separation.
      2. brow_raise     – normalised brow-to-eye distance (raised brow = tense/surprised).
      3. eye_squint     – normalised eye aperture (squint = tense/stressed).

    Labels:
        "tense"   – furrowed brows and/or squinted eyes
        "relaxed" – open mouth (slight smile / yawn) with raised brows
        "neutral" – default when no strong signal

    # TODO: FER CNN integration
    # Replace this class body with a call to a FER model for higher accuracy.
    # Recommended model: FER2013-trained MobileNet or ResNet-lite.
    # Install: pip install fer  (or) pip install deepface
    # Example swap-in:
    #
    #   from fer import FER
    #
    #   class LandmarkExpressionClassifier:
    #       def __init__(self) -> None:
    #           self._model = FER(mtcnn=False)  # MTCNN disabled; we crop via landmarks
    #
    #       def classify(self, landmarks: np.ndarray) -> ClassifierResult:
    #           # Crop face ROI from frame using landmarks bounding box
    #           # Pass cropped face_bgr to self._model.detect_emotions(face_bgr)
    #           # Map FER2013 labels (angry→tense, happy→relaxed, neutral→neutral, ...)
    #           ...
    #
    # The raw_scores() method can return model softmax outputs instead.
    """

    def classify(self, landmarks: np.ndarray) -> ClassifierResult:
        """Return the top expression label and its confidence.

        Args:
            landmarks: float32 array shape (478, 3) for one face.
        """
        scores = self.raw_scores(landmarks)
        label = max(scores, key=lambda k: scores[k])
        top_score = scores[label]
        # Softmax-style normalisation for confidence
        total = sum(scores.values()) + 1e-8
        conf = top_score / total
        return ClassifierResult(label=label, confidence=round(conf, 3))

    def raw_scores(self, landmarks: np.ndarray) -> dict[str, float]:
        """Compute raw heuristic scores for each expression label.

        Returns:
            dict with keys "neutral", "tense", "relaxed" and float values.
            Higher score = stronger evidence for that label.
        """
        scale = self._face_scale(landmarks)

        mouth = _norm_dist(landmarks[_UPPER_LIP], landmarks[_LOWER_LIP], scale)
        brow = self._brow_raise(landmarks, scale)
        squint = self._eye_squint(landmarks, scale)

        # Heuristic scoring
        tense_score = squint * 2.0 + max(0.0, 0.05 - brow) * 10.0
        relaxed_score = mouth * 3.0 + brow * 2.0
        neutral_score = 1.0 - 0.5 * abs(tense_score - relaxed_score) / (tense_score + relaxed_score + 1e-6)

        return {
            "neutral": round(max(0.0, neutral_score), 4),
            "tense": round(max(0.0, tense_score), 4),
            "relaxed": round(max(0.0, relaxed_score), 4),
        }

    # ------------------------------------------------------------------
    # Private feature extractors
    # ------------------------------------------------------------------

    def _face_scale(self, landmarks: np.ndarray) -> float:
        """Inter-eye distance as a normalisation scale factor."""
        return float(np.linalg.norm(
            landmarks[_LEFT_EYE_OUTER, :2] - landmarks[_RIGHT_EYE_OUTER, :2]
        ))

    def _brow_raise(self, landmarks: np.ndarray, scale: float) -> float:
        """Mean normalised distance from brow to eye outer corner (higher = raised)."""
        left = _norm_dist(landmarks[_LEFT_BROW_INNER], landmarks[_LEFT_EYE_TOP], scale)
        right = _norm_dist(landmarks[_RIGHT_BROW_INNER], landmarks[_RIGHT_EYE_TOP], scale)
        return (left + right) / 2.0

    def _eye_squint(self, landmarks: np.ndarray, scale: float) -> float:
        """Inverse of eye aperture (higher = more squinted)."""
        left_aperture = _norm_dist(landmarks[_LEFT_EYE_TOP], landmarks[_LEFT_EYE_BOTTOM], scale)
        right_aperture = _norm_dist(landmarks[_RIGHT_EYE_TOP], landmarks[_RIGHT_EYE_BOTTOM], scale)
        avg_aperture = (left_aperture + right_aperture) / 2.0
        # Typical open-eye aperture is ~0.08-0.12 of face scale; invert
        return max(0.0, 0.10 - avg_aperture) * 10.0


# ----------------------------------------------------------------------
# Maintainer notes
# ----------------------------------------------------------------------
# FER CNN integration (see TODO above):
#   The heuristic approach is intentionally simple. FER2013 models achieve
#   ~65-70% accuracy on the 7-class dataset; mapping to our 3-class schema
#   (neutral, tense, relaxed) typically yields >80% accuracy for in-the-wild
#   webcam images. Swap the class body when:
#     - Ground-truth labels are available for your user population.
#     - pip install fer / deepface is acceptable (adds ~200 MB).
#   Keep the same classify() / raw_scores() signatures so no other modules
#   need to change.
#
# Mouth openness threshold:
#   Landmark 13 (upper lip) to 14 (lower lip) normalised by inter-eye
#   distance. Values > 0.04 typically indicate an open mouth (yawn, speech,
#   relaxed expression). Tune by sampling 100 "relaxed" frames.
#
# Brow-raise threshold:
#   Brow-to-eye distance < 0.05 (normalised) suggests furrowed brows
#   commonly associated with concentration/stress. > 0.08 suggests surprise
#   or relaxation.
#
# Eye squint:
#   Aperture < 0.08 indicates squinting. The 0.10 reference is approximate;
#   varies with subject anatomy. Consider per-session calibration.
