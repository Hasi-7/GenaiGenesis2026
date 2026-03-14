"""Blendshape-based facial expression classifier.

Uses the 52 ARKit blendshape coefficients produced by MediaPipe's
FaceLandmarkerTask (face_landmarker_v2_with_blendshapes.task) instead
of raw landmark geometry.  Blendshapes are neural-network outputs that
directly encode individual facial muscle movements, giving much better
signal than distance heuristics.

Expected input:
    blendshapes: dict[str, float] — mapping of blendshape category name
    to score in [0, 1], as returned by FaceLandmarkerTask.last_blendshapes[i].

Expected output:
    classify() -> ClassifierResult with label in {"neutral", "tense", "relaxed"}
                  and a confidence in [0, 1].
    raw_scores() -> dict[str, float] mapping each label to its raw score.

Usage example::

    face_src = FaceLandmarkerTask()
    clf = BlendshapeExpressionClassifier()

    faces = face_src.detect(frame_rgb)
    if faces and face_src.last_blendshapes:
        result = clf.classify(face_src.last_blendshapes[0])
        print(result.label, result.confidence)

Single-thread assumption: stateless classifier, safe to call at 15 FPS.
"""

from __future__ import annotations

from typing import Protocol

from models.types import ClassifierResult


class BlendshapeExpressionClassifierProtocol(Protocol):
    """Protocol for facial expression classification from blendshapes."""

    def classify(self, blendshapes: dict[str, float]) -> ClassifierResult:
        """
        Classify facial expression from blendshape coefficients.

        Labels: "neutral", "tense", "relaxed".
        """
        ...


# -----------------------------------------------------------------------
# Blendshape keys used in scoring
# -----------------------------------------------------------------------

# Tense / stressed signals
_BROW_DOWN_L = "browDownLeft"
_BROW_DOWN_R = "browDownRight"
_BROW_INNER_UP = "browInnerUp"        # inner brow raised in worry/concern
_EYE_SQUINT_L = "eyeSquintLeft"
_EYE_SQUINT_R = "eyeSquintRight"
_NOSE_SNEER_L = "noseSneerLeft"
_NOSE_SNEER_R = "noseSneerRight"
_MOUTH_PRESS_L = "mouthPressLeft"
_MOUTH_PRESS_R = "mouthPressRight"
_MOUTH_FROWN_L = "mouthFrownLeft"
_MOUTH_FROWN_R = "mouthFrownRight"

# Relaxed / positive signals
_MOUTH_SMILE_L = "mouthSmileLeft"
_MOUTH_SMILE_R = "mouthSmileRight"
_CHEEK_SQUINT_L = "cheekSquintLeft"   # Duchenne smile marker (genuine)
_CHEEK_SQUINT_R = "cheekSquintRight"
_JAW_OPEN = "jawOpen"                 # open/relaxed mouth


def _get(bs: dict[str, float], key: str) -> float:
    return bs.get(key, 0.0)


class BlendshapeExpressionClassifier:
    """Classifies facial expression from MediaPipe blendshape coefficients.

    Tense signals:
        - browDownLeft / browDownRight   (brow furrow — concentrated/stressed)
        - browInnerUp                    (inner brow raised — worried/concerned)
        - eyeSquintLeft / eyeSquintRight (eye squint — stress or strain)
        - noseSneerLeft / noseSneerRight (nose wrinkle — disgust/stress)
        - mouthPressLeft / mouthPressRight (tight lips — tension)
        - mouthFrownLeft / mouthFrownRight (mouth frown — negative affect)

    Relaxed signals:
        - mouthSmileLeft / mouthSmileRight (smile)
        - cheekSquintLeft / cheekSquintRight (genuine smile Duchenne marker)
        - jawOpen (relaxed open mouth)
    """

    def classify(self, blendshapes: dict[str, float]) -> ClassifierResult:
        """Return the top expression label and its confidence.

        Args:
            blendshapes: dict mapping ARKit blendshape name to score [0, 1].
        """
        scores = self.raw_scores(blendshapes)
        label = max(scores, key=lambda k: scores[k])
        top = scores[label]
        total = sum(scores.values()) + 1e-8
        conf = round(top / total, 3)
        return ClassifierResult(label=label, confidence=conf)

    def raw_scores(self, blendshapes: dict[str, float]) -> dict[str, float]:
        """Compute raw scores for each expression label.

        Returns:
            dict with keys "neutral", "tense", "relaxed" and float values.
        """
        g = _get  # shorthand

        brow_furrow = (g(blendshapes, _BROW_DOWN_L) + g(blendshapes, _BROW_DOWN_R)) / 2.0
        brow_worry  = g(blendshapes, _BROW_INNER_UP)
        squint      = (g(blendshapes, _EYE_SQUINT_L) + g(blendshapes, _EYE_SQUINT_R)) / 2.0
        sneer       = (g(blendshapes, _NOSE_SNEER_L) + g(blendshapes, _NOSE_SNEER_R)) / 2.0
        lip_press   = (g(blendshapes, _MOUTH_PRESS_L) + g(blendshapes, _MOUTH_PRESS_R)) / 2.0
        frown       = (g(blendshapes, _MOUTH_FROWN_L) + g(blendshapes, _MOUTH_FROWN_R)) / 2.0

        smile       = (g(blendshapes, _MOUTH_SMILE_L) + g(blendshapes, _MOUTH_SMILE_R)) / 2.0
        cheek_sq    = (g(blendshapes, _CHEEK_SQUINT_L) + g(blendshapes, _CHEEK_SQUINT_R)) / 2.0
        jaw_open    = g(blendshapes, _JAW_OPEN)

        tense_score   = (brow_furrow * 1.5 + brow_worry * 1.5 + squint * 1.0
                         + sneer * 2.0 + lip_press * 1.5 + frown * 1.5)
        relaxed_score = smile * 3.0 + cheek_sq * 2.5 + jaw_open * 0.5

        # Neutral wins by default; suppressed when tense or relaxed is strong
        neutral_score = max(0.4, 1.0 - tense_score - relaxed_score)

        return {
            "neutral": round(max(0.0, neutral_score), 4),
            "tense":   round(max(0.0, tense_score),   4),
            "relaxed": round(max(0.0, relaxed_score),  4),
        }


# ----------------------------------------------------------------------
# Backwards-compat alias so any code that imported LandmarkExpressionClassifier
# still works without changes during the transition.
# ----------------------------------------------------------------------
LandmarkExpressionClassifier = BlendshapeExpressionClassifier
