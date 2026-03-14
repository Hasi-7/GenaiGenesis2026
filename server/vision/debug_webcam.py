"""Live webcam debug script for the CognitiveSense vision pipeline.

Runs all five vision classifiers on your webcam feed and overlays the
results on screen so you can visually verify each module is working.

Usage (from the server/ directory)::

    uv run python -m vision.debug_webcam

Press 'q' to quit.  No other modules (state, LLM, UI) are required.
"""

from __future__ import annotations

import time

import numpy as np
from config.third_party import load_cv2
from numpy.typing import NDArray

from vision.blink_detector import EarBlinkDetector
from vision.eye_movement_detector import IrisGazeDetector
from vision.facial_expression_classifier import BlendshapeExpressionClassifier
from vision.face_landmarks import FaceLandmarkerTask
from vision.posture_detector import MediaPipePostureDetector
from models.types import BlinkData, ClassifierResult, GazeData, PostureData

cv2 = load_cv2()


# -----------------------------------------------------------------------
# Colors (BGR)
# -----------------------------------------------------------------------

_GREEN  = (60, 210, 60)
_ORANGE = (0, 165, 255)
_RED    = (60, 60, 230)
_WHITE  = (240, 240, 240)
_GREY   = (150, 150, 150)
_DARK   = (20, 20, 20)

_POSITIVE = {"focused", "normal", "upright", "relaxed", "center"}
_WARNING  = {"elevated", "leaning", "distracted"}
_ALERT    = {"stressed", "fatigued", "slouched", "tense"}

_BLINK_WARMUP_FRAMES = 30  # frames before blink rate is meaningful


def _label_color(label: str) -> tuple[int, int, int]:
    if label in _POSITIVE:
        return _GREEN
    if label in _WARNING:
        return _ORANGE
    if label in _ALERT:
        return _RED
    return _WHITE


def _put(
    frame: NDArray[np.uint8],
    text: str,
    x: int,
    y: int,
    color: tuple[int, int, int] = _WHITE,
    scale: float = 0.55,
    thickness: int = 1,
) -> None:
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, _DARK, thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _pill(
    frame: NDArray[np.uint8],
    label: str,
    conf: float,
    x: int,
    y: int,
) -> None:
    """Draw a colored rounded-rectangle chip showing label + confidence %."""
    color = _label_color(label)
    text = f"{label.upper()}  {int(conf * 100)}%"
    scale = 0.52
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    pad_x, pad_y = 8, 5
    x1, y1 = x, y - th - pad_y
    x2, y2 = x + tw + pad_x * 2, y + baseline + pad_y
    # filled background
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
    # text in dark
    cv2.putText(frame, text, (x + pad_x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, _DARK, thickness + 1, cv2.LINE_AA)
    cv2.putText(frame, text, (x + pad_x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def _ear_bar(
    frame: NDArray[np.uint8],
    ear: float,
    x: int,
    y: int,
    label: str,
) -> None:
    """Draw a small horizontal bar showing EAR value (0.0–0.4 range)."""
    bar_w, bar_h = 80, 8
    ratio = min(1.0, ear / 0.4)
    filled = int(bar_w * ratio)
    color = _GREEN if ear > 0.25 else (_ORANGE if ear > 0.21 else _RED)
    cv2.rectangle(frame, (x, y - bar_h), (x + bar_w, y), (60, 60, 60), -1)
    if filled > 0:
        cv2.rectangle(frame, (x, y - bar_h), (x + filled, y), color, -1)
    _put(frame, label, x + bar_w + 4, y - 1, _GREY, 0.38)


def _overlay_panel(
    frame: NDArray[np.uint8],
    blink: BlinkData | None,
    blink_cls: ClassifierResult | None,
    gaze: GazeData | None,
    gaze_cls: ClassifierResult | None,
    expr_cls: ClassifierResult | None,
    posture: PostureData | None,
    posture_cls: ClassifierResult | None,
    fps: float,
    frame_count: int,
) -> None:
    h, w = frame.shape[:2]

    # ── left column: raw data ──────────────────────────────────────────
    # dark sidebar
    sidebar_w = 260
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (sidebar_w, h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)

    x = 12
    y = 28
    step = 22

    # FPS (top right of whole frame)
    fps_text = f"FPS {fps:.1f}"
    (tw, _), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    _put(frame, fps_text, w - tw - 10, 20, _GREY, 0.45)

    # ── SECTION: BLINK ─────────────────────────────────────────────────
    _put(frame, "BLINK", x, y, _GREY, 0.42)
    y += step

    if blink:
        # EAR bars for each eye
        _ear_bar(frame, blink.ear_right, x, y, f"R {blink.ear_right:.2f}")
        y += 14
        _ear_bar(frame, blink.ear_left, x, y, f"L {blink.ear_left:.2f}")
        y += 16

        # BPM line
        if frame_count < _BLINK_WARMUP_FRAMES:
            _put(frame, "warming up...", x, y, _GREY, 0.40)
        else:
            bpm_color = _label_color(blink_cls.label) if blink_cls else _WHITE
            _put(frame, f"{blink.blinks_per_minute:.1f} bpm", x, y, bpm_color, 0.48)
        y += step

        # pill
        if blink_cls and frame_count >= _BLINK_WARMUP_FRAMES:
            _pill(frame, blink_cls.label, blink_cls.confidence, x, y)
        y += step + 4
    else:
        _put(frame, "no face detected", x, y, _ORANGE, 0.42)
        y += step * 3

    # divider
    cv2.line(frame, (x, y), (sidebar_w - x, y), (60, 60, 60), 1)
    y += 10

    # ── SECTION: GAZE ──────────────────────────────────────────────────
    _put(frame, "GAZE", x, y, _GREY, 0.42)
    y += step

    if gaze:
        _put(frame, f"H {gaze.horizontal_ratio:+.2f}  V {gaze.vertical_ratio:+.2f}", x, y, _WHITE, 0.44)
        y += step
        _put(frame, f"direction: {gaze.direction}", x, y, _WHITE, 0.44)
        y += step
        if gaze_cls:
            _pill(frame, gaze_cls.label, gaze_cls.confidence, x, y)
        y += step + 4
    else:
        _put(frame, "no face detected", x, y, _ORANGE, 0.42)
        y += step * 3

    cv2.line(frame, (x, y), (sidebar_w - x, y), (60, 60, 60), 1)
    y += 10

    # ── SECTION: EXPRESSION ────────────────────────────────────────────
    _put(frame, "EXPRESSION", x, y, _GREY, 0.42)
    y += step

    if expr_cls:
        _pill(frame, expr_cls.label, expr_cls.confidence, x, y)
        y += step + 4
    else:
        _put(frame, "no face detected", x, y, _ORANGE, 0.42)
        y += step

    cv2.line(frame, (x, y), (sidebar_w - x, y), (60, 60, 60), 1)
    y += 10

    # ── SECTION: POSTURE ───────────────────────────────────────────────
    _put(frame, "POSTURE", x, y, _GREY, 0.42)
    y += step

    if posture:
        s_color = _RED if posture.is_slouching else _GREEN
        _put(frame, f"shoulder tilt: {posture.shoulder_angle:.1f}deg", x, y, s_color, 0.44)
        y += step
        t_color = _ORANGE if posture.head_tilt > 20 else _GREEN
        _put(frame, f"head tilt:     {posture.head_tilt:.1f}deg", x, y, t_color, 0.44)
        y += step
        if posture_cls:
            _pill(frame, posture_cls.label, posture_cls.confidence, x, y)
        y += step + 4
    else:
        _put(frame, "no body detected", x, y, _ORANGE, 0.42)
        y += step

    # hint at bottom
    _put(frame, "q = quit", x, h - 12, _GREY, 0.38)


def run() -> None:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (index 0).")

    face_src = FaceLandmarkerTask()
    blink_det = EarBlinkDetector()
    gaze_det = IrisGazeDetector()
    expr_clf = BlendshapeExpressionClassifier()
    posture_det = MediaPipePostureDetector()

    prev_time = time.time()
    frame_count = 0

    cv2.namedWindow("CognitiveSense — vision debug", cv2.WINDOW_NORMAL)

    try:
        while True:
            ret, bgr = cap.read()
            if not ret or bgr is None:
                break

            frame_count += 1

            # Flip for mirror-view
            bgr = cv2.flip(bgr, 1)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            # --- Face-dependent modules ---
            blink: BlinkData | None = None
            blink_cls: ClassifierResult | None = None
            gaze: GazeData | None = None
            gaze_cls: ClassifierResult | None = None
            expr_cls: ClassifierResult | None = None

            faces = face_src.detect(rgb)
            if faces:
                lm = faces[0]  # use first face only
                blink = blink_det.detect(lm)
                blink_cls = blink_det.classify(blink)
                gaze = gaze_det.detect(lm)
                gaze_cls = gaze_det.classify(gaze)
                if face_src.last_blendshapes:
                    expr_cls = expr_clf.classify(face_src.last_blendshapes[0])

            # --- Posture (uses full frame, not landmarks) ---
            posture: PostureData | None = posture_det.detect(rgb)
            posture_cls: ClassifierResult | None = (
                posture_det.classify(posture) if posture else None
            )

            # --- FPS ---
            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            # --- Overlay ---
            _overlay_panel(
                bgr,
                blink, blink_cls,
                gaze, gaze_cls,
                expr_cls,
                posture, posture_cls,
                fps,
                frame_count,
            )

            cv2.imshow("CognitiveSense — vision debug", bgr)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        face_src.close()
        posture_det.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
