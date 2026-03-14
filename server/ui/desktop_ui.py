"""Desktop UI renderer — OpenCV overlay showing cognitive state and debug info."""

from __future__ import annotations

import textwrap
import time

import numpy as np
from config.third_party import load_cv2
from models.types import (
    CognitiveState,
    CognitiveStateLabel,
    FrameAnalysis,
    LLMResponse,
)
from numpy.typing import NDArray

cv2 = load_cv2()

# ---------------------------------------------------------------------------
# Colors (BGR)
# ---------------------------------------------------------------------------

_GREEN = (60, 210, 60)
_ORANGE = (0, 165, 255)
_RED = (60, 60, 230)
_WHITE = (240, 240, 240)
_GREY = (150, 150, 150)
_DARK = (20, 20, 20)
_YELLOW = (0, 220, 220)

_STATE_COLORS: dict[CognitiveStateLabel, tuple[int, int, int]] = {
    CognitiveStateLabel.FOCUSED: _GREEN,
    CognitiveStateLabel.FATIGUED: _ORANGE,
    CognitiveStateLabel.STRESSED: _RED,
    CognitiveStateLabel.DISTRACTED: _YELLOW,
    CognitiveStateLabel.UNKNOWN: _GREY,
}

_POSITIVE = {"focused", "normal", "upright", "relaxed", "center", "calm"}
_WARNING = {"elevated", "leaning", "distracted", "monotone"}
_ALERT = {"stressed", "fatigued", "slouched", "tense", "silent"}


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
    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        _DARK,
        thickness + 2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def _pill(
    frame: NDArray[np.uint8],
    label: str,
    conf: float,
    x: int,
    y: int,
) -> None:
    color = _label_color(label)
    text = f"{label.upper()}  {int(conf * 100)}%"
    scale, thickness = 0.52, 1
    (tw, th), baseline = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        thickness,
    )
    pad_x, pad_y = 8, 5
    x1, y1 = x, y - th - pad_y
    x2, y2 = x + tw + pad_x * 2, y + baseline + pad_y
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
    cv2.putText(
        frame,
        text,
        (x + pad_x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        _DARK,
        thickness + 1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        text,
        (x + pad_x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def _ear_bar(
    frame: NDArray[np.uint8],
    ear: float,
    x: int,
    y: int,
    label: str,
) -> None:
    bar_w, bar_h = 80, 8
    ratio = min(1.0, ear / 0.4)
    filled = int(bar_w * ratio)
    color = _GREEN if ear > 0.25 else (_ORANGE if ear > 0.21 else _RED)
    cv2.rectangle(frame, (x, y - bar_h), (x + bar_w, y), (60, 60, 60), -1)
    if filled > 0:
        cv2.rectangle(frame, (x, y - bar_h), (x + filled, y), color, -1)
    _put(frame, label, x + bar_w + 4, y - 1, _GREY, 0.38)


class DesktopUI:
    """OpenCV-based desktop renderer.

    Displays the live camera feed with a sidebar of signal-level debug data,
    a top-bar showing the current cognitive state, and a bottom strip for
    the latest LLM feedback message.
    """

    _WINDOW = "CognitiveSense"
    _SIDEBAR_W = 270
    _BOTTOM_H = 70
    _TOP_H = 44

    def __init__(self) -> None:
        cv2.namedWindow(self._WINDOW, cv2.WINDOW_NORMAL)
        self._frame_count = 0
        self._prev_time = time.time()
        self._fps = 0.0
        self._quit = False

    # ------------------------------------------------------------------
    # Protocol interface
    # ------------------------------------------------------------------

    def render(
        self,
        frame: NDArray[np.uint8],
        state: CognitiveState,
        llm_response: LLMResponse | None = None,
        analysis: FrameAnalysis | None = None,
    ) -> None:
        """Draw overlays and show the frame."""
        self._frame_count += 1
        now = time.time()
        self._fps = 1.0 / max(now - self._prev_time, 1e-6)
        self._prev_time = now

        # Work on a copy so we don't mutate the original
        canvas = frame.copy()
        canvas = cv2.flip(canvas, 1)

        self._draw_sidebar(canvas, analysis)
        self._draw_top_bar(canvas, state)
        self._draw_bottom_bar(canvas, llm_response)

        cv2.imshow(self._WINDOW, canvas)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            self._quit = True

    def should_quit(self) -> bool:
        return self._quit

    def destroy(self) -> None:
        cv2.destroyAllWindows()

    # ------------------------------------------------------------------
    # Private drawing helpers
    # ------------------------------------------------------------------

    def _draw_top_bar(
        self,
        frame: NDArray[np.uint8],
        state: CognitiveState,
    ) -> None:
        _, w = frame.shape[:2]
        color = _STATE_COLORS.get(state.label, _GREY)
        overlay = frame.copy()
        cv2.rectangle(overlay, (self._SIDEBAR_W, 0), (w, self._TOP_H), color, -1)
        cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

        # State label + confidence
        label_text = f"{state.label.value.upper()}  {int(state.confidence * 100)}%"
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
        tx = self._SIDEBAR_W + (w - self._SIDEBAR_W - tw) // 2
        ty = (self._TOP_H + th) // 2
        cv2.putText(
            frame,
            label_text,
            (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            _DARK,
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            label_text,
            (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # FPS top-right corner
        fps_text = f"FPS {self._fps:.1f}"
        (fw, _), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        _put(frame, fps_text, w - fw - 8, self._TOP_H - 6, _GREY, 0.42)

    def _draw_bottom_bar(
        self,
        frame: NDArray[np.uint8],
        llm_response: LLMResponse | None,
    ) -> None:
        h, w = frame.shape[:2]
        if llm_response is None:
            return

        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (self._SIDEBAR_W, h - self._BOTTOM_H),
            (w, h),
            (20, 20, 20),
            -1,
        )
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # Wrap text to fit the available width
        avail_w = w - self._SIDEBAR_W - 16
        char_w = max(1, avail_w // 8)  # rough chars per line at scale 0.45
        lines = textwrap.wrap(llm_response.feedback_text, width=char_w)[:3]

        y = h - self._BOTTOM_H + 18
        for line in lines:
            _put(frame, line, self._SIDEBAR_W + 8, y, _WHITE, 0.42)
            y += 18

    def _draw_sidebar(
        self,
        frame: NDArray[np.uint8],
        analysis: FrameAnalysis | None,
    ) -> None:
        h = frame.shape[0]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (self._SIDEBAR_W, h), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

        x = 12
        y = self._TOP_H + 18
        step = 22

        if analysis is None:
            _put(frame, "waiting for data...", x, y, _GREY, 0.44)
            _put(frame, "q = quit", x, h - 12, _GREY, 0.38)
            return

        # ── BLINK ──────────────────────────────────────────────────────
        _put(frame, "BLINK", x, y, _GREY, 0.42)
        y += step

        blink = analysis.blink
        if blink:
            _ear_bar(frame, blink.ear_right, x, y, f"R {blink.ear_right:.2f}")
            y += 14
            _ear_bar(frame, blink.ear_left, x, y, f"L {blink.ear_left:.2f}")
            y += 16
            if self._frame_count < 30:
                _put(frame, "warming up...", x, y, _GREY, 0.40)
            else:
                _put(frame, f"{blink.blinks_per_minute:.1f} bpm", x, y, _WHITE, 0.48)
            y += step
            if analysis.blink_label:
                _pill(
                    frame,
                    analysis.blink_label.label,
                    analysis.blink_label.confidence,
                    x,
                    y,
                )
                y += step + 4
        else:
            _put(frame, "no face", x, y, _ORANGE, 0.42)
            y += step * 2

        cv2.line(frame, (x, y), (self._SIDEBAR_W - x, y), (60, 60, 60), 1)
        y += 10

        # ── GAZE ───────────────────────────────────────────────────────
        _put(frame, "GAZE", x, y, _GREY, 0.42)
        y += step

        gaze = analysis.gaze
        if gaze:
            _put(
                frame,
                f"H {gaze.horizontal_ratio:+.2f}  V {gaze.vertical_ratio:+.2f}",
                x,
                y,
                _WHITE,
                0.44,
            )
            y += step
            _put(
                frame,
                f"dir: {gaze.direction}",
                x,
                y,
                _label_color(
                    "center" if gaze.direction == "center" else "distracted"
                ),
                0.44,
            )
            y += step
            if analysis.gaze_label:
                _pill(
                    frame,
                    analysis.gaze_label.label,
                    analysis.gaze_label.confidence,
                    x,
                    y,
                )
                y += step + 4
        else:
            _put(frame, "no face", x, y, _ORANGE, 0.42)
            y += step * 2

        cv2.line(frame, (x, y), (self._SIDEBAR_W - x, y), (60, 60, 60), 1)
        y += 10

        # ── EXPRESSION ─────────────────────────────────────────────────
        _put(frame, "EXPRESSION", x, y, _GREY, 0.42)
        y += step

        if analysis.expression:
            _pill(
                frame,
                analysis.expression.label,
                analysis.expression.confidence,
                x,
                y,
            )
            y += step + 4
        else:
            _put(frame, "no face", x, y, _ORANGE, 0.42)
            y += step

        cv2.line(frame, (x, y), (self._SIDEBAR_W - x, y), (60, 60, 60), 1)
        y += 10

        # ── POSTURE ────────────────────────────────────────────────────
        _put(frame, "POSTURE", x, y, _GREY, 0.42)
        y += step

        if analysis.posture:
            _pill(frame, analysis.posture.label, analysis.posture.confidence, x, y)
            y += step + 4
        else:
            _put(frame, "no body", x, y, _ORANGE, 0.42)
            y += step

        cv2.line(frame, (x, y), (self._SIDEBAR_W - x, y), (60, 60, 60), 1)
        y += 10

        # ── SPEECH TONE ────────────────────────────────────────────────
        _put(frame, "SPEECH TONE", x, y, _GREY, 0.42)
        y += step

        if analysis.speech_tone:
            _pill(
                frame,
                analysis.speech_tone.label,
                analysis.speech_tone.confidence,
                x,
                y,
            )
            y += step + 4
        else:
            _put(frame, "no audio", x, y, _GREY, 0.42)
            y += step

        _put(frame, "q = quit", x, h - 12, _GREY, 0.38)
