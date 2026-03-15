"""Desktop UI renderer — OpenCV overlay showing cognitive state and debug info."""

from __future__ import annotations

import time

import numpy as np
from config.third_party import load_cv2
from models.types import CognitiveState, FrameAnalysis, LLMResponse
from numpy.typing import NDArray
from ui.frame_overlay import annotate_frame

cv2 = load_cv2()


class DesktopUI:
    """OpenCV-based desktop renderer."""

    _WINDOW = "CognitiveSense"

    def __init__(self) -> None:
        cv2.namedWindow(self._WINDOW, cv2.WINDOW_NORMAL)
        self._prev_time = time.time()
        self._fps = 0.0
        self._quit = False

    def render(
        self,
        frame: NDArray[np.uint8],
        state: CognitiveState,
        llm_response: LLMResponse | None = None,
        analysis: FrameAnalysis | None = None,
    ) -> None:
        now = time.time()
        self._fps = 1.0 / max(now - self._prev_time, 1e-6)
        self._prev_time = now

        canvas = cv2.flip(frame, 1)
        canvas = annotate_frame(
            canvas,
            state,
            llm_response,
            analysis,
            fps=self._fps,
        )

        cv2.imshow(self._WINDOW, canvas)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            self._quit = True

    def should_quit(self) -> bool:
        return self._quit

    def destroy(self) -> None:
        cv2.destroyAllWindows()
