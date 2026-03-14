from __future__ import annotations

import time

from models.types import ClassifierResult, CognitiveState, FrameAnalysis, LLMResponse


class MirrorUI:
    """Terminal renderer for mirror mode."""

    def __init__(self, refresh_interval_seconds: float = 1.0) -> None:
        self._refresh_interval_seconds = refresh_interval_seconds
        self._last_render_time = 0.0

    def render(
        self,
        state: CognitiveState,
        llm_response: LLMResponse | None = None,
        analysis: FrameAnalysis | None = None,
    ) -> None:
        now = time.monotonic()
        if now - self._last_render_time < self._refresh_interval_seconds:
            return
        self._last_render_time = now

        lines = [
            "",
            "=== CognitiveSense Mirror ===",
            f"state: {state.label.value} ({state.confidence:.2f})",
        ]

        if analysis is None:
            lines.append("frame: waiting for analysis")
        else:
            lines.append(
                f"blink: {self._format_classifier(analysis.blink_label, 'no face')}"
            )
            lines.append(
                f"gaze: {self._format_classifier(analysis.gaze_label, 'no face')}"
            )
            lines.append(
                f"expression: {self._format_classifier(analysis.expression, 'no face')}"
            )
            lines.append(
                f"posture: {self._format_classifier(analysis.posture, 'no body')}"
            )
            lines.append(
                f"speech: {self._format_classifier(analysis.speech_tone, 'disabled')}"
            )

        if llm_response is not None:
            lines.append(f"llm: {llm_response.feedback_text}")

        print("\n".join(lines), flush=True)

    def should_quit(self) -> bool:
        return False

    def destroy(self) -> None:
        return

    def _format_classifier(
        self,
        result: ClassifierResult | None,
        default_text: str,
    ) -> str:
        if result is None:
            return default_text
        return f"{result.label} ({result.confidence:.2f})"
