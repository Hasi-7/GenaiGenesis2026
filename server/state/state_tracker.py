from __future__ import annotations

import base64
import json
import logging
import time
from collections import deque
from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol

from models.types import (
    ClassifierResult,
    CognitiveState,
    CognitiveStateLabel,
    FrameAnalysis,
    StateTransition,
)

if TYPE_CHECKING:
    from input.screenshot_manager import ScreenshotManager
    from openai import OpenAI

logger = logging.getLogger(__name__)


class StateTrackerProtocol(Protocol):
    """Protocol for sliding window state tracking and change detection."""

    def add_frame(self, analysis: FrameAnalysis) -> None:
        """Add a frame analysis to the sliding window."""
        ...

    def get_current_state(self) -> CognitiveState:
        """
        Compute smoothed cognitive state from the sliding window.

        Uses weighted voting across all signals in the window.
        """
        ...

    def detect_transition(self) -> StateTransition | None:
        """
        Compare first half (5s) and second half (5s) of the window.

        Returns StateTransition if:
        - Dominant label differs between halves
        - New confidence > confidence_threshold (0.6)
        - Confidence delta > state_change_threshold (0.3)
        - Both halves are sufficiently populated

        Returns None otherwise.
        """
        ...

    def get_recent_analyses(self, seconds: float = ...) -> list[FrameAnalysis]:
        """Return frame analyses from the last N seconds."""
        ...


# Maps classifier labels → CognitiveStateLabel
_LABEL_TO_STATE: dict[str, CognitiveStateLabel] = {
    # FATIGUED signals
    "fatigued": CognitiveStateLabel.FATIGUED,
    "eyes_closed": CognitiveStateLabel.FATIGUED,
    "elevated": CognitiveStateLabel.FATIGUED,
    "slouching": CognitiveStateLabel.FATIGUED,
    "slouched": CognitiveStateLabel.FATIGUED,
    "leaning": CognitiveStateLabel.FATIGUED,
    "monotone": CognitiveStateLabel.FATIGUED,
    "head_down": CognitiveStateLabel.FATIGUED,
    "yawning": CognitiveStateLabel.FATIGUED,
    # STRESSED signals
    "stressed": CognitiveStateLabel.STRESSED,
    "tense": CognitiveStateLabel.STRESSED,
    # DISTRACTED signals
    "distracted": CognitiveStateLabel.DISTRACTED,
    "head_away": CognitiveStateLabel.DISTRACTED,
    "left": CognitiveStateLabel.DISTRACTED,
    "right": CognitiveStateLabel.DISTRACTED,
    # FOCUSED signals
    "focused": CognitiveStateLabel.FOCUSED,
    "upright": CognitiveStateLabel.FOCUSED,
    "normal": CognitiveStateLabel.FOCUSED,
    "center": CognitiveStateLabel.FOCUSED,
    "calm": CognitiveStateLabel.FOCUSED,
    "relaxed": CognitiveStateLabel.FOCUSED,
    "neutral": CognitiveStateLabel.FOCUSED,
}

_DEFAULT_WINDOW_SECONDS = 10.0
_CONFIDENCE_THRESHOLD = 0.6
_STATE_CHANGE_THRESHOLD = 0.3
_DEBOUNCE_SECONDS = 5.0
_MIN_FRAMES_PER_HALF = 3

_SIGNAL_SCORE_WEIGHTS: dict[str, float] = {
    # Strong negative cues.
    "eyes_closed": 1.7,
    "fatigued": 1.25,
    "head_down": 1.2,
    "yawning": 1.15,
    "stressed": 1.15,
    "head_away": 1.1,
    "distracted": 1.1,
    # Supportive negative cues.
    "elevated": 1.0,
    "slouching": 0.95,
    "slouched": 0.95,
    "leaning": 0.9,
    "monotone": 0.9,
    "tense": 0.9,
    "left": 0.85,
    "right": 0.85,
    # Focus should be the absence of strong negatives, not an easy winner.
    "focused": 0.75,
    "center": 0.55,
    "upright": 0.45,
    "calm": 0.4,
    "relaxed": 0.3,
    "neutral": 0.2,
    "normal": 0.15,
}

_HARD_OVERRIDE_THRESHOLDS: dict[str, tuple[CognitiveStateLabel, float]] = {
    "eyes_closed": (CognitiveStateLabel.FATIGUED, 0.82),
}


def _to_contributing(
    signals: list[ClassifierResult],
) -> list[ClassifierResult]:
    return [ClassifierResult(label=s.label, confidence=s.confidence) for s in signals]


class StateTracker:
    """Sliding window state tracking and change detection."""

    def __init__(self, window_seconds: float = _DEFAULT_WINDOW_SECONDS) -> None:
        self._window_seconds = window_seconds
        self._frames: deque[FrameAnalysis] = deque()
        self._last_transition_time: float = 0.0
        self._last_state: CognitiveState | None = None

    def add_frame(self, analysis: FrameAnalysis) -> None:
        """Add a frame analysis and prune stale entries."""
        self._frames.append(analysis)
        self._prune()

    def get_current_state(self) -> CognitiveState:
        """Compute smoothed cognitive state from the window."""
        signals = _collect_signals(self._frames)
        label, confidence = _weighted_vote(signals)
        state = CognitiveState(
            label=label,
            confidence=confidence,
            contributing_signals=_to_contributing(signals),
            timestamp=time.time(),
        )
        self._last_state = state
        return state

    def detect_transition(self) -> StateTransition | None:
        """Compare first/second half of window for state change."""
        if len(self._frames) == 0:
            return None

        now = time.time()
        if now - self._last_transition_time < _DEBOUNCE_SECONDS:
            return None

        midpoint = now - self._window_seconds / 2.0
        first_half = [f for f in self._frames if f.timestamp < midpoint]
        second_half = [f for f in self._frames if f.timestamp >= midpoint]

        if (
            len(first_half) < _MIN_FRAMES_PER_HALF
            or len(second_half) < _MIN_FRAMES_PER_HALF
        ):
            return None

        old_signals = _collect_signals(first_half)
        new_signals = _collect_signals(second_half)

        if not old_signals or not new_signals:
            return None

        old_label, old_conf = _weighted_vote(old_signals)
        new_label, new_conf = _weighted_vote(new_signals)

        if old_label == new_label:
            return None
        if new_conf < _CONFIDENCE_THRESHOLD:
            return None
        if abs(new_conf - old_conf) < _STATE_CHANGE_THRESHOLD:
            return None

        old_state = CognitiveState(
            label=old_label,
            confidence=old_conf,
            contributing_signals=_to_contributing(old_signals),
            timestamp=midpoint,
        )
        new_state = CognitiveState(
            label=new_label,
            confidence=new_conf,
            contributing_signals=_to_contributing(new_signals),
            timestamp=now,
        )

        self._last_transition_time = now
        self._last_state = new_state
        return StateTransition(
            previous_state=old_state,
            new_state=new_state,
            transition_time=now,
        )

    def get_recent_analyses(self, seconds: float = 5.0) -> list[FrameAnalysis]:
        """Return frame analyses from the last N seconds."""
        cutoff = time.time() - seconds
        return [f for f in self._frames if f.timestamp >= cutoff]

    def _prune(self) -> None:
        """Remove frames older than the window."""
        cutoff = time.time() - self._window_seconds
        while self._frames and self._frames[0].timestamp < cutoff:
            self._frames.popleft()


_LLM_TRACKER_SYSTEM_PROMPT = """\
You are a cognitive state analyst. You receive a webcam image of the user \
alongside sensor data. Your job is to determine if the user's cognitive \
state has meaningfully changed or if they need assistance.

Analyze the image and data, then respond in exactly this JSON format:
{"transition": true/false, "new_state": "focused|fatigued|stressed|"
"distracted", "confidence": 0.0-1.0, "reasoning": "brief explanation"}

Consider:
- Visual cues: facial expression, posture, eye openness, tension
- Sensor signals provided
- Whether the user appears stuck or has been in the same state too long
- Whether they show signs of stress or fatigue not captured by sensors\
"""

_LLM_TRACKER_MODEL = "gpt-4.1-mini"
_LLM_TRACKER_COOLDOWN = 10.0
_LLM_TRACKER_CHECK_INTERVAL = 30.0

_STATE_FROM_STRING: dict[str, CognitiveStateLabel] = {
    "focused": CognitiveStateLabel.FOCUSED,
    "fatigued": CognitiveStateLabel.FATIGUED,
    "stressed": CognitiveStateLabel.STRESSED,
    "distracted": CognitiveStateLabel.DISTRACTED,
}


class LLMStateTracker:
    """State tracker that uses OpenAI vision for transition detection."""

    def __init__(
        self,
        client: OpenAI,
        screenshot_manager: ScreenshotManager,
        window_seconds: float = _DEFAULT_WINDOW_SECONDS,
        cooldown_seconds: float = _LLM_TRACKER_COOLDOWN,
        check_interval_seconds: float = _LLM_TRACKER_CHECK_INTERVAL,
        model: str = _LLM_TRACKER_MODEL,
    ) -> None:
        self._client = client
        self._screenshot_manager = screenshot_manager
        self._window_seconds = window_seconds
        self._cooldown_seconds = cooldown_seconds
        self._check_interval_seconds = check_interval_seconds
        self._model = model

        self._frames: deque[FrameAnalysis] = deque()
        self._last_transition_time: float = 0.0
        self._last_check_time: float = 0.0
        self._last_state: CognitiveState | None = None

    def add_frame(self, analysis: FrameAnalysis) -> None:
        self._frames.append(analysis)
        self._prune()

    def get_current_state(self) -> CognitiveState:
        signals = _collect_signals(self._frames)
        label, confidence = _weighted_vote(signals)
        state = CognitiveState(
            label=label,
            confidence=confidence,
            contributing_signals=_to_contributing(signals),
            timestamp=time.time(),
        )
        self._last_state = state
        return state

    def detect_transition(self) -> StateTransition | None:
        if len(self._frames) == 0:
            return None

        now = time.time()
        if now - self._last_transition_time < self._cooldown_seconds:
            return None

        # Only call the LLM periodically or when local signals shift
        current = self.get_current_state()
        state_changed = (
            self._last_state is not None and current.label != self._last_state.label
        )
        interval_elapsed = now - self._last_check_time >= self._check_interval_seconds

        if not state_changed and not interval_elapsed:
            return None

        jpeg_bytes = self._screenshot_manager.encode_jpeg()
        if not jpeg_bytes:
            return None

        self._last_check_time = now

        reason = "state change" if state_changed else "interval check"
        logger.info(
            "LLMStateTracker: starting detection (%s), current=%s confidence=%.0f%%",
            reason,
            current.label.value,
            current.confidence * 100,
        )

        transition = self._query_llm(jpeg_bytes, current)
        if transition is not None:
            self._last_transition_time = now
            self._last_state = transition.new_state
            logger.info(
                "LLMStateTracker: transition detected %s -> %s (confidence=%.0f%%)",
                transition.previous_state.label.value,
                transition.new_state.label.value,
                transition.new_state.confidence * 100,
            )
        else:
            logger.info("LLMStateTracker: no transition detected")
        return transition

    def get_recent_analyses(self, seconds: float = 5.0) -> list[FrameAnalysis]:
        cutoff = time.time() - seconds
        return [f for f in self._frames if f.timestamp >= cutoff]

    def _prune(self) -> None:
        cutoff = time.time() - self._window_seconds
        while self._frames and self._frames[0].timestamp < cutoff:
            self._frames.popleft()

    def _query_llm(
        self,
        jpeg_bytes: bytes,
        current: CognitiveState,
    ) -> StateTransition | None:
        context = self._build_context(current)
        b64 = base64.b64encode(jpeg_bytes).decode()

        try:
            completion = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": _LLM_TRACKER_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{b64}",
                                    "detail": "low",
                                },
                            },
                            {"type": "text", "text": context},
                        ],
                    },
                ],
                max_tokens=200,
                temperature=0.3,
            )
        except Exception:
            logger.exception("LLMStateTracker: OpenAI call failed")
            return None

        raw = (completion.choices[0].message.content or "").strip()
        return self._parse_response(raw, current)

    def _build_context(self, current: CognitiveState) -> str:
        recent = self.get_recent_analyses(5.0)
        lines = [
            f"Current state: {current.label.value} "
            f"(confidence: {current.confidence:.0%})",
            f"Time since last transition: "
            f"{time.time() - self._last_transition_time:.0f}s",
        ]
        if self._last_state is not None:
            lines.append(f"Previous state: {self._last_state.label.value}")

        # Summarise recent signals
        signals = _collect_signals(recent)
        if signals:
            sig_summary = ", ".join(
                f"{s.label} ({s.confidence:.0%})" for s in signals[:8]
            )
            lines.append(f"Recent signals: {sig_summary}")

        # Blink rate average
        blink_frames = [f for f in recent if f.blink is not None]
        if blink_frames:
            avg_bpm = sum(
                frame.blink.blinks_per_minute
                for frame in blink_frames
                if frame.blink is not None
            ) / len(blink_frames)
            lines.append(f"Avg blink rate: {avg_bpm:.1f} bpm")

        return "\n".join(lines)

    def _parse_response(
        self,
        raw: str,
        current: CognitiveState,
    ) -> StateTransition | None:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("LLMStateTracker: failed to parse JSON: %s", raw)
            return None

        if not data.get("transition", False):
            return None

        new_label_str = data.get("new_state", "").lower()
        new_label = _STATE_FROM_STRING.get(new_label_str)
        if new_label is None:
            return None

        confidence = float(data.get("confidence", 0.5))
        now = time.time()

        previous = self._last_state or current
        new_state = CognitiveState(
            label=new_label,
            confidence=confidence,
            contributing_signals=current.contributing_signals,
            timestamp=now,
        )
        return StateTransition(
            previous_state=previous,
            new_state=new_state,
            transition_time=now,
        )


def _collect_signals(
    frames: Sequence[FrameAnalysis],
) -> list[ClassifierResult]:
    """Extract all non-None classifier results from frames."""
    signals: list[ClassifierResult] = []
    for f in frames:
        if f.blink_label is not None:
            signals.append(f.blink_label)
        elif f.blink is not None:
            if f.blink.blinks_per_minute > 25:
                signals.append(ClassifierResult(label="fatigued", confidence=0.7))
            elif f.blink.blinks_per_minute < 10:
                signals.append(ClassifierResult(label="stressed", confidence=0.6))
            else:
                signals.append(ClassifierResult(label="normal", confidence=0.8))
        if f.eye_state_label is not None:
            signals.append(f.eye_state_label)
        if f.head_pose_label is not None:
            signals.append(f.head_pose_label)
        if f.yawn_label is not None:
            signals.append(f.yawn_label)
        if f.gaze_label is not None:
            signals.append(f.gaze_label)
        elif f.gaze is not None:
            signals.append(ClassifierResult(label=f.gaze.direction, confidence=0.7))
        if f.expression is not None:
            signals.append(f.expression)
        if f.posture is not None:
            signals.append(f.posture)
        if f.speech_tone is not None and f.speech_tone.label != "silent":
            signals.append(f.speech_tone)
    return signals


def _weighted_vote(
    signals: list[ClassifierResult],
) -> tuple[CognitiveStateLabel, float]:
    """Aggregate signals via weighted voting."""
    if not signals:
        return CognitiveStateLabel.UNKNOWN, 0.0

    scores: dict[CognitiveStateLabel, float] = {
        CognitiveStateLabel.FOCUSED: 0.0,
        CognitiveStateLabel.FATIGUED: 0.0,
        CognitiveStateLabel.STRESSED: 0.0,
        CognitiveStateLabel.DISTRACTED: 0.0,
    }
    total_weight = 0.0

    for s in signals:
        override = _HARD_OVERRIDE_THRESHOLDS.get(s.label.lower())
        if override is not None and s.confidence >= override[1]:
            return override[0], min(1.0, s.confidence)

    for s in signals:
        label = s.label.lower()
        state = _LABEL_TO_STATE.get(label)
        if state is not None and state in scores:
            weight = s.confidence * _SIGNAL_SCORE_WEIGHTS.get(label, 1.0)
            scores[state] += weight
            total_weight += weight

    if total_weight == 0.0:
        return CognitiveStateLabel.UNKNOWN, 0.0

    best = max(scores, key=lambda k: scores[k])
    confidence = scores[best] / total_weight
    return best, confidence


if __name__ == "__main__":
    print("=== StateTracker Demo ===\n")
    tracker = StateTracker(window_seconds=10.0)

    # Simulate 10 seconds of "focused" frames
    base_time = time.time() - 10.0
    for i in range(15):
        t = base_time + i * 0.66
        analysis = FrameAnalysis(
            timestamp=t,
            expression=ClassifierResult(label="neutral", confidence=0.85),
            posture=ClassifierResult(label="upright", confidence=0.9),
            speech_tone=ClassifierResult(label="calm", confidence=0.8),
        )
        tracker.add_frame(analysis)

    state = tracker.get_current_state()
    print(f"Current state: {state.label.value} (confidence: {state.confidence:.3f})")
    print(f"Contributing signals: {len(state.contributing_signals)}")

    transition = tracker.detect_transition()
    print(f"Transition detected: {transition is not None}")

    # Now inject "stressed" frames in the second half
    tracker2 = StateTracker(window_seconds=10.0)
    base_time = time.time() - 10.0

    # First half: focused
    for i in range(10):
        t = base_time + i * 0.5
        tracker2.add_frame(
            FrameAnalysis(
                timestamp=t,
                expression=ClassifierResult(label="relaxed", confidence=0.9),
                posture=ClassifierResult(label="upright", confidence=0.85),
            )
        )

    # Second half: stressed
    for i in range(10):
        t = base_time + 5.0 + i * 0.5
        tracker2.add_frame(
            FrameAnalysis(
                timestamp=t,
                expression=ClassifierResult(label="tense", confidence=0.9),
                speech_tone=ClassifierResult(label="stressed", confidence=0.85),
            )
        )

    state2 = tracker2.get_current_state()
    print(
        f"\nMixed window state: {state2.label.value}"
        f" (confidence: {state2.confidence:.3f})"
    )

    transition2 = tracker2.detect_transition()
    if transition2 is not None:
        prev = transition2.previous_state
        new = transition2.new_state
        print(f"Transition: {prev.label.value} -> {new.label.value}")
        print(f"  Old confidence: {prev.confidence:.3f}")
        print(f"  New confidence: {new.confidence:.3f}")
    else:
        print("No transition detected (debounce or thresholds not met)")

    # Test get_recent_analyses
    recent = tracker2.get_recent_analyses(seconds=3.0)
    print(f"\nRecent analyses (last 3s): {len(recent)} frames")

    print("\n=== Done ===")
