from __future__ import annotations

import time
from collections import deque
from collections.abc import Sequence
from typing import Protocol

from server.models.types import (
    ClassifierResult,
    CognitiveState,
    CognitiveStateLabel,
    FrameAnalysis,
    StateTransition,
)


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
    "slouching": CognitiveStateLabel.FATIGUED,
    "monotone": CognitiveStateLabel.FATIGUED,
    # STRESSED signals
    "stressed": CognitiveStateLabel.STRESSED,
    "tense": CognitiveStateLabel.STRESSED,
    # DISTRACTED signals
    "distracted": CognitiveStateLabel.DISTRACTED,
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


def _to_contributing(
    signals: list[ClassifierResult],
) -> list[ClassifierResult]:
    return [
        ClassifierResult(label=s.label, confidence=s.confidence)
        for s in signals
    ]


class StateTracker:
    """Sliding window state tracking and change detection."""

    def __init__(
        self, window_seconds: float = _DEFAULT_WINDOW_SECONDS
    ) -> None:
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
        first_half = [
            f for f in self._frames if f.timestamp < midpoint
        ]
        second_half = [
            f for f in self._frames if f.timestamp >= midpoint
        ]

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

    def get_recent_analyses(
        self, seconds: float = 5.0
    ) -> list[FrameAnalysis]:
        """Return frame analyses from the last N seconds."""
        cutoff = time.time() - seconds
        return [f for f in self._frames if f.timestamp >= cutoff]

    def _prune(self) -> None:
        """Remove frames older than the window."""
        cutoff = time.time() - self._window_seconds
        while self._frames and self._frames[0].timestamp < cutoff:
            self._frames.popleft()


def _collect_signals(
    frames: Sequence[FrameAnalysis],
) -> list[ClassifierResult]:
    """Extract all non-None classifier results from frames."""
    signals: list[ClassifierResult] = []
    for f in frames:
        if f.blink is not None:
            if f.blink.blinks_per_minute > 25:
                signals.append(
                    ClassifierResult(label="fatigued", confidence=0.7)
                )
            elif f.blink.blinks_per_minute < 10:
                signals.append(
                    ClassifierResult(label="stressed", confidence=0.6)
                )
            else:
                signals.append(
                    ClassifierResult(label="normal", confidence=0.8)
                )
        if f.gaze is not None:
            signals.append(
                ClassifierResult(
                    label=f.gaze.direction, confidence=0.7
                )
            )
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
        state = _LABEL_TO_STATE.get(s.label.lower())
        if state is not None and state in scores:
            scores[state] += s.confidence
            total_weight += s.confidence

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
            expression=ClassifierResult(
                label="neutral", confidence=0.85
            ),
            posture=ClassifierResult(
                label="upright", confidence=0.9
            ),
            speech_tone=ClassifierResult(
                label="calm", confidence=0.8
            ),
        )
        tracker.add_frame(analysis)

    state = tracker.get_current_state()
    print(
        f"Current state: {state.label.value}"
        f" (confidence: {state.confidence:.3f})"
    )
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
                expression=ClassifierResult(
                    label="relaxed", confidence=0.9
                ),
                posture=ClassifierResult(
                    label="upright", confidence=0.85
                ),
            )
        )

    # Second half: stressed
    for i in range(10):
        t = base_time + 5.0 + i * 0.5
        tracker2.add_frame(
            FrameAnalysis(
                timestamp=t,
                expression=ClassifierResult(
                    label="tense", confidence=0.9
                ),
                speech_tone=ClassifierResult(
                    label="stressed", confidence=0.85
                ),
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
        print(
            f"Transition: {prev.label.value}"
            f" -> {new.label.value}"
        )
        print(f"  Old confidence: {prev.confidence:.3f}")
        print(f"  New confidence: {new.confidence:.3f}")
    else:
        print("No transition detected (debounce or thresholds not met)")

    # Test get_recent_analyses
    recent = tracker2.get_recent_analyses(seconds=3.0)
    print(f"\nRecent analyses (last 3s): {len(recent)} frames")

    print("\n=== Done ===")
