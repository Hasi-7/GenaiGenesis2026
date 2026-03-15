from __future__ import annotations

import base64
import json
import logging
import threading
import time
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
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

    def start(self) -> None:
        """Start background processing (if any)."""
        ...

    def stop(self) -> None:
        """Stop background processing (if any)."""
        ...


# Maps classifier labels → CognitiveStateLabel
_LABEL_TO_STATE: dict[str, CognitiveStateLabel] = {
    # FATIGUED signals
    "fatigued": CognitiveStateLabel.FATIGUED,
    "elevated": CognitiveStateLabel.FATIGUED,
    "slouching": CognitiveStateLabel.FATIGUED,
    "slouched": CognitiveStateLabel.FATIGUED,
    "leaning": CognitiveStateLabel.FATIGUED,
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

    def start(self) -> None:
        """No-op for rule-based tracker."""

    def stop(self) -> None:
        """No-op for rule-based tracker."""

    def _prune(self) -> None:
        """Remove frames older than the window."""
        cutoff = time.time() - self._window_seconds
        while self._frames and self._frames[0].timestamp < cutoff:
            self._frames.popleft()


@dataclass(slots=True)
class _EmotionalSnapshot:
    """A timestamped emotional state summary produced by the LLM."""

    timestamp: float
    iso_time: str
    summary: str
    state: str
    confidence: float


_LLM_TRACKER_SYSTEM_PROMPT = """\
You are a cognitive state analyst. You receive a webcam image of the user \
alongside sensor data and a history of your previous emotional assessments \
with timestamps.

You are the primary source for facial expression assessment — no rule-based \
expression classifier is used. Your visual analysis of the user's face is \
the sole input for expression-related state.

Your tasks:
1. Produce a brief (1-2 sentence) emotional state summary describing what \
you observe about the user right now — facial expression (including micro-expressions, \
brow position, mouth shape, and eye tension), body language, apparent mood or \
energy level.

2. The transition must always be true.

Respond in exactly this JSON:
{"emotional_summary": "1-2 sentence observation of current emotional state", \
"transition": true, "new_state": "focused|fatigued|stressed|distracted", \
"confidence": 0.0-1.0, "reasoning": "brief explanation, referencing temporal \
persistence if transition is true"}

Consider:
- Visual cues: facial expression, posture, eye openness, tension
- Sensor signals provided
- Previous emotional assessments and their timestamps for temporal context
- Whether they show signs of stress or fatigue not captured by sensors\
"""

_LLM_TRACKER_MODEL = "gpt-4.1-mini"
_LLM_TRACKER_COOLDOWN = 10.0
_LLM_TRACKER_CHECK_INTERVAL = 5.0

_STATE_FROM_STRING: dict[str, CognitiveStateLabel] = {
    "focused": CognitiveStateLabel.FOCUSED,
    "fatigued": CognitiveStateLabel.FATIGUED,
    "stressed": CognitiveStateLabel.STRESSED,
    "distracted": CognitiveStateLabel.DISTRACTED,
}


class LLMStateTracker:
    """State tracker that uses OpenAI vision for transition detection.

    LLM detection runs in a daemon background thread so the pipeline
    thread is never blocked by the 1-5 s OpenAI call.
    """

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
        self._emotional_history: deque[_EmotionalSnapshot] = deque(maxlen=10)

        # Threading
        self._lock = threading.Lock()
        self._pending_transition: StateTransition | None = None
        self._detection_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    # -- Lifecycle ------------------------------------------------------------

    def start(self) -> None:
        """Spawn the background detection thread."""
        if self._detection_thread is not None:
            return
        self._stop_event.clear()
        self._detection_thread = threading.Thread(
            target=self._detection_loop,
            name="llm-state-detection",
            daemon=True,
        )
        self._detection_thread.start()
        logger.info("LLMStateTracker: background detection thread started")

    def stop(self) -> None:
        """Signal the background thread to stop and wait for it."""
        self._stop_event.set()
        if self._detection_thread is not None:
            self._detection_thread.join(timeout=10.0)
            self._detection_thread = None
            logger.info("LLMStateTracker: background detection thread stopped")

    # -- Pipeline-facing methods (called from main thread) --------------------

    def add_frame(self, analysis: FrameAnalysis) -> None:
        with self._lock:
            self._frames.append(analysis)
            self._prune_locked()

    def get_current_state(self) -> CognitiveState:
        with self._lock:
            return self._get_current_state_locked()

    def detect_transition(self) -> StateTransition | None:
        """Non-blocking poll: return any result the background thread produced."""
        with self._lock:
            result = self._pending_transition
            self._pending_transition = None
            return result

    def get_recent_analyses(self, seconds: float = 5.0) -> list[FrameAnalysis]:
        with self._lock:
            cutoff = time.time() - seconds
            return [f for f in self._frames if f.timestamp >= cutoff]

    # -- Background thread ----------------------------------------------------

    def _detection_loop(self) -> None:
        """Runs in a daemon thread. Periodically queries the LLM."""
        while not self._stop_event.is_set():
            try:
                self._detection_tick()
            except Exception:
                logger.exception("LLMStateTracker: background detection error")
            self._stop_event.wait(timeout=self._check_interval_seconds)

    def _detection_tick(self) -> None:
        now = time.time()

        # Under lock: check guards, snapshot state
        with self._lock:
            if len(self._frames) == 0:
                return

            if now - self._last_transition_time < self._cooldown_seconds:
                return

            if now - self._last_check_time < self._check_interval_seconds:
                return

            current = self._get_current_state_locked()
            self._last_check_time = now

        # Outside lock: capture JPEG (thread-safe on ScreenshotManager)
        jpeg_bytes = self._screenshot_manager.encode_jpeg()
        if not jpeg_bytes:
            return

        state_changed = (
            self._last_state is not None
            and current.label != self._last_state.label
        )
        reason = "state change" if state_changed else "interval check"
        logger.debug(
            "LLMStateTracker: background starting detection (%s), "
            "current=%s confidence=%.0f%%",
            reason,
            current.label.value,
            current.confidence * 100,
        )

        # Blocking LLM call — runs without any lock held
        transition = self._query_llm(jpeg_bytes, current)

        # Store result under lock
        with self._lock:
            if transition is not None:
                self._last_transition_time = time.time()
                self._last_state = transition.new_state
                self._pending_transition = transition
                logger.info(
                    "LLMStateTracker: background detected %s -> %s "
                    "(confidence=%.0f%%)",
                    transition.previous_state.label.value,
                    transition.new_state.label.value,
                    transition.new_state.confidence * 100,
                )
            else:
                logger.debug(
                    "LLMStateTracker: background no transition detected"
                )

    # -- Internal helpers -----------------------------------------------------

    def _get_current_state_locked(self) -> CognitiveState:
        """Compute current state. Caller must hold self._lock."""
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

    def _prune_locked(self) -> None:
        """Remove stale frames. Caller must hold self._lock."""
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

        logger.debug(
            "LLMStateTracker: LLM call model=%s system_prompt=%r context=%r",
            self._model,
            _LLM_TRACKER_SYSTEM_PROMPT,
            context,
        )

        try:
            t_start = time.time()
            stream = self._client.chat.completions.create(
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
                stream=True,
                stream_options={"include_usage": True},
            )

            chunks: list[str] = []
            ttft: float | None = None
            usage = None

            for chunk in stream:
                if chunk.usage is not None:
                    usage = chunk.usage
                if chunk.choices:
                    delta = chunk.choices[0].delta
                    if delta and delta.content:
                        if ttft is None:
                            ttft = time.time() - t_start
                        chunks.append(delta.content)

            t_total = time.time() - t_start
            raw = "".join(chunks).strip()
            logger.info("LLMStateTracker: LLM response raw=%r", raw)

            input_tokens = usage.prompt_tokens if usage else -1
            output_tokens = usage.completion_tokens if usage else -1
            logger.info(
                "LLMStateTracker: tokens_in=%d tokens_out=%d "
                "ttft=%.3fs latency=%.3fs",
                input_tokens,
                output_tokens,
                ttft if ttft is not None else -1.0,
                t_total,
            )
        except Exception:
            logger.exception("LLMStateTracker: OpenAI call failed")
            return None

        return self._parse_response(raw, current)

    def _build_context(self, current: CognitiveState) -> str:
        # Snapshot shared state under lock
        with self._lock:
            recent = list(
                f for f in self._frames if f.timestamp >= time.time() - 5.0
            )
            last_transition_time = self._last_transition_time
            last_state = self._last_state
            emotional_history = list(self._emotional_history)

        lines = [
            f"Current state: {current.label.value} "
            f"(confidence: {current.confidence:.0%})",
            f"Time since last transition: "
            f"{time.time() - last_transition_time:.0f}s",
        ]
        if last_state is not None:
            lines.append(f"Previous state: {last_state.label.value}")

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

        if emotional_history:
            lines.append("\n--- Previous Emotional Assessments ---")
            for snap in emotional_history:
                lines.append(
                    f"[{snap.iso_time}] ({snap.state}, "
                    f"{snap.confidence:.0%}): {snap.summary}"
                )

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

        now = time.time()
        summary = data.get("emotional_summary", "")
        state_str = data.get("new_state", current.label.value).lower()
        confidence = float(data.get("confidence", 0.5))

        if summary:
            snapshot = _EmotionalSnapshot(
                timestamp=now,
                iso_time=datetime.fromtimestamp(now).isoformat(
                    timespec="seconds"
                ),
                summary=summary,
                state=state_str,
                confidence=confidence,
            )
            with self._lock:
                self._emotional_history.append(snapshot)
            logger.info("LLMStateTracker: emotional state: %s", summary)

        if not data.get("transition", False):
            return None

        new_label = _STATE_FROM_STRING.get(state_str)
        if new_label is None:
            return None

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
        if f.gaze_label is not None:
            signals.append(f.gaze_label)
        elif f.gaze is not None:
            signals.append(ClassifierResult(label=f.gaze.direction, confidence=0.7))
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
                posture=ClassifierResult(label="upright", confidence=0.85),
            )
        )

    # Second half: stressed
    for i in range(10):
        t = base_time + 5.0 + i * 0.5
        tracker2.add_frame(
            FrameAnalysis(
                timestamp=t,
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
