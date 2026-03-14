from __future__ import annotations

import time

from dotenv import load_dotenv
from openai import OpenAI

from models.types import (
    CognitiveStateLabel,
    LLMRequest,
    LLMResponse,
)

load_dotenv()

_DEFAULT_COOLDOWN_SECONDS = 30.0
_MODEL = "gpt-4.1-nano"

_SYSTEM_PROMPT = """\
You are a concise cognitive wellness advisor. The user is being \
monitored by sensors that detect their cognitive state in real time. \
You receive state transition data and must provide brief, actionable \
feedback.

Respond in exactly this format:
FEEDBACK: <1-2 sentence observation and advice>

Be warm but direct. Do not repeat the raw data back.\
"""


class RateLimiter:
    """Enforces cooldown between LLM calls."""

    def __init__(
        self,
        cooldown_seconds: float = _DEFAULT_COOLDOWN_SECONDS,
    ) -> None:
        self._cooldown_seconds = cooldown_seconds
        self._last_call_timestamp: float = 0.0

    def is_allowed(self) -> bool:
        """Return True if cooldown has elapsed."""
        elapsed = time.time() - self._last_call_timestamp
        return elapsed >= self._cooldown_seconds

    def record_call(self) -> None:
        """Record that a call was made."""
        self._last_call_timestamp = time.time()


class LLMEngine:
    """LLM reasoning engine using OpenAI for cognitive feedback."""

    def __init__(
        self,
        client: OpenAI,
        rate_limiter: RateLimiter | None = None,
        model: str = _MODEL,
    ) -> None:
        self._client = client
        self._rate_limiter = rate_limiter or RateLimiter()
        self._model = model

    @property
    def rate_limiter(self) -> RateLimiter:
        return self._rate_limiter

    def request_feedback(
        self, request: LLMRequest
    ) -> LLMResponse | None:
        """Call OpenAI for feedback, or None if rate limited."""
        if not self._rate_limiter.is_allowed():
            return None

        user_msg = _build_user_message(request)

        completion = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=200,
            temperature=0.7,
        )

        self._rate_limiter.record_call()

        raw = (completion.choices[0].message.content or "").strip()
        feedback = raw.removeprefix("FEEDBACK:").strip()

        return LLMResponse(
            feedback_text=feedback,
            timestamp=time.time(),
        )


def _build_user_message(request: LLMRequest) -> str:
    cur = request.current_state
    prev = request.transition.previous_state
    signals = ", ".join(
        f"{s.label} ({s.confidence:.0%})"
        for s in cur.contributing_signals[:5]
    )
    return (
        f"State changed: {prev.label.value} -> {cur.label.value}\n"
        f"New confidence: {cur.confidence:.0%}\n"
        f"Signals: {signals or 'none'}"
    )


if __name__ == "__main__":
    from models.types import (
        ClassifierResult,
        CognitiveState,
        FrameAnalysis,
        StateTransition,
    )

    def _make_request(
        label: CognitiveStateLabel,
        prev_label: CognitiveStateLabel = CognitiveStateLabel.FOCUSED,
    ) -> LLMRequest:
        now = time.time()
        state = CognitiveState(
            label=label,
            confidence=0.8,
            contributing_signals=[
                ClassifierResult(label="tense", confidence=0.9),
                ClassifierResult(
                    label="stressed", confidence=0.85
                ),
            ],
            timestamp=now,
        )
        prev = CognitiveState(
            label=prev_label,
            confidence=0.7,
            contributing_signals=[],
            timestamp=now - 5.0,
        )
        return LLMRequest(
            frame_jpeg_bytes=b"",
            current_state=state,
            transition=StateTransition(
                previous_state=prev,
                new_state=state,
                transition_time=now,
            ),
            recent_analyses=[FrameAnalysis(timestamp=now)],
        )

    print("=== LLMEngine Demo ===\n")

    client = OpenAI()
    limiter = RateLimiter(cooldown_seconds=0.0)
    engine = LLMEngine(
        client=client, rate_limiter=limiter
    )

    # Test 1: Get feedback for a state transition
    print("--- OpenAI call test ---")
    req = _make_request(CognitiveStateLabel.STRESSED)
    resp = engine.request_feedback(req)
    assert resp is not None, "Expected a response"
    assert len(resp.feedback_text) > 0, "Feedback should not be empty"
    assert resp.timestamp > 0, "Timestamp should be set"
    print(f"Feedback: {resp.feedback_text}")
    print(f"Timestamp: {resp.timestamp:.1f}")
    print("OpenAI call ✓\n")

    # Test 2: Cooldown blocks second call
    print("--- Cooldown test ---")
    limiter2 = RateLimiter(cooldown_seconds=30.0)
    engine2 = LLMEngine(client=client, rate_limiter=limiter2)

    resp1 = engine2.request_feedback(
        _make_request(CognitiveStateLabel.FATIGUED)
    )
    assert resp1 is not None, "First call should succeed"
    print(f"Call 1: {resp1.feedback_text}")

    resp2 = engine2.request_feedback(
        _make_request(CognitiveStateLabel.DISTRACTED)
    )
    assert resp2 is None, "Second call should be rate limited"
    print("Call 2: None (rate limited) ✓")

    print("\n=== Done ===")
