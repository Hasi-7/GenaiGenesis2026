from __future__ import annotations

from collections.abc import Iterable

from models.types import CognitiveState, CognitiveStateLabel

INDICATOR_NONE = 0
INDICATOR_BLINK_ELEVATED = 1
INDICATOR_BLINK_SUPPRESSED = 2
INDICATOR_POSTURE_SLOUCHED = 3
INDICATOR_POSTURE_LEANING = 4
INDICATOR_GAZE_DISTRACTED = 5
INDICATOR_EXPRESSION_TENSE = 6
INDICATOR_SPEECH_STRESSED = 7
INDICATOR_SPEECH_MONOTONE = 8
INDICATOR_POSTURE_UPRIGHT = 9
INDICATOR_GAZE_FOCUSED = 10
INDICATOR_EXPRESSION_RELAXED = 11
INDICATOR_SPEECH_CALM = 12

RECOMMENDATION_NONE = 0
RECOMMENDATION_TAKE_BREAK = 1
RECOMMENDATION_HYDRATE = 2
RECOMMENDATION_STRETCH = 3
RECOMMENDATION_BREATHE = 4
RECOMMENDATION_REFOCUS = 5
RECOMMENDATION_SILENCE_DISTRACTIONS = 6
RECOMMENDATION_KEEP_PACE = 7
RECOMMENDATION_POSTURE_RESET = 8

MAX_INDICATORS = 3
MAX_RECOMMENDATIONS = 3

_SIGNAL_TO_INDICATOR = {
    "fatigued": INDICATOR_BLINK_ELEVATED,
    "elevated": INDICATOR_BLINK_ELEVATED,
    "stressed": INDICATOR_SPEECH_STRESSED,
    "slouched": INDICATOR_POSTURE_SLOUCHED,
    "slouching": INDICATOR_POSTURE_SLOUCHED,
    "leaning": INDICATOR_POSTURE_LEANING,
    "distracted": INDICATOR_GAZE_DISTRACTED,
    "tense": INDICATOR_EXPRESSION_TENSE,
    "monotone": INDICATOR_SPEECH_MONOTONE,
    "upright": INDICATOR_POSTURE_UPRIGHT,
    "focused": INDICATOR_GAZE_FOCUSED,
    "relaxed": INDICATOR_EXPRESSION_RELAXED,
    "calm": INDICATOR_SPEECH_CALM,
}

_STATE_TO_RECOMMENDATIONS = {
    CognitiveStateLabel.FATIGUED: (
        RECOMMENDATION_TAKE_BREAK,
        RECOMMENDATION_HYDRATE,
        RECOMMENDATION_STRETCH,
    ),
    CognitiveStateLabel.STRESSED: (
        RECOMMENDATION_BREATHE,
        RECOMMENDATION_HYDRATE,
        RECOMMENDATION_POSTURE_RESET,
    ),
    CognitiveStateLabel.DISTRACTED: (
        RECOMMENDATION_REFOCUS,
        RECOMMENDATION_SILENCE_DISTRACTIONS,
        RECOMMENDATION_POSTURE_RESET,
    ),
    CognitiveStateLabel.FOCUSED: (
        RECOMMENDATION_KEEP_PACE,
        RECOMMENDATION_HYDRATE,
        RECOMMENDATION_POSTURE_RESET,
    ),
    CognitiveStateLabel.UNKNOWN: (),
}


def top_indicator_codes(state: CognitiveState) -> tuple[int, ...]:
    seen: set[int] = set()
    ordered: list[int] = []

    for signal in _sorted_unique_signal_labels(state):
        indicator = _SIGNAL_TO_INDICATOR.get(signal)
        if indicator is None or indicator in seen:
            continue
        seen.add(indicator)
        ordered.append(indicator)
        if len(ordered) == MAX_INDICATORS:
            return tuple(ordered)

    fallback = _fallback_indicator(state.label)
    if fallback is not None and fallback not in seen:
        ordered.append(fallback)

    return tuple(ordered[:MAX_INDICATORS])


def recommendation_codes(state: CognitiveState) -> tuple[int, ...]:
    return _STATE_TO_RECOMMENDATIONS.get(state.label, ())[:MAX_RECOMMENDATIONS]


def padded_codes(codes: Iterable[int], size: int) -> tuple[int, ...]:
    trimmed = list(codes)[:size]
    return tuple(trimmed + [0] * (size - len(trimmed)))


def _sorted_unique_signal_labels(state: CognitiveState) -> list[str]:
    labels: list[str] = []
    seen: set[str] = set()
    for signal in sorted(
        state.contributing_signals,
        key=lambda item: item.confidence,
        reverse=True,
    ):
        label = signal.label.lower()
        if label in seen:
            continue
        seen.add(label)
        labels.append(label)
    return labels


def _fallback_indicator(state_label: CognitiveStateLabel) -> int | None:
    if state_label is CognitiveStateLabel.FATIGUED:
        return INDICATOR_BLINK_ELEVATED
    if state_label is CognitiveStateLabel.STRESSED:
        return INDICATOR_EXPRESSION_TENSE
    if state_label is CognitiveStateLabel.DISTRACTED:
        return INDICATOR_GAZE_DISTRACTED
    if state_label is CognitiveStateLabel.FOCUSED:
        return INDICATOR_GAZE_FOCUSED
    return None
