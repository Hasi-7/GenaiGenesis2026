from __future__ import annotations

import time
from dataclasses import dataclass

from models.types import (
    CognitiveState,
    CognitiveStateLabel,
    PipelineConfig,
    SustainedStateAlert,
)


@dataclass(slots=True)
class _AlertEpisode:
    label: CognitiveStateLabel
    started_at: float
    last_seen_at: float
    peak_confidence: float
    last_notified_at: float = 0.0
    notification_count: int = 0
    recovery_started_at: float | None = None


class SustainedStateMonitor:
    def __init__(self, config: PipelineConfig) -> None:
        self._enabled = config.sustained_alerts_enabled
        self._threshold_seconds = max(
            config.sustained_alert_window_seconds,
            config.sustained_alert_min_duration_seconds,
        )
        self._recovery_seconds = config.sustained_alert_recovery_seconds
        self._repeat_cooldown_seconds = config.sustained_alert_repeat_cooldown_seconds
        self._max_notifications_per_episode = (
            config.sustained_alert_max_notifications_per_episode
        )
        self._bad_labels = {
            CognitiveStateLabel(label) for label in config.sustained_alert_bad_labels
        }
        self._last_sample_second = -1
        self._active_episode: _AlertEpisode | None = None

    def observe(self, state: CognitiveState) -> SustainedStateAlert | None:
        if not self._enabled:
            return None

        now = time.time()
        sample_second = int(now)
        if sample_second == self._last_sample_second:
            return None
        self._last_sample_second = sample_second

        episode = self._active_episode
        if state.label not in self._bad_labels:
            if episode is None:
                return None
            if episode.recovery_started_at is None:
                episode.recovery_started_at = now
            if now - episode.recovery_started_at >= self._recovery_seconds:
                self._active_episode = None
            return None

        if episode is None or episode.label is not state.label:
            self._active_episode = _AlertEpisode(
                label=state.label,
                started_at=now,
                last_seen_at=now,
                peak_confidence=state.confidence,
            )
            return None

        episode.last_seen_at = now
        episode.peak_confidence = max(episode.peak_confidence, state.confidence)
        episode.recovery_started_at = None

        duration_seconds = max(0.0, now - episode.started_at)
        if duration_seconds < self._threshold_seconds:
            return None
        if episode.notification_count >= self._max_notifications_per_episode:
            return None
        if (
            episode.notification_count > 0
            and now - episode.last_notified_at < self._repeat_cooldown_seconds
        ):
            return None

        episode.last_notified_at = now
        episode.notification_count += 1
        return SustainedStateAlert(
            label=episode.label,
            confidence=max(state.confidence, episode.peak_confidence),
            started_at=episode.started_at,
            triggered_at=now,
            duration_seconds=duration_seconds,
            sample_count=max(1, int(duration_seconds)),
            repeat_count=max(0, episode.notification_count - 1),
        )
