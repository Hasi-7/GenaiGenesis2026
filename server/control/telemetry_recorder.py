from __future__ import annotations

import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from models.types import CognitiveState, FrameAnalysis, LLMResponse

from control.store import ControlStore


@dataclass(slots=True)
class DeviceIdentity:
    device_key: str
    display_name: str
    source_kind: str
    last_ip: str | None
    transport_session_label: str


class CompositeTelemetrySink:
    def __init__(self, *sinks: object) -> None:
        self._sinks = sinks

    def publish_state(self, state: CognitiveState, analysis: FrameAnalysis) -> None:
        for sink in self._sinks:
            publish = getattr(sink, "publish_state", None)
            if callable(publish):
                publish(state, analysis)

    def publish_feedback(self, response: LLMResponse, state: CognitiveState) -> None:
        for sink in self._sinks:
            publish = getattr(sink, "publish_feedback", None)
            if callable(publish):
                publish(response, state)

    def publish_snapshot(self, jpeg_bytes: bytes, recorded_at: float) -> None:
        for sink in self._sinks:
            publish = getattr(sink, "publish_snapshot", None)
            if callable(publish):
                publish(jpeg_bytes, recorded_at)


class TelemetryRecorder:
    def __init__(
        self,
        store: ControlStore,
        identity_provider: Callable[[], DeviceIdentity],
    ) -> None:
        self._store = store
        self._identity_provider = identity_provider
        self._db_session_id: int | None = None
        self._last_state_sample_at = 0.0
        self._last_snapshot_at = 0.0
        self._snapshot_dir = Path(store.db_path).parent / "snapshots"
        self._snapshot_dir.mkdir(parents=True, exist_ok=True)

    def publish_state(self, state: CognitiveState, analysis: FrameAnalysis) -> None:
        now = time.time()
        if now - self._last_state_sample_at < 1.0:
            return
        self._last_state_sample_at = now
        session_id = self._ensure_session()
        self._store.add_state_sample(
            session_id=session_id,
            recorded_at=analysis.timestamp,
            state_label=state.label.value,
            confidence=state.confidence,
            indicators=[signal.label for signal in state.contributing_signals[:5]],
        )

    def publish_feedback(self, response: LLMResponse, state: CognitiveState) -> None:
        del state
        session_id = self._ensure_session()
        self._store.add_feedback_event(
            session_id=session_id,
            recorded_at=response.timestamp,
            trigger_kind=response.trigger_kind,
            severity=response.severity,
            should_notify=response.should_notify,
            text=response.feedback_text,
        )

    def publish_snapshot(self, jpeg_bytes: bytes, recorded_at: float) -> None:
        del recorded_at
        now = time.time()
        if now - self._last_snapshot_at < 1.0:
            return
        self._last_snapshot_at = now
        identity = self._identity_provider()
        snapshot_path = (
            self._snapshot_dir / f"{_safe_filename(identity.device_key)}.jpg"
        )
        snapshot_path.write_bytes(jpeg_bytes)

    def close(self) -> None:
        if self._db_session_id is None:
            return
        self._store.close_session(self._db_session_id)
        self._db_session_id = None

    def _ensure_session(self) -> int:
        if self._db_session_id is not None:
            return self._db_session_id
        identity: DeviceIdentity = self._identity_provider()
        self._db_session_id = self._store.open_session(
            device_key=identity.device_key,
            source_kind=identity.source_kind,
            display_name=identity.display_name,
            transport_session_label=identity.transport_session_label,
            last_ip=identity.last_ip,
        )
        return self._db_session_id


def _safe_filename(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-") or "device"
