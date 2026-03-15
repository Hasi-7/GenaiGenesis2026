from __future__ import annotations

import logging
import threading

from config.settings import get_settings
from core.pipeline_controller import PipelineController
from input.remote_media_server import RemoteClientSession
from input.screenshot_manager import ScreenshotManager
from models.types import PipelineConfig
from openai import OpenAI
from reasoning.llm_engine import LLMEngine, RateLimiter
from state.state_tracker import LLMStateTracker, StateTracker

from server.control.store import get_control_store
from server.control.telemetry_recorder import (
    CompositeTelemetrySink,
    DeviceIdentity,
    TelemetryRecorder,
)

logger = logging.getLogger(__name__)


class RemoteSessionRunner:
    def __init__(
        self,
        session: RemoteClientSession,
        *,
        config: PipelineConfig,
        tracker_type: str,
        openai_client: OpenAI | None,
        speech_tone_backend: str = "heuristic",
    ) -> None:
        self._session = session
        self._thread: threading.Thread | None = None
        self._telemetry_recorder = TelemetryRecorder(
            get_control_store(),
            lambda: DeviceIdentity(
                device_key=session.device_key,
                display_name=session.display_name,
                source_kind=session.source_name,
                last_ip=session.remote_ip,
                transport_session_label=session.session_label,
            ),
        )

        settings = get_settings()
        screenshot_manager = ScreenshotManager(session.make_camera_source())
        if tracker_type == "llm" and openai_client is not None:
            state_tracker = LLMStateTracker(
                client=openai_client,
                screenshot_manager=screenshot_manager,
                window_seconds=config.smoothing_window_seconds,
                cooldown_seconds=settings.llm_state_tracker_cooldown,
                check_interval_seconds=settings.llm_state_tracker_min_interval,
                model=settings.llm_state_tracker_model,
            )
        else:
            state_tracker = StateTracker(
                window_seconds=config.smoothing_window_seconds,
            )

        llm_engine = LLMEngine(
            client=openai_client,
            rate_limiter=RateLimiter(cooldown_seconds=config.llm_cooldown_seconds),
            model=settings.llm_feedback_model,
        )

        self._controller = PipelineController(
            config=config,
            llm_engine=llm_engine,
            screenshot_manager=screenshot_manager,
            state_tracker=state_tracker,
            mic_source=session.make_mic_source(),
            telemetry_sink=CompositeTelemetrySink(session, self._telemetry_recorder),
            speech_tone_backend=speech_tone_backend,
        )

    def start(self) -> None:
        if self._thread is not None:
            return

        self._thread = threading.Thread(
            target=self._run,
            name=f"remote-session-{self._session.session_id}",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._controller.request_stop()
        self._telemetry_recorder.close()

    def join(self, timeout: float | None = None) -> None:
        if self._thread is None:
            return
        self._thread.join(timeout=timeout)

    def _run(self) -> None:
        logger.debug(
            "Starting pipeline for remote session %s", self._session.session_label
        )
        try:
            self._controller.run()
        finally:
            self._telemetry_recorder.close()
            logger.debug(
                "Remote session %s pipeline stopped", self._session.session_label
            )
