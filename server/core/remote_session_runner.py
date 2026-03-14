from __future__ import annotations

import logging
import threading

from core.pipeline_controller import PipelineController
from input.remote_media_server import RemoteClientSession
from input.screenshot_manager import ScreenshotManager
from models.types import PipelineConfig
from openai import OpenAI
from reasoning.llm_engine import LLMEngine, RateLimiter
from state.state_tracker import LLMStateTracker, StateTracker

logger = logging.getLogger(__name__)


class RemoteSessionRunner:
    def __init__(
        self,
        session: RemoteClientSession,
        *,
        config: PipelineConfig,
        tracker_type: str,
        openai_client: OpenAI | None,
    ) -> None:
        self._session = session
        self._thread: threading.Thread | None = None

        screenshot_manager = ScreenshotManager(session.make_camera_source())
        if tracker_type == "llm" and openai_client is not None:
            state_tracker = LLMStateTracker(
                client=openai_client,
                screenshot_manager=screenshot_manager,
                window_seconds=config.smoothing_window_seconds,
            )
        else:
            state_tracker = StateTracker(
                window_seconds=config.smoothing_window_seconds,
            )

        llm_engine = LLMEngine(
            client=openai_client,
            rate_limiter=RateLimiter(cooldown_seconds=config.llm_cooldown_seconds),
        )

        self._controller = PipelineController(
            config=config,
            llm_engine=llm_engine,
            screenshot_manager=screenshot_manager,
            state_tracker=state_tracker,
            mic_source=session.make_mic_source(),
            telemetry_sink=session,
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

    def join(self, timeout: float | None = None) -> None:
        if self._thread is None:
            return
        self._thread.join(timeout=timeout)

    def _run(self) -> None:
        logger.info(
            "Starting pipeline for remote session %s", self._session.session_label
        )
        try:
            self._controller.run()
        finally:
            logger.info(
                "Remote session %s pipeline stopped", self._session.session_label
            )
