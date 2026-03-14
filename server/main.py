"""CognitiveSense entry point.

Usage::

    uv run python main.py [desktop|mirror|server]

Or directly::

    uv run python -m server.main [desktop|mirror|server]
"""

from __future__ import annotations

import logging
import os
import sys

# Add the server directory to the path so sibling packages are importable.
sys.path.insert(0, os.path.dirname(__file__))

from config.logging_config import setup_logging
from core.pipeline_controller import PipelineController
from input.camera_adapter import LocalCameraAdapter, NetworkCameraAdapter
from input.remote_media_server import RemoteMediaServer
from input.screenshot_manager import ScreenshotManager
from models.types import PipelineConfig
from openai import OpenAI
from reasoning.llm_engine import LLMEngine, RateLimiter
from state.state_tracker import LLMStateTracker, StateTracker

logger = logging.getLogger(__name__)


def _running_in_wsl() -> bool:
    if "WSL_DISTRO_NAME" in os.environ:
        return True

    try:
        with open("/proc/version", encoding="utf-8") as version_file:
            version = version_file.read()
    except OSError:
        return False

    return "microsoft" in version.lower()


def _log_wsl_mirror_networking_help(port: int) -> None:
    if not _running_in_wsl():
        return

    logger.warning(
        "Mirror mode is running inside WSL2. Raspberry Pi devices usually cannot "
        "reach the Python receiver directly through the Windows Wi-Fi IP unless "
        "Windows forwards TCP %d into WSL.",
        port,
    )
    logger.warning(
        "Run an elevated Windows PowerShell and execute "
        "scripts/setup_wsl_mirror_proxy.ps1 -Port %d, or run the mirror server "
        "from native Windows Python instead of WSL.",
        port,
    )


def main() -> None:
    setup_logging()

    env = sys.argv[1] if len(sys.argv) > 1 else "desktop"
    if env == "mirror":
        config = PipelineConfig.mirror()
    elif env in {"server", "frontend"}:
        if env == "frontend":
            logger.warning('Mode "frontend" is deprecated; use "server" instead')
        config = PipelineConfig.server()
    else:
        config = PipelineConfig.desktop()

    tracker_type = os.environ.get("STATE_TRACKER_TYPE", config.state_tracker_type)
    api_key = os.environ.get("OPENAI_API_KEY")
    client: OpenAI | None = OpenAI(api_key=api_key) if api_key else None

    logger.info(
        "Starting CognitiveSense in %s mode (fps=%d, tracker=%s)",
        config.environment.value,
        config.target_fps,
        tracker_type,
    )
    if env == "mirror":
        logger.info(
            "Mirror receiver configured for %s:%d",
            config.mirror_listen_host,
            config.mirror_listen_port,
        )
        _log_wsl_mirror_networking_help(config.mirror_listen_port)
    else:
        logger.info("Using local camera index %d", config.camera_index)

    if client is None:
        logger.info("OPENAI_API_KEY not set; LLM feedback is disabled")
        if tracker_type == "llm":
            logger.warning(
                "STATE_TRACKER_TYPE=llm requested without OPENAI_API_KEY; "
                "falling back to rule-based state tracking",
            )
            tracker_type = "rule"

    engine = LLMEngine(
        client=client,
        rate_limiter=RateLimiter(cooldown_seconds=config.llm_cooldown_seconds),
    )

    remote_media_server: RemoteMediaServer | None = None
    mic_source = None

    if config.remote_media_enabled:
        remote_media_server = RemoteMediaServer(
            host=config.remote_media_host,
            port=config.remote_media_port,
        )
        camera = remote_media_server.make_camera_source()
        mic_source = (
            remote_media_server.make_mic_source() if config.mic_enabled else None
        )
    elif env == "mirror":
        camera = NetworkCameraAdapter(
            config.mirror_listen_host,
            config.mirror_listen_port,
        )
    else:
        camera = LocalCameraAdapter(config.camera_index)
    screenshot_manager = ScreenshotManager(camera)
    if not screenshot_manager.is_opened():
        logger.error(
            "Receiver startup failed. Another process is already using %s:%d.",
            config.mirror_listen_host,
            config.mirror_listen_port,
        )
        screenshot_manager.release()
        return

    state_tracker: StateTracker | LLMStateTracker
    if tracker_type == "llm" and client is not None:
        state_tracker = LLMStateTracker(
            client=client,
            screenshot_manager=screenshot_manager,
            window_seconds=config.smoothing_window_seconds,
        )
    else:
        state_tracker = StateTracker(
            window_seconds=config.smoothing_window_seconds,
        )

    controller = PipelineController(
        config=config,
        llm_engine=engine,
        screenshot_manager=screenshot_manager,
        state_tracker=state_tracker,
        mic_source=mic_source,
        telemetry_sink=remote_media_server,
    )
    logger.info("Server startup complete; entering pipeline loop")
    controller.run()


if __name__ == "__main__":
    main()
