"""CognitiveSense entry point.

Usage::

    uv run python main.py [desktop|mirror|server]

Or directly::

    uv run python -m server.main [desktop|mirror|server]
"""

from __future__ import annotations

import atexit
import logging
import os
import signal
import sys
from collections.abc import Callable
from typing import cast

# Add the server directory to the path so sibling packages are importable.
sys.path.insert(0, os.path.dirname(__file__))

from config.logging_config import setup_logging
from core.pipeline_controller import PipelineController
from input.camera_adapter import LocalCameraAdapter, NetworkCameraAdapter
from input.remote_media_server import RemoteMediaServer
from input.screenshot_manager import ScreenshotManager
from models.types import InputSource, PipelineConfig
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


def _active_receiver_details(config: PipelineConfig) -> tuple[str, str, int] | None:
    if config.input_source is InputSource.MIRROR_TCP:
        return ("Mirror receiver", config.mirror_listen_host, config.mirror_listen_port)
    if config.input_source is InputSource.REMOTE_MEDIA:
        return (
            "Remote media bridge",
            config.remote_media_host,
            config.remote_media_port,
        )
    return None


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

    mirror_port = os.environ.get("COGNITIVESENSE_MIRROR_PORT")
    server_port = os.environ.get("COGNITIVESENSE_SERVER_PORT")
    if mirror_port is not None and config.input_source is InputSource.MIRROR_TCP:
        config.mirror_listen_port = int(mirror_port)
    if server_port is not None and config.input_source is InputSource.REMOTE_MEDIA:
        config.remote_media_port = int(server_port)

    tracker_type = os.environ.get("STATE_TRACKER_TYPE", config.state_tracker_type)
    api_key = os.environ.get("OPENAI_API_KEY")
    client: OpenAI | None = OpenAI(api_key=api_key) if api_key else None

    logger.info(
        "Starting CognitiveSense in %s mode (fps=%d, tracker=%s)",
        config.environment.value,
        config.target_fps,
        tracker_type,
    )
    if config.input_source is InputSource.MIRROR_TCP:
        logger.info(
            "Mirror receiver configured for %s:%d",
            config.mirror_listen_host,
            config.mirror_listen_port,
        )
        _log_wsl_mirror_networking_help(config.mirror_listen_port)
    elif config.input_source is InputSource.REMOTE_MEDIA:
        logger.info(
            "Remote media bridge configured for %s:%d",
            config.remote_media_host,
            config.remote_media_port,
        )
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

    cleanup_callbacks: list[Callable[[], None]] = []
    cleanup_started = False

    def _run_cleanup() -> None:
        nonlocal cleanup_started
        if cleanup_started:
            return
        cleanup_started = True

        while cleanup_callbacks:
            callback = cleanup_callbacks.pop()
            try:
                callback()
            except Exception:
                logger.exception("Cleanup callback failed")

    atexit.register(_run_cleanup)

    remote_media_server: RemoteMediaServer | None = None
    mic_source = None

    if config.input_source is InputSource.REMOTE_MEDIA:
        remote_media_server = RemoteMediaServer(
            host=config.remote_media_host,
            port=config.remote_media_port,
        )
        cleanup_callbacks.append(remote_media_server.release)
        camera = remote_media_server.make_camera_source()
        mic_source = (
            remote_media_server.make_mic_source() if config.mic_enabled else None
        )
    elif config.input_source is InputSource.MIRROR_TCP:
        camera = NetworkCameraAdapter(
            config.mirror_listen_host,
            config.mirror_listen_port,
        )
    else:
        camera = LocalCameraAdapter(config.camera_index)
    screenshot_manager = ScreenshotManager(camera)
    cleanup_callbacks.append(screenshot_manager.release)
    if not screenshot_manager.is_opened():
        receiver = _active_receiver_details(config)
        if receiver is None:
            logger.error(
                "Camera startup failed. Could not open local camera index %d.",
                config.camera_index,
            )
        else:
            receiver_name, receiver_host, receiver_port = receiver
            logger.error(
                "%s startup failed. Another process may already be using %s:%d.",
                receiver_name,
                receiver_host,
                receiver_port,
            )
        _run_cleanup()
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
    cleanup_callbacks.append(controller.close)

    handled_signals: list[signal.Signals] = [signal.SIGINT, signal.SIGTERM]
    sigbreak = getattr(signal, "SIGBREAK", None)
    if isinstance(sigbreak, signal.Signals):
        handled_signals.append(sigbreak)

    previous_handlers: dict[signal.Signals, signal.Handlers] = {}

    def _handle_shutdown(signum: int, _frame: object) -> None:
        logger.info("Received signal %s; shutting down", signum)
        controller.request_stop()

    for sig in handled_signals:
        previous_handlers[sig] = cast(signal.Handlers, signal.getsignal(sig))
        signal.signal(sig, _handle_shutdown)

    logger.info("Server startup complete; entering pipeline loop")
    try:
        controller.run()
    finally:
        for sig, previous in previous_handlers.items():
            signal.signal(sig, previous)
        atexit.unregister(_run_cleanup)
        _run_cleanup()


if __name__ == "__main__":
    main()
