"""CognitiveSense entry point.

Usage::

    uv run python main.py [desktop|mirror|server|replay <session_dir>]

Or directly::

    uv run python -m server.main [desktop|mirror|server|replay <session_dir>]

Modes:
    desktop   Local webcam + mic with desktop overlay UI (default).
    mirror    Alias for server mode; accepts remote media on port 9000.
    server    Headless network receiver for remote camera/mic streams.
    replay    Replay a recorded session from <session_dir> through the
              full pipeline. See server/scripts/record_sample.py to
              capture sessions.
"""

from __future__ import annotations

import atexit
import logging
import os
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from typing import cast

# Add the server directory to the path so sibling packages are importable.
sys.path.insert(0, os.path.dirname(__file__))

from config.logging_config import setup_logging
from config.settings import get_settings
from core.pipeline_controller import PipelineController
from core.remote_session_runner import RemoteSessionRunner
from input.camera_adapter import LocalCameraAdapter
from input.remote_media_server import RemoteClientSession, RemoteMediaServer
from input.replay_adapters import ReplayCameraAdapter, ReplayMicAdapter
from input.screenshot_manager import ScreenshotManager
from models.types import InputSource, PipelineConfig
from openai import OpenAI
from samples.llm_engine import LLMEngine, RateLimiter
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


def _kill_processes_using_ports(*ports: int) -> None:
    unique_ports = tuple(sorted(set(ports)))
    if not unique_ports:
        return

    fuser = shutil.which("fuser")
    if fuser is not None:
        result = subprocess.run(
            [fuser, "-k", *[f"{port}/tcp" for port in unique_ports]],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode in {0, 1}:
            if result.returncode == 0:
                logger.warning(
                    "Killed existing processes using TCP ports %s",
                    ", ".join(str(port) for port in unique_ports),
                )
            return

    lsof = shutil.which("lsof")
    if lsof is None:
        logger.warning(
            "Could not verify or kill existing listeners on TCP ports %s",
            ", ".join(str(port) for port in unique_ports),
        )
        return

    result = subprocess.run(
        [lsof, "-nP", "-t", *[f"-iTCP:{port}" for port in unique_ports]],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return

    pids = {
        int(line.strip())
        for line in result.stdout.splitlines()
        if line.strip().isdigit() and int(line.strip()) != os.getpid()
    }
    for pid in sorted(pids):
        os.kill(pid, signal.SIGKILL)

    if pids:
        logger.warning(
            "Killed existing processes using TCP ports %s",
            ", ".join(str(port) for port in unique_ports),
        )


def _active_receiver_details(config: PipelineConfig) -> str | None:
    if config.input_source is InputSource.REMOTE_MEDIA:
        return (
            f"network ingress on {config.remote_media_host}:{config.remote_media_port}"
        )
    return None


def main() -> None:
    settings = get_settings()
    setup_logging()

    env = sys.argv[1] if len(sys.argv) > 1 else "desktop"
    if env == "replay":
        if len(sys.argv) < 3:
            logger.error("Usage: main.py replay <session_dir>")
            return
        config = PipelineConfig.replay(sys.argv[2])
    elif env in {"server", "frontend", "mirror"}:
        if env == "frontend":
            logger.warning('Mode "frontend" is deprecated; use "server" instead')
        if env == "mirror":
            logger.warning(
                'Mode "mirror" now uses the unified network server on port 9000'
            )
        config = PipelineConfig.server()
    else:
        config = PipelineConfig.desktop()

    if (
        settings.cognitivesense_server_port is not None
        and config.input_source is InputSource.REMOTE_MEDIA
    ):
        config.remote_media_port = settings.cognitivesense_server_port
    if (
        settings.cognitivesense_server_host is not None
        and config.input_source is InputSource.REMOTE_MEDIA
    ):
        config.remote_media_host = settings.cognitivesense_server_host

    if config.input_source is InputSource.REMOTE_MEDIA:
        _kill_processes_using_ports(config.remote_media_port)

    tracker_type = settings.state_tracker_type
    client: OpenAI | None = (
        OpenAI(api_key=settings.OPENAI_API_KEY)
        if settings.OPENAI_API_KEY
        else None
    )

    config.target_fps = settings.target_fps
    config.llm_cooldown_seconds = settings.llm_feedback_cooldown

    logger.debug(
        "Starting CognitiveSense in %s mode (fps=%d, tracker=%s)",
        config.environment.value,
        config.target_fps,
        tracker_type,
    )
    if config.input_source is InputSource.REMOTE_MEDIA:
        logger.debug(
            "Network ingress configured for %s:%d",
            config.remote_media_host,
            config.remote_media_port,
        )
        _log_wsl_mirror_networking_help(config.remote_media_port)
    elif config.input_source is InputSource.REPLAY:
        logger.debug("Replaying session from %s", config.replay_session_dir)
    else:
        logger.debug("Using local camera index %d", config.camera_index)

    if client is None:
        logger.debug("openai_api_key not set; LLM feedback is disabled")
        if tracker_type == "llm":
            logger.warning(
                "state_tracker_type=llm requested without openai_api_key; "
                "falling back to rule-based state tracking",
            )
            tracker_type = "rule"

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
    session_runners: dict[int, RemoteSessionRunner] = {}
    controller: PipelineController | None = None

    if config.input_source is InputSource.REMOTE_MEDIA:

        def _handle_session_connected(session: RemoteClientSession) -> None:
            runner = RemoteSessionRunner(
                session,
                config=config,
                tracker_type=tracker_type,
                openai_client=client,
                speech_tone_backend=settings.speech_tone_backend,
            )
            session_runners[session.session_id] = runner
            runner.start()

        def _handle_session_disconnected(session: RemoteClientSession) -> None:
            runner = session_runners.pop(session.session_id, None)
            if runner is None:
                return
            runner.stop()
            runner.join(timeout=2.0)

        remote_media_server = RemoteMediaServer(
            host=config.remote_media_host,
            port=config.remote_media_port,
            on_session_connected=_handle_session_connected,
            on_session_disconnected=_handle_session_disconnected,
        )
        cleanup_callbacks.append(remote_media_server.release)

        if not remote_media_server.is_opened():
            receiver = _active_receiver_details(config)
            logger.error("Receiver startup failed for %s.", receiver)
            _run_cleanup()
            return
    else:
        engine = LLMEngine(
            client=client,
            rate_limiter=RateLimiter(cooldown_seconds=config.llm_cooldown_seconds),
            model=settings.llm_feedback_model,
        )

        mic_source_instance = None
        if config.input_source is InputSource.REPLAY:
            camera = ReplayCameraAdapter(config.replay_session_dir)  # type: ignore[assignment]
            replay_mic = ReplayMicAdapter(config.replay_session_dir)
            mic_source_instance = replay_mic
            cleanup_callbacks.append(replay_mic.stop)
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
                logger.error(
                    "Receiver startup failed for %s.",
                    receiver,
                )
            _run_cleanup()
            return

        state_tracker: StateTracker | LLMStateTracker
        if tracker_type == "llm" and client is not None:
            state_tracker = LLMStateTracker(
                client=client,
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

        controller = PipelineController(
            config=config,
            llm_engine=engine,
            screenshot_manager=screenshot_manager,
            state_tracker=state_tracker,
            mic_source=mic_source_instance,
            telemetry_sink=None,
            speech_tone_backend=settings.speech_tone_backend,
        )
        cleanup_callbacks.append(controller.close)

    handled_signals: list[signal.Signals] = [signal.SIGINT, signal.SIGTERM]
    sigbreak = getattr(signal, "SIGBREAK", None)
    if isinstance(sigbreak, signal.Signals):
        handled_signals.append(sigbreak)

    previous_handlers: dict[signal.Signals, signal.Handlers] = {}

    def _handle_shutdown(signum: int, _frame: object) -> None:
        logger.debug("Received signal %s; shutting down", signum)
        if remote_media_server is not None:
            remote_media_server.release()
        for runner in list(session_runners.values()):
            runner.stop()
        if controller is not None:
            controller.request_stop()

    for sig in handled_signals:
        previous_handlers[sig] = cast(signal.Handlers, signal.getsignal(sig))
        signal.signal(sig, _handle_shutdown)

    logger.debug("Server startup complete; entering pipeline loop")
    try:
        if config.input_source is InputSource.REMOTE_MEDIA:
            while remote_media_server is not None and remote_media_server.is_opened():
                if cleanup_started:
                    break
                time.sleep(0.25)
        else:
            assert controller is not None
            controller.run()
    finally:
        for sig, previous in previous_handlers.items():
            signal.signal(sig, previous)
        atexit.unregister(_run_cleanup)
        _run_cleanup()


if __name__ == "__main__":
    main()
