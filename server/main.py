"""CognitiveSense entry point.

Usage::

    uv run cognitivesense [desktop|mirror]

Or directly::

    cd server && uv run python -m main [desktop|mirror]
"""

from __future__ import annotations

import os
import sys

# Add the server directory to the path so all server modules are importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "server"))

from openai import OpenAI
from config.logging_config import setup_logging
from core.pipeline_controller import PipelineController
from input.camera_adapter import LocalCameraAdapter
from input.screenshot_manager import ScreenshotManager
from models.types import PipelineConfig
from reasoning.llm_engine import LLMEngine, RateLimiter
from state.state_tracker import LLMStateTracker, StateTracker


def main() -> None:
    setup_logging()

    env = sys.argv[1] if len(sys.argv) > 1 else "desktop"
    config = PipelineConfig.mirror() if env == "mirror" else PipelineConfig.desktop()

    tracker_type = os.environ.get("STATE_TRACKER_TYPE", config.state_tracker_type)

    client = OpenAI()
    engine = LLMEngine(
        client=client,
        rate_limiter=RateLimiter(cooldown_seconds=config.llm_cooldown_seconds),
    )

    camera = LocalCameraAdapter(config.camera_index)
    screenshot_manager = ScreenshotManager(camera)

    state_tracker: StateTracker | LLMStateTracker
    if tracker_type == "llm":
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
    )
    controller.run()


if __name__ == "__main__":
    main()
