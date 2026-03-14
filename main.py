"""CognitiveSense entry point.

Usage::

    uv run cognitivesense [desktop|mirror]

Or directly::

    cd server && uv run python -m main [desktop|mirror]
"""

from __future__ import annotations

import sys
import os

# Add the server directory to the path so all server modules are importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "server"))

from openai import OpenAI
from config.logging_config import setup_logging
from models.types import PipelineConfig
from core.pipeline_controller import PipelineController
from reasoning.llm_engine import LLMEngine, RateLimiter


def main() -> None:
    setup_logging()

    env = sys.argv[1] if len(sys.argv) > 1 else "desktop"
    config = PipelineConfig.mirror() if env == "mirror" else PipelineConfig.desktop()

    client = OpenAI()
    engine = LLMEngine(
        client=client,
        rate_limiter=RateLimiter(cooldown_seconds=config.llm_cooldown_seconds),
    )

    controller = PipelineController(config=config, llm_engine=engine)
    controller.run()


if __name__ == "__main__":
    main()
