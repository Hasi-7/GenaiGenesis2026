from __future__ import annotations

import logging
import os

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
PROJECT_PACKAGE = "server"


class _InfoFilter(logging.Filter):
    """Pass only INFO-level records."""

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno == logging.INFO


def setup_logging() -> None:
    """Configure logging for the project.

    - logs/all_logs.log: ALL logs (every logger, every level)
    - logs/project_logs.log: project logs only, DEBUG and above
    - logs/info_logs.log: project logs only, INFO only
    - console: project logs only, INFO and above
    """
    os.makedirs(LOG_DIR, exist_ok=True)

    fmt = "%(asctime)s %(name)s %(levelname)s %(message)s"
    formatter = logging.Formatter(fmt)

    # --- File handler: every log from every logger ---
    all_handler = logging.FileHandler(
        os.path.join(LOG_DIR, "all_logs.log"),
    )
    all_handler.setLevel(logging.DEBUG)
    all_handler.setFormatter(formatter)

    # --- File handler: project logs only (DEBUG+) ---
    project_file_handler = logging.FileHandler(
        os.path.join(LOG_DIR, "project_logs.log"),
    )
    project_file_handler.setLevel(logging.DEBUG)
    project_file_handler.setFormatter(formatter)

    # --- File handler: project INFO logs only ---
    info_handler = logging.FileHandler(
        os.path.join(LOG_DIR, "info_logs.log"),
    )
    info_handler.setLevel(logging.INFO)
    info_handler.addFilter(_InfoFilter())
    info_handler.setFormatter(formatter)

    # --- Console handler: project logs only (INFO+) ---
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # Root logger gets the all-logs handler
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(all_handler)

    # Project loggers get the project-specific handlers.
    # Modules use logging.getLogger(__name__) which produces names like
    # "core.pipeline_controller", "vision.blink_detector", etc.
    # Attach handlers to each top-level project package.
    project_packages = [
        "core",
        "vision",
        "audio",
        "input",
        "state",
        "reasoning",
        "ui",
        "models",
        "config",
        "__main__",
    ]
    for pkg in project_packages:
        pkg_logger = logging.getLogger(pkg)
        pkg_logger.addHandler(project_file_handler)
        pkg_logger.addHandler(info_handler)
        pkg_logger.addHandler(console_handler)
