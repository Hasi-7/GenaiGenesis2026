from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

try:
    from enum import StrEnum
except ImportError:  # pragma: no cover - Python < 3.11 compatibility.
    class StrEnum(str, Enum):  # noqa: UP042
        """Fallback for Python versions without enum.StrEnum."""


class ClientType(StrEnum):
    """Supported capture clients."""

    DESKTOP = "desktop"
    MIRROR = "mirror"


@dataclass(slots=True)
class VideoFrame:
    """Raw frame captured from a local camera device."""

    client_id: str
    client_type: ClientType
    timestamp: float
    width: int
    height: int
    pixel_format: str
    frame_bytes: bytes


@dataclass(slots=True)
class AudioChunk:
    """Raw PCM audio captured from a local microphone device."""

    client_id: str
    client_type: ClientType
    timestamp: float
    sample_rate: int
    channels: int
    sample_count: int
    pcm_bytes: bytes


@dataclass(slots=True)
class CaptureStatus:
    """Health snapshot for a capture source."""

    source_name: str
    healthy: bool
    opened: bool
    running: bool
    last_timestamp: float | None = None
    dropped_count: int = 0
    failure_count: int = 0
    error_message: str | None = None
