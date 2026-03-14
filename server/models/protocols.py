from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from models.types import (
    CognitiveState,
    FrameAnalysis,
    LLMResponse,
)

# ---------------------------------------------------------------------------
# Input Layer
# ---------------------------------------------------------------------------


class CameraSource(Protocol):
    """Protocol for camera input adapters (desktop webcam or Pi camera)."""

    def read_frame(self) -> NDArray[np.uint8] | None:
        """Return a BGR frame, or None if the read failed."""
        ...

    def release(self) -> None:
        """Release camera resources."""
        ...

    def is_opened(self) -> bool:
        """Whether the camera device is currently open."""
        ...

    @property
    def frame_width(self) -> int: ...

    @property
    def frame_height(self) -> int: ...


class MicSource(Protocol):
    """Protocol for microphone input adapters."""

    def start(self) -> None:
        """Start recording audio in a background thread."""
        ...

    def stop(self) -> None:
        """Stop recording and release resources."""
        ...

    def get_latest_chunk(self) -> NDArray[np.float32] | None:
        """Return the latest audio chunk (non-blocking), or None."""
        ...

    @property
    def is_recording(self) -> bool: ...


# ---------------------------------------------------------------------------
# UI Layer
# ---------------------------------------------------------------------------


class DesktopRenderer(Protocol):
    """Protocol for desktop UI rendering (OpenCV overlay on video)."""

    def render(
        self,
        frame: NDArray[np.uint8],
        state: CognitiveState,
        llm_response: LLMResponse | None = ...,
        analysis: FrameAnalysis | None = ...,
    ) -> None:
        """Overlay cognitive state info on frame and display via cv2.imshow."""
        ...

    def should_quit(self) -> bool:
        """Check if user pressed 'q' to quit."""
        ...

    def destroy(self) -> None:
        """Clean up windows."""
        ...


class MirrorRenderer(Protocol):
    """Protocol for mirror UI rendering (terminal text output)."""

    def render(
        self,
        state: CognitiveState,
        llm_response: LLMResponse | None = ...,
        analysis: FrameAnalysis | None = ...,
    ) -> None:
        """Print state to terminal, throttled to 1 update/second."""
        ...

    def should_quit(self) -> bool:
        """Check for quit signal."""
        ...

    def destroy(self) -> None:
        """Clean up."""
        ...
