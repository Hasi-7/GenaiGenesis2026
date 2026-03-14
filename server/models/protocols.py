from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from models.types import (
    BlinkData,
    ClassifierResult,
    CognitiveState,
    FrameAnalysis,
    GazeData,
    LLMRequest,
    LLMResponse,
    PostureData,
    StateTransition,
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
# Vision Layer
# ---------------------------------------------------------------------------


class FaceLandmarkSource(Protocol):
    """Protocol for face landmark detection (MediaPipe Face Mesh)."""

    def detect(self, frame_rgb: NDArray[np.uint8]) -> list[NDArray[np.float32]] | None:
        """
        Detect faces and return landmark arrays.

        Each array has shape (478, 3) with normalized (x, y, z) coords.
        Returns None if no face is detected.
        """
        ...

    def close(self) -> None:
        """Release detector resources."""
        ...


class BlinkDetectorProtocol(Protocol):
    """Protocol for blink detection via Eye Aspect Ratio."""

    def detect(self, landmarks: NDArray[np.float32]) -> BlinkData:
        """Compute EAR values and detect blinks from face landmarks."""
        ...

    def classify(self, blink_data: BlinkData) -> ClassifierResult:
        """
        Classify fatigue based on blink rate.

        Labels: "normal" (15-20/min), "fatigued" (>25/min), "stressed" (<10/min).
        """
        ...

    def reset(self) -> None:
        """Reset blink counter and frame history."""
        ...


class EyeMovementDetectorProtocol(Protocol):
    """Protocol for gaze direction estimation from iris landmarks."""

    def detect(self, landmarks: NDArray[np.float32]) -> GazeData:
        """
        Compute gaze direction from iris position relative to eye contours.

        Uses iris landmarks 468 (left center) and 473 (right center).
        """
        ...

    def classify(self, gaze_data: GazeData) -> ClassifierResult:
        """
        Classify attention based on gaze stability.

        Labels: "focused" (sustained center), "distracted" (frequent off-center).
        """
        ...


class ExpressionClassifierProtocol(Protocol):
    """Protocol for facial expression classification from blendshapes."""

    def classify(self, blendshapes: dict[str, float]) -> ClassifierResult:
        """
        Classify facial expression from MediaPipe blendshape coefficients.

        Labels: "neutral", "tense", "relaxed".
        """
        ...


class PostureDetectorProtocol(Protocol):
    """Protocol for posture detection via MediaPipe Pose."""

    def detect(self, frame_rgb: NDArray[np.uint8]) -> PostureData | None:
        """
        Detect pose and compute posture metrics.

        Uses shoulder landmarks (11, 12), ear landmarks (7, 8), nose (0).
        Returns None if no pose is detected.
        """
        ...

    def classify(self, posture_data: PostureData) -> ClassifierResult:
        """
        Classify posture quality.

        Labels: "upright", "slouching" (shoulder deviation >15deg),
                "leaning" (head tilt >20deg).
        """
        ...

    def close(self) -> None:
        """Release detector resources."""
        ...


# ---------------------------------------------------------------------------
# Audio Layer
# ---------------------------------------------------------------------------


class SpeechToneClassifierProtocol(Protocol):
    """Protocol for speech tone classification from audio."""

    def classify(
        self,
        audio_chunk: NDArray[np.float32],
        sample_rate: int = ...,
    ) -> ClassifierResult:
        """
        Classify tone from audio features.

        Labels: "calm", "stressed", "monotone", "silent".
        """
        ...


# ---------------------------------------------------------------------------
# State Layer
# ---------------------------------------------------------------------------


class StateTrackerProtocol(Protocol):
    """Protocol for sliding window state tracking and change detection."""

    def add_frame(self, analysis: FrameAnalysis) -> None:
        """Add a frame analysis to the sliding window."""
        ...

    def get_current_state(self) -> CognitiveState:
        """
        Compute smoothed cognitive state from the sliding window.

        Uses weighted voting across all signals in the window.
        """
        ...

    def detect_transition(self) -> StateTransition | None:
        """
        Compare first half (5s) and second half (5s) of the window.

        Returns StateTransition if:
        - Dominant label differs between halves
        - New confidence > confidence_threshold (0.6)
        - Confidence delta > state_change_threshold (0.3)
        - Both halves are sufficiently populated

        Returns None otherwise.
        """
        ...

    def get_recent_analyses(self, seconds: float = ...) -> list[FrameAnalysis]:
        """Return frame analyses from the last N seconds."""
        ...


# ---------------------------------------------------------------------------
# Reasoning Layer
# ---------------------------------------------------------------------------


class ReasoningEngine(Protocol):
    """Protocol for LLM-based cognitive feedback with rate limiting."""

    def request_feedback(self, request: LLMRequest) -> LLMResponse | None:
        """
        Send state transition to LLM for analysis.

        Returns None if rate limited (cooldown not elapsed) or
        budget exceeded ($5 cumulative cap).
        """
        ...


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
    ) -> None:
        """Print state to terminal, throttled to 1 update/second."""
        ...

    def should_quit(self) -> bool:
        """Check for quit signal."""
        ...

    def destroy(self) -> None:
        """Clean up."""
        ...
