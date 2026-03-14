from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class CognitiveStateLabel(Enum):
    FOCUSED = "focused"
    FATIGUED = "fatigued"
    STRESSED = "stressed"
    DISTRACTED = "distracted"
    UNKNOWN = "unknown"


class Environment(Enum):
    DESKTOP = "desktop"
    MIRROR = "mirror"
    SERVER = "server"


class InputSource(Enum):
    LOCAL_CAMERA = "local_camera"
    MIRROR_TCP = "mirror_tcp"
    REMOTE_MEDIA = "remote_media"


@dataclass(slots=True)
class ClassifierResult:
    """Universal output of every classifier module."""

    label: str
    confidence: float


@dataclass(slots=True)
class BlinkData:
    """Blink-specific data from blink detector."""

    ear_left: float
    ear_right: float
    ear_average: float
    blink_detected: bool
    blinks_per_minute: float


@dataclass(slots=True)
class GazeData:
    """Gaze direction from eye landmarks."""

    horizontal_ratio: float  # -1.0 (left) to 1.0 (right)
    vertical_ratio: float  # -1.0 (down) to 1.0 (up)
    direction: str  # "center", "left", "right", "up", "down"


@dataclass(slots=True)
class PostureData:
    """Posture classification from MediaPipe Pose."""

    shoulder_angle: float
    head_tilt: float
    is_slouching: bool


@dataclass(slots=True)
class FrameAnalysis:
    """Complete analysis of a single frame."""

    timestamp: float
    blink: BlinkData | None = None
    blink_label: ClassifierResult | None = None
    gaze: GazeData | None = None
    gaze_label: ClassifierResult | None = None
    expression: ClassifierResult | None = None
    posture: ClassifierResult | None = None
    speech_tone: ClassifierResult | None = None


@dataclass(slots=True)
class CognitiveState:
    """Aggregated cognitive state derived from smoothed window."""

    label: CognitiveStateLabel
    confidence: float
    contributing_signals: list[ClassifierResult]
    timestamp: float


@dataclass(slots=True)
class StateTransition:
    """Detected change in cognitive state."""

    previous_state: CognitiveState
    new_state: CognitiveState
    transition_time: float


@dataclass(slots=True)
class LLMRequest:
    """Data sent to the LLM for cognitive feedback."""

    frame_jpeg_bytes: bytes
    current_state: CognitiveState
    transition: StateTransition
    recent_analyses: list[FrameAnalysis]


@dataclass(slots=True)
class LLMResponse:
    """Response from LLM reasoning engine."""

    feedback_text: str
    timestamp: float


@dataclass(slots=True)
class CostTracker:
    """Tracks cumulative LLM spend."""

    total_spent_usd: float = 0.0
    cap_usd: float = 5.0
    call_count: int = 0
    last_call_timestamp: float = 0.0

    @property
    def budget_remaining(self) -> float:
        return self.cap_usd - self.total_spent_usd

    @property
    def is_budget_exceeded(self) -> bool:
        return self.total_spent_usd >= self.cap_usd


@dataclass(slots=True)
class PipelineConfig:
    """Configuration controlling pipeline behavior."""

    environment: Environment
    input_source: InputSource = InputSource.LOCAL_CAMERA
    camera_index: int = 0
    state_tracker_type: str = "rule"  # "rule" or "llm"
    mic_enabled: bool = True
    target_fps: int = 15
    remote_media_host: str = "0.0.0.0"
    remote_media_port: int = 9100
    renderer_enabled: bool = True
    mirror_listen_host: str = "0.0.0.0"
    mirror_listen_port: int = 9000
    smoothing_window_seconds: float = 10.0
    llm_cooldown_seconds: float = 30.0
    llm_cost_cap_usd: float = 5.0
    confidence_threshold: float = 0.6
    state_change_threshold: float = 0.3
    ear_blink_threshold: float = 0.21
    blink_consec_frames: int = 3

    @staticmethod
    def desktop() -> PipelineConfig:
        return PipelineConfig(
            environment=Environment.DESKTOP,
            input_source=InputSource.LOCAL_CAMERA,
            camera_index=0,
            mic_enabled=True,
        )

    @staticmethod
    def mirror() -> PipelineConfig:
        return PipelineConfig(
            environment=Environment.MIRROR,
            input_source=InputSource.MIRROR_TCP,
            camera_index=0,
            mic_enabled=False,
            target_fps=15,
            renderer_enabled=False,
            mirror_listen_host="0.0.0.0",
            mirror_listen_port=9000,
        )

    @staticmethod
    def server() -> PipelineConfig:
        return PipelineConfig(
            environment=Environment.SERVER,
            input_source=InputSource.REMOTE_MEDIA,
            camera_index=0,
            mic_enabled=True,
            target_fps=15,
            remote_media_host="0.0.0.0",
            remote_media_port=9100,
            renderer_enabled=False,
        )
