from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import Protocol

from audio.speech_tone_classifier import SpeechToneClassifier
from dotenv import load_dotenv
from input.mic_adapter import LocalMicAdapter
from input.screenshot_manager import ScreenshotManager
from models.protocols import MicSource
from models.types import (
    CognitiveState,
    FrameAnalysis,
    LLMRequest,
    LLMResponse,
    PipelineConfig,
)
from reasoning.llm_engine import LLMEngine
from state.state_tracker import StateTrackerProtocol
from ui.desktop_ui import DesktopUI
from ui.mirror_ui import MirrorUI
from vision.blink_detector import EarBlinkDetector
from vision.eye_movement_detector import IrisGazeDetector
from vision.face_landmarks import FaceLandmarkerTask
from vision.facial_expression_classifier import (
    BlendshapeExpressionClassifier,
)
from vision.posture_detector import MediaPipePostureDetector

load_dotenv()

logger = logging.getLogger(__name__)


class PipelineControllerProtocol(Protocol):
    """Protocol for the pipeline controller."""

    def subscribe(self, callback: Callable[[LLMResponse], None]) -> None: ...

    def run(self) -> None: ...


class TelemetrySinkProtocol(Protocol):
    def publish_state(self, state: CognitiveState, analysis: FrameAnalysis) -> None: ...

    def publish_feedback(
        self, response: LLMResponse, state: CognitiveState
    ) -> None: ...


class PipelineController:
    """Orchestrates input -> vision/audio -> state -> reasoning."""

    def __init__(
        self,
        config: PipelineConfig,
        llm_engine: LLMEngine,
        screenshot_manager: ScreenshotManager,
        state_tracker: StateTrackerProtocol,
        mic_source: MicSource | None = None,
        telemetry_sink: TelemetrySinkProtocol | None = None,
    ) -> None:
        self._config = config
        self._llm_engine = llm_engine
        self._screenshot_manager = screenshot_manager
        self._state_tracker = state_tracker

        self._mic: MicSource | None = mic_source
        if self._mic is None and config.mic_enabled:
            self._mic = LocalMicAdapter()
        self._telemetry_sink = telemetry_sink

        # Vision
        self._face_detector = FaceLandmarkerTask()
        self._blink_detector = EarBlinkDetector(
            ear_threshold=config.ear_blink_threshold,
            consecutive_frames=config.blink_consec_frames,
        )
        self._gaze_detector = IrisGazeDetector()
        self._expression_classifier = BlendshapeExpressionClassifier()
        self._posture_detector = MediaPipePostureDetector()

        # Audio
        self._speech_classifier: SpeechToneClassifier | None = None
        if config.mic_enabled:
            self._speech_classifier = SpeechToneClassifier()

        # Subscribers
        self._subscribers: list[Callable[[LLMResponse], None]] = []

        # Latest LLM response for UI
        self._last_response: LLMResponse | None = None
        self._waiting_for_frames_since: float | None = None
        self._last_waiting_log_time = 0.0
        self._stop_event = threading.Event()
        self._cleanup_lock = threading.Lock()
        self._closed = False

        # UI renderer
        if not config.renderer_enabled:
            self._renderer: DesktopUI | MirrorUI | None = None
        elif config.environment.value == "desktop":
            self._renderer = DesktopUI()
        else:
            self._renderer = MirrorUI()

    def subscribe(self, callback: Callable[[LLMResponse], None]) -> None:
        """Register a callback invoked on each LLM response."""
        self._subscribers.append(callback)

    def request_stop(self) -> None:
        """Ask the main loop to exit at the next safe checkpoint."""
        self._stop_event.set()

    def close(self) -> None:
        """Release pipeline resources. Safe to call multiple times."""
        self._cleanup()

    def run(self) -> None:
        """Run the main pipeline loop. Blocks until stopped."""
        if not self._screenshot_manager.is_opened():
            logger.error("Camera failed to open")
            return

        if self._mic is not None:
            self._mic.start()

        frame_interval = 1.0 / self._config.target_fps
        logger.info(
            "Pipeline running at %d FPS in %s mode",
            self._config.target_fps,
            self._config.environment.value,
        )

        try:
            while not self._stop_event.is_set():
                t_start = time.monotonic()
                self._tick()
                elapsed = time.monotonic() - t_start
                sleep_time = frame_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                if self._renderer is not None and self._renderer.should_quit():
                    logger.info("User quit via UI")
                    self.request_stop()
        except KeyboardInterrupt:
            logger.info("Pipeline interrupted")
            self.request_stop()
        finally:
            self.close()

    def _tick(self) -> None:
        # 1. Read frame via screenshot manager
        self._screenshot_manager.tick()
        bgr_frame = self._screenshot_manager.bgr_frame
        if bgr_frame is None:
            self._log_waiting_for_frames()
            return
        self._log_frame_stream_resumed()

        logger.debug(
            "_tick: read_frame %dx%d",
            bgr_frame.shape[1],
            bgr_frame.shape[0],
        )

        rgb_frame = self._screenshot_manager.rgb_frame
        if rgb_frame is None:
            return
        logger.debug("_tick: cvtColor BGR->RGB done")

        now = time.time()
        analysis = FrameAnalysis(timestamp=now)

        # 2. Face landmarks
        faces = self._face_detector.detect(rgb_frame)
        logger.debug("_tick: face_detector.detect -> %s", "found" if faces else "none")
        if faces:
            landmarks = faces[0]
            logger.info("_tick: face detected, %d landmarks", len(landmarks))

            # 3. Blink
            blink_data = self._blink_detector.detect(landmarks)
            analysis.blink = blink_data
            analysis.blink_label = self._blink_detector.classify(blink_data)
            logger.info(
                "_tick: blink=%s conf=%.2f ear=%.3f bpm=%.1f",
                analysis.blink_label.label,
                analysis.blink_label.confidence,
                blink_data.ear_average,
                blink_data.blinks_per_minute,
            )

            # 4. Gaze
            gaze_data = self._gaze_detector.detect(landmarks)
            analysis.gaze = gaze_data
            analysis.gaze_label = self._gaze_detector.classify(gaze_data)
            logger.info(
                "_tick: gaze=%s conf=%.2f direction=%s h=%.2f v=%.2f",
                analysis.gaze_label.label,
                analysis.gaze_label.confidence,
                gaze_data.direction,
                gaze_data.horizontal_ratio,
                gaze_data.vertical_ratio,
            )

            # 5. Expression (blendshape-based)
            blendshapes = self._face_detector.last_blendshapes
            logger.debug(
                "_tick: last_blendshapes -> %s",
                f"{len(blendshapes)} faces" if blendshapes else "none",
            )
            if blendshapes:
                expression = self._expression_classifier.classify(blendshapes[0])
                analysis.expression = expression
                logger.info(
                    "_tick: expression=%s conf=%.2f",
                    expression.label,
                    expression.confidence,
                )

        # 6. Posture
        posture_data = self._posture_detector.detect(rgb_frame)
        logger.debug(
            "_tick: posture_detector.detect -> %s",
            f"angle={posture_data.shoulder_angle:.1f} tilt={posture_data.head_tilt:.1f}"
            if posture_data
            else "none",
        )
        if posture_data is not None:
            analysis.posture = self._posture_detector.classify(posture_data)
            logger.info(
                "_tick: posture=%s conf=%.2f",
                analysis.posture.label,
                analysis.posture.confidence,
            )

        # 7. Audio
        if self._mic is not None:
            chunk = self._mic.get_latest_chunk()
            logger.debug(
                "_tick: mic.get_latest_chunk -> %s",
                f"{chunk.shape[0]} samples" if chunk is not None else "none",
            )
            if chunk is not None and self._speech_classifier is not None:
                analysis.speech_tone = self._speech_classifier.classify(chunk)
                logger.info(
                    "_tick: speech_tone=%s conf=%.2f",
                    analysis.speech_tone.label,
                    analysis.speech_tone.confidence,
                )

        # 8. State tracking
        self._state_tracker.add_frame(analysis)
        logger.debug("_tick: add_frame done")
        current_state = self._state_tracker.get_current_state()
        logger.info(
            "_tick: state=%s conf=%.2f",
            current_state.label.value,
            current_state.confidence,
        )
        transition = self._state_tracker.detect_transition()
        logger.debug(
            "_tick: detect_transition -> %s",
            "transition" if transition else "none",
        )
        if self._telemetry_sink is not None:
            self._telemetry_sink.publish_state(current_state, analysis)

        # 9. LLM reasoning on state transition
        if transition is not None:
            logger.info(
                "_tick: transition %s -> %s",
                transition.previous_state.label.value,
                transition.new_state.label.value,
            )
            jpeg_bytes = self._screenshot_manager.encode_jpeg()
            logger.debug("_tick: encode_jpeg -> %d bytes", len(jpeg_bytes))
            request = LLMRequest(
                frame_jpeg_bytes=jpeg_bytes,
                current_state=current_state,
                transition=transition,
                recent_analyses=(self._state_tracker.get_recent_analyses(5.0)),
            )
            response = self._llm_engine.request_feedback(request)
            logger.debug(
                "_tick: request_feedback -> %s",
                "response" if response else "rate-limited",
            )
            if response is not None:
                self._last_response = response
                logger.info(
                    "_tick: LLM response emitted, notifying %d subscribers",
                    len(self._subscribers),
                )
                print(response.feedback_text)
                self._notify_subscribers(response)
                if self._telemetry_sink is not None:
                    self._telemetry_sink.publish_feedback(response, current_state)

        # 10. Render UI
        if isinstance(self._renderer, DesktopUI):
            self._renderer.render(
                bgr_frame,
                current_state,
                self._last_response,
                analysis,
            )
        elif self._renderer is not None:
            self._renderer.render(
                current_state,
                self._last_response,
                analysis,
            )

    def _notify_subscribers(self, response: LLMResponse) -> None:
        for callback in self._subscribers:
            try:
                callback(response)
            except Exception:
                logger.exception("Response subscriber raised an exception")

    def _cleanup(self) -> None:
        with self._cleanup_lock:
            if self._closed:
                return
            self._closed = True

        self._stop_event.set()

        if self._mic is not None:
            try:
                self._mic.stop()
            except Exception:
                logger.exception("Failed to stop microphone during cleanup")
        try:
            self._screenshot_manager.release()
        except Exception:
            logger.exception("Failed to release screenshot manager during cleanup")
        try:
            self._face_detector.close()
        except Exception:
            logger.exception("Failed to close face detector during cleanup")
        try:
            self._posture_detector.close()
        except Exception:
            logger.exception("Failed to close posture detector during cleanup")
        if self._renderer is not None:
            try:
                self._renderer.destroy()
            except Exception:
                logger.exception("Failed to destroy renderer during cleanup")

    def _log_waiting_for_frames(self) -> None:
        now = time.monotonic()
        if self._waiting_for_frames_since is None:
            self._waiting_for_frames_since = now

        if now - self._last_waiting_log_time < 5.0:
            return

        waited = now - self._waiting_for_frames_since
        if self._config.environment.value == "mirror":
            logger.info(
                "Waiting for mirror frames on %s:%d (%.1fs elapsed)",
                self._config.mirror_listen_host,
                self._config.mirror_listen_port,
                waited,
            )
        else:
            logger.info(
                "Waiting for local camera frames (%.1fs elapsed)",
                waited,
            )
        self._last_waiting_log_time = now

    def _log_frame_stream_resumed(self) -> None:
        if self._waiting_for_frames_since is None:
            return

        waited = time.monotonic() - self._waiting_for_frames_since
        logger.info("Frame stream became available after %.1fs", waited)
        self._waiting_for_frames_since = None
        self._last_waiting_log_time = 0.0


if __name__ == "__main__":
    from config.logging_config import setup_logging
    from input.camera_adapter import LocalCameraAdapter
    from openai import OpenAI
    from reasoning.llm_engine import RateLimiter
    from state.state_tracker import StateTracker

    setup_logging()

    config = PipelineConfig.desktop()
    client = OpenAI()
    engine = LLMEngine(
        client=client,
        rate_limiter=RateLimiter(
            cooldown_seconds=config.llm_cooldown_seconds,
        ),
    )

    camera = LocalCameraAdapter(config.camera_index)
    screenshots = ScreenshotManager(camera)
    tracker = StateTracker(window_seconds=config.smoothing_window_seconds)

    controller = PipelineController(
        config=config,
        llm_engine=engine,
        screenshot_manager=screenshots,
        state_tracker=tracker,
    )
    controller.run()
