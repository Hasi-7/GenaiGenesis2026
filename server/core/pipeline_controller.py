from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Protocol

from dotenv import load_dotenv
load_dotenv()

import os

from audio.speech_tone_classifier import SpeechToneClassifier
from input.mic_adapter import LocalMicAdapter
from input.screenshot_manager import ScreenshotManager
from models.types import (
    FrameAnalysis,
    LLMRequest,
    LLMResponse,
    PipelineConfig,
)
from reasoning.llm_engine import LLMEngine
from state.state_tracker import StateTrackerProtocol
from vision.blink_detector import EarBlinkDetector
from vision.eye_movement_detector import IrisGazeDetector
from vision.face_landmarks import FaceLandmarkerTask
from vision.facial_expression_classifier import (
    BlendshapeExpressionClassifier,
)
from vision.posture_detector import MediaPipePostureDetector
from ui.desktop_ui import DesktopUI

logger = logging.getLogger(__name__)


class PipelineControllerProtocol(Protocol):
    """Protocol for the pipeline controller."""

    def subscribe(
        self, callback: Callable[[LLMResponse], None]
    ) -> None: ...

    def run(self) -> None: ...


class PipelineController:
    """Orchestrates input -> vision/audio -> state -> reasoning."""

    def __init__(
        self,
        config: PipelineConfig,
        llm_engine: LLMEngine,
        screenshot_manager: ScreenshotManager,
        state_tracker: StateTrackerProtocol,
    ) -> None:
        self._config = config
        self._llm_engine = llm_engine
        self._screenshot_manager = screenshot_manager
        self._state_tracker = state_tracker

        # Input
        self._mic: LocalMicAdapter | None = None
        if config.mic_enabled:
            self._mic = LocalMicAdapter()

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
        self._speech_classifier = SpeechToneClassifier()

        # Subscribers
        self._subscribers: list[Callable[[LLMResponse], None]] = []

        # Latest LLM response for UI
        self._last_response: LLMResponse | None = None

        # UI renderer
        self._renderer: DesktopUI | None = (
            DesktopUI() if config.environment.value == "desktop" else None
        )

    def subscribe(
        self, callback: Callable[[LLMResponse], None]
    ) -> None:
        """Register a callback invoked on each LLM response."""
        self._subscribers.append(callback)

    def run(self) -> None:
        """Run the main pipeline loop. Blocks until stopped."""
        if not self._screenshot_manager.is_opened():
            logger.error("Camera failed to open")
            return

        if self._mic is not None:
            self._mic.start()

        frame_interval = 1.0 / self._config.target_fps

        try:
            while True:
                t_start = time.monotonic()
                self._tick()
                elapsed = time.monotonic() - t_start
                sleep_time = frame_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                if self._renderer is not None and self._renderer.should_quit():
                    logger.info("User quit via UI")
                    break
        except KeyboardInterrupt:
            logger.info("Pipeline interrupted")
        finally:
            self._cleanup()

    def _tick(self) -> None:
        # 1. Read frame via screenshot manager
        self._screenshot_manager.tick()
        bgr_frame = self._screenshot_manager.bgr_frame
        if bgr_frame is None:
            return

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
            logger.info(
                "_tick: blink ear=%.3f bpm=%.1f",
                blink_data.ear_average,
                blink_data.blinks_per_minute,
            )

            # 4. Gaze
            gaze_data = self._gaze_detector.detect(landmarks)
            analysis.gaze = gaze_data
            logger.info(
                "_tick: gaze direction=%s h=%.2f v=%.2f",
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
                expression = self._expression_classifier.classify(
                    blendshapes[0]
                )
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
            analysis.posture = self._posture_detector.classify(
                posture_data
            )
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
            if chunk is not None:
                analysis.speech_tone = (
                    self._speech_classifier.classify(chunk)
                )
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

        # 9. LLM reasoning on state transition
        if transition is not None:
            logger.info(
                "_tick: transition %s -> %s",
                transition.previous_state.label.value,
                transition.new_state.label.value,
            )
            jpeg_bytes = self._screenshot_manager.encode_jpeg()
            logger.debug(
                "_tick: encode_jpeg -> %d bytes", len(jpeg_bytes)
            )
            request = LLMRequest(
                frame_jpeg_bytes=jpeg_bytes,
                current_state=current_state,
                transition=transition,
                recent_analyses=(
                    self._state_tracker.get_recent_analyses(5.0)
                ),
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

        # 10. Render UI
        if self._renderer is not None:
            self._renderer.render(
                bgr_frame,
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
        if self._mic is not None:
            self._mic.stop()
        self._screenshot_manager.release()
        self._face_detector.close()
        self._posture_detector.close()
        if self._renderer is not None:
            self._renderer.destroy()


if __name__ == "__main__":
    from openai import OpenAI
    from input.camera_adapter import LocalCameraAdapter
    from reasoning.llm_engine import RateLimiter
    from state.state_tracker import StateTracker

    from config.logging_config import setup_logging

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
