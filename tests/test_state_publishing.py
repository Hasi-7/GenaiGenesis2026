from __future__ import annotations

# ruff: noqa: E402
import struct
import socket
import sys
import time
import unittest
from contextlib import suppress
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock, patch

SERVER_ROOT = Path(__file__).resolve().parents[1] / "server"
if str(SERVER_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVER_ROOT))

from server.core.remote_session_runner import RemoteSessionRunner
from server.core.pipeline_controller import PipelineController
from server.core.feedback_codec import MAX_INDICATORS, MAX_RECOMMENDATIONS
from server.input.remote_media_server import RemoteClientSession
from server.models.types import (
    CognitiveState,
    CognitiveStateLabel,
    FrameAnalysis,
    StateTransition,
    PipelineConfig,
)

HEADER = struct.Struct("!4sIIIQ")
HELLO_PAYLOAD = struct.Struct("!BBH")
STATE_PAYLOAD = struct.Struct(f"!BBBBI{MAX_INDICATORS}B{MAX_RECOMMENDATIONS}B")
EVENT_MAGIC = b"CSM1"
EVENT_HELLO = 1
EVENT_STATE = 2
HELLO_VERSION = 1
SOURCE_DESKTOP = 1
CAPABILITY_RECEIVE_STATE = 1 << 2


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    buffer = bytearray()
    while len(buffer) < size:
        chunk = sock.recv(size - len(buffer))
        if not chunk:
            raise RuntimeError("socket closed before packet was fully received")
        buffer.extend(chunk)
    return bytes(buffer)


def _send_control_packet(
    sock: socket.socket,
    message_type: int,
    flags: int,
    payload: bytes,
) -> None:
    packet = HEADER.pack(EVENT_MAGIC, message_type, flags, len(payload), 0) + payload
    sock.sendall(packet)


def _wait_until(predicate: object, timeout: float = 1.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if callable(predicate) and predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition was not met before timeout")


class RemoteClientSessionTests(unittest.TestCase):
    def setUp(self) -> None:
        client_socket, server_socket = socket.socketpair()
        self.client_socket = client_socket
        self.server_socket = server_socket
        self.client_socket.settimeout(1.0)
        self.session = RemoteClientSession(
            7,
            self.server_socket,
            ("127.0.0.1", 54321),
            on_disconnect=lambda _session: None,
        )

    def tearDown(self) -> None:
        self.session.close(wait=False)
        self.session.join(timeout=1.0)
        with suppress(OSError):
            self.client_socket.close()

    def test_extended_hello_enables_state_telemetry_and_identity(self) -> None:
        self.session.start()
        payload = HELLO_PAYLOAD.pack(HELLO_VERSION, SOURCE_DESKTOP, 0) + (
            b"device-123|Desk Monitor"
        )
        _send_control_packet(
            self.client_socket,
            EVENT_HELLO,
            CAPABILITY_RECEIVE_STATE,
            payload,
        )

        _wait_until(lambda: self.session.has_state_telemetry)

        self.assertTrue(self.session.has_state_telemetry)
        self.assertEqual(self.session.source_name, "desktop")
        self.assertEqual(self.session.device_key, "device-123")
        self.assertEqual(self.session.display_name, "Desk Monitor")
        self.assertEqual(self.session.remote_ip, "127.0.0.1")

    def test_publish_state_sends_packet_after_extended_hello(self) -> None:
        self.session.start()
        payload = HELLO_PAYLOAD.pack(HELLO_VERSION, SOURCE_DESKTOP, 0) + (
            b"device-123|Desk Monitor"
        )
        _send_control_packet(
            self.client_socket,
            EVENT_HELLO,
            CAPABILITY_RECEIVE_STATE,
            payload,
        )
        _wait_until(lambda: self.session.has_state_telemetry)

        state = CognitiveState(
            label=CognitiveStateLabel.FOCUSED,
            confidence=0.8,
            contributing_signals=[],
            timestamp=123.0,
        )

        self.session.publish_state(state, FrameAnalysis(timestamp=123.0))

        header = _recv_exact(self.client_socket, HEADER.size)
        magic, meta1, meta2, payload_size, _timestamp_ns = HEADER.unpack(header)
        packet_payload = _recv_exact(self.client_socket, payload_size)

        self.assertEqual(magic, EVENT_MAGIC)
        self.assertEqual(meta1, EVENT_STATE)
        self.assertEqual(meta2, 0)
        self.assertEqual(payload_size, STATE_PAYLOAD.size)
        self.assertEqual(packet_payload[1], 1)


class RemoteSessionRunnerTests(unittest.TestCase):
    def test_remote_runner_combines_session_and_recorder_telemetry(self) -> None:
        fake_settings = SimpleNamespace(
            llm_state_tracker_cooldown=10.0,
            llm_state_tracker_min_interval=5.0,
            llm_state_tracker_model="gpt-4.1-mini",
            llm_feedback_model="gpt-4.1-mini",
        )
        session = cast(
            RemoteClientSession,
            SimpleNamespace(
                session_id=7,
                session_label="desktop:7@127.0.0.1:54321",
                source_name="desktop",
                device_key="device-123",
                display_name="Desk Monitor",
                remote_ip="127.0.0.1",
                make_camera_source=lambda: object(),
                make_mic_source=lambda: object(),
            ),
        )

        with (
            patch(
                "server.core.remote_session_runner.get_settings",
                return_value=fake_settings,
            ),
            patch(
                "server.core.remote_session_runner.get_control_store",
                return_value=object(),
            ),
            patch(
                "server.core.remote_session_runner.ScreenshotManager",
                return_value=object(),
            ),
            patch(
                "server.core.remote_session_runner.StateTracker",
                return_value=object(),
            ),
            patch("server.core.remote_session_runner.LLMEngine", return_value=object()),
            patch(
                "server.core.remote_session_runner.TelemetryRecorder"
            ) as telemetry_recorder,
            patch(
                "server.core.remote_session_runner.CompositeTelemetrySink"
            ) as composite_sink,
            patch(
                "server.core.remote_session_runner.PipelineController"
            ) as pipeline_controller,
        ):
            recorder = MagicMock()
            telemetry_recorder.return_value = recorder
            combined_sink = object()
            composite_sink.return_value = combined_sink
            controller = MagicMock()
            pipeline_controller.return_value = controller

            runner = RemoteSessionRunner(
                session,
                config=PipelineConfig.server(),
                tracker_type="rule",
                openai_client=None,
            )

            identity_provider = telemetry_recorder.call_args.args[1]
            identity = identity_provider()

            self.assertEqual(identity.device_key, "device-123")
            self.assertEqual(identity.display_name, "Desk Monitor")
            self.assertEqual(identity.source_kind, "desktop")
            self.assertEqual(identity.last_ip, "127.0.0.1")
            self.assertEqual(identity.transport_session_label, session.session_label)
            composite_sink.assert_called_once_with(session, recorder)
            self.assertIs(
                pipeline_controller.call_args.kwargs["telemetry_sink"],
                combined_sink,
            )

            runner.stop()

            controller.request_stop.assert_called_once_with()
            recorder.close.assert_called_once_with()


class PipelineControllerSnapshotTests(unittest.TestCase):
    def test_publish_snapshot_if_due_emits_jpeg(self) -> None:
        controller = PipelineController.__new__(PipelineController)
        telemetry_sink = MagicMock()
        screenshot_manager = MagicMock()
        screenshot_manager.encode_jpeg.return_value = b"jpeg-bytes"
        setattr(controller, "_telemetry_sink", telemetry_sink)
        setattr(controller, "_screenshot_manager", screenshot_manager)
        setattr(controller, "_last_snapshot_publish_at", 0.0)

        with patch("server.core.pipeline_controller.time.monotonic", return_value=10.0):
            getattr(controller, "_publish_snapshot_if_due")(recorded_at=123.0)

        telemetry_sink.publish_snapshot.assert_called_once_with(b"jpeg-bytes", 123.0)

    def test_publish_snapshot_if_due_respects_rate_limit(self) -> None:
        controller = PipelineController.__new__(PipelineController)
        telemetry_sink = MagicMock()
        screenshot_manager = MagicMock()
        screenshot_manager.encode_jpeg.return_value = b"jpeg-bytes"
        setattr(controller, "_telemetry_sink", telemetry_sink)
        setattr(controller, "_screenshot_manager", screenshot_manager)
        setattr(controller, "_last_snapshot_publish_at", 10.0)

        with patch("server.core.pipeline_controller.time.monotonic", return_value=10.5):
            getattr(controller, "_publish_snapshot_if_due")(recorded_at=124.0)

        telemetry_sink.publish_snapshot.assert_not_called()

    def test_request_transition_feedback_falls_back_when_llm_returns_none(self) -> None:
        controller = PipelineController.__new__(PipelineController)
        screenshot_manager = MagicMock()
        screenshot_manager.encode_jpeg.return_value = b"jpeg-bytes"
        state_tracker = MagicMock()
        state_tracker.get_recent_analyses.return_value = [FrameAnalysis(timestamp=123.0)]
        llm_engine = MagicMock()
        llm_engine.request_feedback.return_value = None

        setattr(controller, "_screenshot_manager", screenshot_manager)
        setattr(controller, "_state_tracker", state_tracker)
        setattr(controller, "_llm_engine", llm_engine)

        previous_state = CognitiveState(
            label=CognitiveStateLabel.FOCUSED,
            confidence=0.7,
            contributing_signals=[],
            timestamp=120.0,
        )
        current_state = CognitiveState(
            label=CognitiveStateLabel.STRESSED,
            confidence=0.9,
            contributing_signals=[],
            timestamp=123.0,
        )
        transition = StateTransition(
            previous_state=previous_state,
            new_state=current_state,
            transition_time=123.0,
        )

        response = getattr(controller, "_request_transition_feedback")(
            current_state,
            transition,
        )

        self.assertIsNotNone(response)
        self.assertEqual(response.trigger_kind, "transition")
        self.assertEqual(response.severity, "urgent")
        self.assertIn("slow breath", response.feedback_text)
        llm_engine.request_feedback.assert_called_once()


if __name__ == "__main__":
    unittest.main()
