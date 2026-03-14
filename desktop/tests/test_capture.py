from __future__ import annotations

import time
import unittest

from desktop.capture.camera_capture import CameraCapture
from desktop.capture.mic_capture import AudioStreamCallback, MicCapture
from desktop.capture.ring_buffer import LatestValueBuffer


class LatestValueBufferTests(unittest.TestCase):
    def test_latest_value_buffer_drops_stale_items(self) -> None:
        buffer: LatestValueBuffer[int] = LatestValueBuffer(capacity=2)

        buffer.push(1)
        buffer.push(2)
        buffer.push(3)

        self.assertEqual(buffer.get_latest(), 3)
        self.assertEqual(buffer.stats().dropped_count, 1)
        self.assertIsNone(buffer.get_latest())


class _ClosedCapture:
    def isOpened(self) -> bool:
        return False

    def read(self) -> tuple[bool, None]:
        return False, None

    def release(self) -> None:
        return None


class _FakeFrame:
    def __init__(self, byte_value: int) -> None:
        self.width = 2
        self.height = 2
        self._payload = bytes([byte_value] * 12)

    def tobytes(self) -> bytes:
        return self._payload


class _SequenceCapture:
    def __init__(self) -> None:
        self._frames: list[_FakeFrame] = [
            _FakeFrame(0),
            _FakeFrame(1),
            _FakeFrame(2),
        ]
        self._index = 0
        self._released = False

    def isOpened(self) -> bool:
        return True

    def read(self) -> tuple[bool, _FakeFrame | None]:
        if self._released:
            return False, None
        if self._index < len(self._frames):
            frame = self._frames[self._index]
            self._index += 1
            return True, frame
        time.sleep(0.01)
        return False, None

    def release(self) -> None:
        self._released = True


class CameraCaptureTests(unittest.TestCase):
    def test_camera_open_failure_sets_unavailable_status(self) -> None:
        def factory(_camera_index: int) -> _ClosedCapture:
            return _ClosedCapture()

        camera = CameraCapture(client_id="desktop-1", capture_factory=factory)

        camera.start()
        status = camera.get_status()

        self.assertFalse(camera.is_running)
        self.assertFalse(status.opened)
        self.assertEqual(status.error_message, "camera unavailable")

    def test_camera_returns_latest_frame(self) -> None:
        def factory(_camera_index: int) -> _SequenceCapture:
            return _SequenceCapture()

        camera = CameraCapture(
            client_id="desktop-1",
            capture_factory=factory,
            retry_interval_seconds=0.01,
        )

        camera.start()
        time.sleep(0.05)
        frame = camera.get_latest_frame()
        camera.stop()

        self.assertIsNotNone(frame)
        assert frame is not None
        self.assertEqual((frame.width, frame.height), (2, 2))
        self.assertEqual(frame.client_id, "desktop-1")
        self.assertEqual(frame.pixel_format, "bgr24")
        self.assertGreaterEqual(camera.get_status().failure_count, 0)


class _FakeSamples:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def tobytes(self) -> bytes:
        return self._payload


class _FakeStream:
    def __init__(self, callback: AudioStreamCallback) -> None:
        self._callback = callback
        self._closed = False
        self._started = False

    def start(self) -> None:
        self._started = True

    def stop(self) -> None:
        self._started = False

    def close(self) -> None:
        self._closed = True

    def emit(self, samples: _FakeSamples, frames: int) -> None:
        self._callback(samples, frames, None, None)


class MicCaptureTests(unittest.TestCase):
    def test_mic_capture_buffers_latest_chunk(self) -> None:
        created_stream: _FakeStream | None = None

        def factory(
            *,
            samplerate: int,
            channels: int,
            blocksize: int,
            dtype: str,
            callback: AudioStreamCallback,
        ) -> _FakeStream:
            del samplerate, channels, blocksize, dtype
            nonlocal created_stream
            created_stream = _FakeStream(callback)
            return created_stream

        mic = MicCapture(client_id="desktop-1", stream_factory=factory)

        mic.start()
        assert created_stream is not None
        created_stream.emit(_FakeSamples(b"\x00" * 6400), 1600)
        created_stream.emit(_FakeSamples(b"\x01" * 6400), 1600)
        chunk = mic.get_latest_audio_chunk()
        mic.stop()

        self.assertIsNotNone(chunk)
        assert chunk is not None
        self.assertEqual(chunk.client_id, "desktop-1")
        self.assertEqual(chunk.sample_rate, 16_000)
        self.assertEqual(chunk.sample_count, 1600)
        self.assertEqual(mic.get_status().dropped_count, 0)

    def test_mic_start_failure_sets_status(self) -> None:
        def factory(
            *,
            samplerate: int,
            channels: int,
            blocksize: int,
            dtype: str,
            callback: AudioStreamCallback,
        ) -> _FakeStream:
            del samplerate, channels, blocksize, dtype, callback
            raise RuntimeError("no device")

        mic = MicCapture(client_id="desktop-1", stream_factory=factory)

        mic.start()
        status = mic.get_status()

        self.assertFalse(mic.is_running)
        self.assertEqual(status.error_message, "no device")


if __name__ == "__main__":
    unittest.main()
