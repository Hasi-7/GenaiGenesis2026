"""Record a live camera+mic session to disk for later replay.

Usage::

    uv run python server/scripts/record_sample.py [session_name] [--duration 30]
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

# Add server to path for sibling imports.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.third_party import load_cv2
from input.camera_adapter import LocalCameraAdapter
from input.mic_adapter import LocalMicAdapter

cv2 = load_cv2()


def main() -> None:
    parser = argparse.ArgumentParser(description="Record camera+mic sample session")
    parser.add_argument("session_name", nargs="?", default="default")
    parser.add_argument("--duration", type=float, default=30.0, help="Max seconds")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--fps", type=int, default=15, help="Target FPS")
    args = parser.parse_args()

    out_dir = Path(__file__).resolve().parent.parent / "samples" / args.session_name
    out_dir.mkdir(parents=True, exist_ok=True)

    camera = LocalCameraAdapter(args.camera)
    if not camera.is_opened():
        print(f"Cannot open camera {args.camera}", file=sys.stderr)
        return

    mic = LocalMicAdapter(sample_rate=16_000, channels=1, blocksize=1_600)

    width = camera.frame_width
    height = camera.frame_height
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    video_path = str(out_dir / "video.avi")
    writer = cv2.VideoWriter(video_path, fourcc, args.fps, (width, height))

    audio_chunks: list[NDArray[np.float32]] = []

    def _on_audio(chunk: NDArray[np.float32]) -> None:
        audio_chunks.append(chunk.copy())

    mic.subscribe(_on_audio)

    stop = False

    def _handle_signal(signum: int, _frame: object) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    mic.start()
    frame_interval = 1.0 / args.fps
    start_time = time.monotonic()
    frame_count = 0

    print(
        f"Recording to {out_dir} (max {args.duration}s, {args.fps} fps). Press Ctrl+C to stop."
    )

    try:
        while not stop:
            elapsed = time.monotonic() - start_time
            if elapsed >= args.duration:
                break

            frame = camera.read_frame()
            if frame is not None:
                writer.write(frame)
                frame_count += 1

            next_time = start_time + frame_count * frame_interval
            sleep_time = next_time - time.monotonic()
            if sleep_time > 0:
                time.sleep(sleep_time)
    finally:
        mic.stop()
        writer.release()
        camera.release()

    # Flush audio to WAV
    if audio_chunks:
        import soundfile as sf

        audio = np.concatenate(audio_chunks, axis=0)
        if audio.ndim > 1:
            audio = audio[:, 0]
        audio_path = str(out_dir / "audio.wav")
        sf.write(audio_path, audio, 16_000, subtype="FLOAT")
        print(f"Saved {len(audio)} audio samples to {audio_path}")
    else:
        print("No audio captured.")

    # Write meta.json
    meta = {
        "fps": args.fps,
        "width": width,
        "height": height,
        "sample_rate": 16_000,
        "chunk_size": 1_600,
        "duration_seconds": round(time.monotonic() - start_time, 2),
        "frame_count": frame_count,
    }
    meta_path = out_dir / "meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Done. {frame_count} frames, meta written to {meta_path}")


if __name__ == "__main__":
    main()
