## Server

This package uses `uv` for dependency management and command execution.

### Setup

```powershell
uv sync --directory server --all-groups
```

### Run

From the repo root:

```powershell
uv --directory server run cognitivesense
uv --directory server run cognitivesense mirror
```

From inside `server/`:

```powershell
uv run cognitivesense
uv run cognitivesense mirror
```

### Checks

```powershell
uv --directory server run pyright .
uv --directory server run ruff check .
```

## Mirror Smoke Test

On the Raspberry Pi, build the native capture smoke test from the repo root:

```bash
cmake -S mirror -B mirror/build
cmake --build mirror/build
```

Run it directly against the Pi camera:

```bash
./mirror/build/mirror_capture_smoke
./mirror/build/mirror_capture_smoke 1280 720 8000
./mirror/build/mirror_capture_smoke 1280 720 8000 /home/pi/frame.bmp
./mirror/build/mirror_capture_smoke stream 192.168.1.10
./mirror/build/mirror_capture_smoke stream 192.168.1.10 9000 640 480 15
```

Capture arguments are `width height timeout_ms [output_path]`. The smoke test prints one captured frame, saves a BMP to `captured_frame.bmp` by default, and exits with `0` on success.

Streaming arguments are `stream <server_ip> [port] [width] [height] [fps]`. This launches the continuous 15 FPS network stream test through the built `mirror_frame_streamer` binary.

## Mirror Streaming

Run the server in mirror mode on the receiving machine:

```bash
uv run cognitivesense mirror
```

Build the mirror streamer on the Raspberry Pi:

```bash
cmake -S mirror -B mirror/build
cmake --build mirror/build
```

Then stream frames to the server:

```bash
./mirror/build/mirror_frame_streamer 192.168.1.10
./mirror/build/mirror_frame_streamer 192.168.1.10 9000 640 480 15
```

Arguments are `server_ip [port] [width] [height] [fps]`. The sender uses `rpicam-vid` when available and sends MJPEG frames over TCP to the Python mirror receiver listening on port `9000` by default.
