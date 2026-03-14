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
uv --directory server run python main.py server
```

From inside `server/`:

```powershell
uv run cognitivesense
uv run cognitivesense mirror
uv run python main.py server
```

### Checks

```powershell
uv --directory server run pyright .
uv --directory server run ruff check .
```

### Ports

- Desktop Electron bridge mode (`python main.py server`) listens on `9100` by default.
- Raspberry Pi mirror mode (`cognitivesense mirror`) listens on `9000` by default.
- Override them independently with `COGNITIVESENSE_SERVER_PORT` and `COGNITIVESENSE_MIRROR_PORT`.

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

If you run the Python server inside WSL2, Raspberry Pi devices on your LAN usually
cannot connect to it directly through the Windows Wi-Fi IP until Windows forwards
the port into WSL. Run this once from an elevated Windows PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_wsl_mirror_proxy.ps1 -Port 9000
```

Then point the Raspberry Pi sender at the Windows LAN IP that the script prints.

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
