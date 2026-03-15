## CognitiveSense

This repo currently ships as three pieces:

1. `frontend/` = the Electron desktop client
2. `server/` = the Python analysis backend and control-plane API
3. `mirror/` = the Raspberry Pi native streamer

The intended v1 deployment model is:

- ship the Electron app to Windows / macOS / Linux users
- run the Python backend remotely on a machine you control
- use the built-in control website for registration, device claiming, and period scores

## Fastest Way To Run It Now

### 1. Start the remote backend and website

From the repo root:

```bash
docker compose up --build
```

This starts:

- analysis media ingress on `9000`
- control website + API on `8080`

Open the website at:

```text
http://127.0.0.1:8080/
```

There you can:

- register or log in
- see unclaimed devices
- claim a desktop or mirror device to your identity directly from the unclaimed-device list
- view aggregate scores and timeline data

### 2. Run the desktop app locally against that backend

```bash
cd frontend
npm ci
npm run dev
```

The desktop app is a thin client. It captures local camera/mic and streams to the backend on `9000`.

### 3. Package desktop installers

```bash
cd frontend
npm ci
npm run dist
```

Artifacts are written to `frontend/release/`.

Targets currently configured:

- Windows: `nsis`
- macOS: `dmg`
- Linux: `AppImage` and `deb`

## Runtime Configuration

The Electron app stores runtime connection info in a per-user `client-config.json` file inside the Electron user-data directory.

Stored fields:

- `mediaHost`
- `mediaPort`
- `apiBaseUrl`
- `deviceId`
- `deviceName`

This is how a shipped desktop build points at a remote backend without being rebuilt.

Default values:

- media host: `127.0.0.1`
- media port: `9000`
- control API URL: `http://127.0.0.1:8080`

## Local Python Development

This repo uses `uv` for Python dependency management and command execution.

### Setup

```bash
uv sync --all-groups
```

### Run the analysis server without Docker

From the repo root:

```bash
uv run python main.py server
```

Useful aliases:

```bash
uv run cognitivesense
uv run cognitivesense mirror
```

Current port behavior:

- `python main.py server` listens on `9000` for both Electron and Raspberry Pi clients
- `cognitivesense mirror` is an alias for the same unified network server
- override the media port with `COGNITIVESENSE_SERVER_PORT`

### Run the control API directly

```bash
uv run uvicorn server.control.api:app --host 0.0.0.0 --port 8080
```

### Checks

```bash
uv run pyright .
uv run ruff check .
```

## Auth And Device Association

Current model:

- each desktop or mirror device has a stable local `deviceId`
- the analysis backend records those devices automatically when they connect
- the control website shows them as unclaimed until a signed-in user claims them
- users do not need to manually retrieve the key from the app just to claim a device; the website exposes recent unclaimed devices and can claim them directly
- period scores aggregate across all devices claimed by that user

This is local/self-hosted v1 auth, not a production SaaS auth stack yet.

## Shipping And Deployment Docs

- desktop packaging and release notes: `DISTRIBUTION.md`
- remote backend deployment: `DEPLOYMENT.md`
- example deployment env file: `deploy.env.example`

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

Run the backend on the receiving machine:

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

To make a mirror device claimable from the website, either let it use its hostname as the device ID or set an explicit one:

```bash
COGNITIVESENSE_DEVICE_ID=bathroom-mirror ./mirror/build/mirror_frame_streamer 192.168.1.10
```
