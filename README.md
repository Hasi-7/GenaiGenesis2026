## CognitiveSense

CognitiveSense is a multi-client fatigue and cognitive-state monitoring project with:

1. `server/` for the Python analysis pipeline and control-plane API
2. `frontend/` for the Electron desktop client
3. `mirror/` for the Raspberry Pi native frame streamer

The current architecture is a thin-client model:

- desktop and mirror devices capture media locally
- both stream into the Python backend on TCP `9000`
- the backend runs state detection and LLM feedback generation
- the control website on `8080` handles registration, device claiming, snapshots, and timeline views

## Repo Layout

- `main.py`: top-level entrypoint for the Python app
- `server/`: analysis pipeline, remote media receiver, control API, storage, and UI helpers
- `frontend/`: Electron + React desktop shell
- `mirror/`: Raspberry Pi camera capture and frame streaming binaries
- `tests/`: Python regression tests
- `DEPLOYMENT.md`: backend deployment notes
- `DISTRIBUTION.md`: desktop packaging and release notes
- `scripts/setup_wsl_mirror_proxy.ps1`: Windows-to-WSL port forwarding helper for Raspberry Pi connectivity

## Prerequisites

### Backend

- Python `3.11`
- `uv`

### Desktop client

- Node.js `20+`
- npm

### Optional local deployment

- Docker
- Docker Compose

### Raspberry Pi mirror

- CMake `3.16+`
- a C compiler
- `rpicam-vid` available on the Pi path

## Environment Setup

The Python backend reads configuration from [`.env`](/mnt/c/Personal/VSCode/GenaiGenesis2026/.env).

Common variables:

- `OPENAI_API_KEY`: enables OpenAI-backed state tracking and feedback
- `COGNITIVESENSE_SERVER_HOST`: override backend bind host for remote media ingress
- `COGNITIVESENSE_SERVER_PORT`: override backend media port, default `9000`
- `COGNITIVESENSE_LOG_LEVEL`: logging level, default `DEBUG`
- `STATE_TRACKER_TYPE` / `state_tracker_type`: select `llm` or `rule`
- `TARGET_FPS` / `target_fps`: target analysis FPS
- `LLM_STATE_TRACKER_MODEL` / `llm_state_tracker_model`: model used for state inference
- `LLM_FEEDBACK_MODEL` / `llm_feedback_model`: model used for feedback generation

The main settings live in [settings.py](/mnt/c/Personal/VSCode/GenaiGenesis2026/server/config/settings.py). Current defaults include:

- `state_tracker_type = "llm"`
- `target_fps = 15`
- `llm_state_tracker_model = "gpt-5-mini"`
- `llm_feedback_model = "gpt-4.1-nano"`

## Fastest Way To Run Everything

### Option 1: Docker for backend, local desktop app

Start the backend services from the repo root:

```bash
docker compose up --build
```

This starts:

- `analysis` on TCP `9000`
- `control` on HTTP `8080`

Open:

```text
http://127.0.0.1:8080/
```

Then start the desktop app:

```bash
cd frontend
npm ci
npm run dev
```

If `npm run dev` fails with missing commands such as `concurrently`, run `npm ci` first in `frontend/`.

### Option 2: Run backend directly with `uv`

Install Python dependencies:

```bash
uv sync --all-groups
```

Start the unified Python server:

```bash
uv run python main.py server
```

Useful equivalents:

```bash
uv run cognitivesense server
uv run cognitivesense mirror
```

Notes:

- `server` and `mirror` both start the same unified remote media receiver
- `frontend` is still accepted as a deprecated alias
- the default desktop mode is `uv run python main.py`

Start the control website directly if needed:

```bash
uv run python -m uvicorn server.control.api:app --host 0.0.0.0 --port 8080
```

## Desktop Client

The Electron app is a thin client. It captures camera and microphone data locally and forwards them to the backend.

Development:

```bash
cd frontend
npm ci
npm run dev
```

Build desktop bundles:

```bash
cd frontend
npm ci
npm run dist
```

Artifacts are written to `frontend/release/`.

Currently configured packaging targets:

- Windows: `nsis`
- macOS: `dmg`
- Linux: `AppImage` and `deb`

The desktop app stores runtime connection info in a per-user `client-config.json` inside the Electron user-data directory.

Stored fields:

- `mediaHost`
- `mediaPort`
- `apiBaseUrl`
- `deviceId`
- `deviceName`

Defaults:

- media host: `127.0.0.1`
- media port: `9000`
- API base URL: `http://127.0.0.1:8080`

## Mirror Streaming

Build the Raspberry Pi binaries from the repo root:

```bash
cmake -S mirror -B mirror/build
cmake --build mirror/build
```

Run the capture smoke test on the Pi:

```bash
./mirror/build/mirror_capture_smoke
./mirror/build/mirror_capture_smoke 1280 720 8000
./mirror/build/mirror_capture_smoke 1280 720 8000 /home/pi/frame.bmp
```

Arguments are `width height timeout_ms [output_path]`.

To continuously stream frames to the backend:

```bash
./mirror/build/mirror_frame_streamer 192.168.1.10
./mirror/build/mirror_frame_streamer 192.168.1.10 9000 640 480 15
```

Arguments are `server_ip [port] [width] [height] [fps]`.

The streamer uses `rpicam-vid` when available and sends MJPEG frames over TCP.

To make a mirror device claimable with a stable identity:

```bash
COGNITIVESENSE_DEVICE_ID=bathroom-mirror ./mirror/build/mirror_frame_streamer 192.168.1.10
```

If the backend runs inside WSL2, Raspberry Pi devices on your LAN usually need Windows to forward the port into WSL first. Run this once from an elevated Windows PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_wsl_mirror_proxy.ps1 -Port 9000
```

Then use the printed Windows LAN IP as the Raspberry Pi target address.

## Control Website

Open the control site at:

```text
http://127.0.0.1:8080/
```

The website currently supports:

- registration and login
- device discovery for newly connected but unclaimed devices
- claiming desktop and mirror devices
- renaming claimed devices
- recent feedback history
- overview metrics and timeline views
- device snapshot previews
- live feed updates for claimed devices

## Auth And Device Association

The current model is local and self-hosted:

- each desktop or mirror client sends a stable `deviceId`
- the backend records devices automatically when they connect
- unclaimed devices appear in the control website
- a signed-in user can claim a device from the website
- user-level summaries aggregate across claimed devices

Persisted data currently includes:

- users
- auth tokens
- devices
- monitoring sessions
- state samples
- feedback events
- snapshot images

Raw audio and raw frame archives are not stored by default.

## Development Checks

Python:

```bash
uv run pyright .
uv run ruff check .
```

Frontend:

```bash
cd frontend
npm ci
npm run build
```

## Deployment And Distribution

For more detailed docs:

- [DEPLOYMENT.md](/mnt/c/Personal/VSCode/GenaiGenesis2026/DEPLOYMENT.md): remote backend deployment
- [DISTRIBUTION.md](/mnt/c/Personal/VSCode/GenaiGenesis2026/DISTRIBUTION.md): desktop packaging
- [deploy.env.example](/mnt/c/Personal/VSCode/GenaiGenesis2026/deploy.env.example): example deployment environment file
