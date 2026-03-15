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

### Mirror LCD Feedback

The Raspberry Pi mirror streamer can now mirror server state and LLM feedback onto a directly wired HD44780-style 16x2 LCD with no I2C backpack.

Build dependency on the Pi:

```bash
sudo apt install libgpiod-dev gpiod
cmake -S mirror -B mirror/build -DCOGNITIVESENSE_REQUIRE_LCD=ON
cmake --build mirror/build
```

When CMake is configured correctly, it should print a line like:

```text
mirror_frame_streamer: LCD support enabled with libgpiod
```

If that line does not appear, the LCD code is not compiled into the binary.

Enable the LCD by exporting the BCM GPIO pins you wired for `RS`, `E`, and `D4`-`D7`:

```bash
export COGNITIVESENSE_LCD_ENABLE=1
export COGNITIVESENSE_LCD_RS_PIN=26
export COGNITIVESENSE_LCD_E_PIN=19
export COGNITIVESENSE_LCD_D4_PIN=13
export COGNITIVESENSE_LCD_D5_PIN=6
export COGNITIVESENSE_LCD_D6_PIN=5
export COGNITIVESENSE_LCD_D7_PIN=11
```

Those are BCM GPIO numbers. For the Raspberry Pi header they map to:

- `GPIO26` -> physical pin `37`
- `GPIO19` -> physical pin `35`
- `GPIO13` -> physical pin `33`
- `GPIO6` -> physical pin `31`
- `GPIO5` -> physical pin `29`
- `GPIO11` -> physical pin `23`

If your Pi exposes the header lines through a different gpiochip, set it explicitly:

```bash
export COGNITIVESENSE_LCD_GPIO_CHIP=/dev/gpiochip4
```

Then start the mirror streamer with the LCD environment enabled:

```bash
COGNITIVESENSE_LCD_ENABLE=1 \
COGNITIVESENSE_LCD_RS_PIN=26 \
COGNITIVESENSE_LCD_E_PIN=19 \
COGNITIVESENSE_LCD_D4_PIN=13 \
COGNITIVESENSE_LCD_D5_PIN=6 \
COGNITIVESENSE_LCD_D6_PIN=5 \
COGNITIVESENSE_LCD_D7_PIN=11 \
./mirror/build/mirror_frame_streamer 192.168.1.10 9000 640 480 15 2>&1 | tee mirror_lcd.log
```

On startup, look for:

```text
LCD initialized on /dev/gpiochipX ...
```

If you instead see one of these messages, the LCD init path is not active yet:

- `LCD enabled, but libgpiod support was not compiled in`
- `LCD enabled, but no gpiochip could be opened`
- `LCD line request failed`

Once initialized, the first line shows the current analyzed state, and incoming LLM feedback is written onto the display.

## Control Website Live Feeds

Claimed devices now publish labeled snapshot previews into the control website. The selected device preview still updates live, and the main dashboard also shows a live feed grid for all claimed devices so mirror and desktop streams stay visible outside of debugging.
