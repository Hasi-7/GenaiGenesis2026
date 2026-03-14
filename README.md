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
```

Arguments are `width height timeout_ms`. The smoke test prints one captured frame and exits with `0` on success.
