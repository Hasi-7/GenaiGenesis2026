#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if command -v uv >/dev/null 2>&1; then
  RUNNER="uv run"
else
  PY="$SCRIPT_DIR/server/.venv/Scripts/python.exe"
  [ -x "$PY" ] || PY="$SCRIPT_DIR/server/.venv/bin/python"
  [ -x "$PY" ] || PY="$(command -v python)"
  RUNNER="$PY"
fi

echo "Using: $RUNNER"

trap 'kill $PID_ANALYSIS $PID_CONTROL 2>/dev/null; wait 2>/dev/null' EXIT INT TERM

cd "$SCRIPT_DIR"
$RUNNER main.py server &
PID_ANALYSIS=$!

cd "$SCRIPT_DIR"
$RUNNER -m uvicorn server.control.api:app --host 0.0.0.0 --port 8080 &
PID_CONTROL=$!

echo "Analysis PID=$PID_ANALYSIS  Control PID=$PID_CONTROL"
wait $PID_ANALYSIS $PID_CONTROL
