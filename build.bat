@echo off
setlocal

:: Use uv if available, else fall back to venv python
where uv >nul 2>&1
if not errorlevel 1 (
    set RUNNER=uv run
    echo Using uv run
    goto start
)

echo uv not found, falling back to venv python
set PYTHON=%~dp0server\.venv\Scripts\python.exe
if not exist "%PYTHON%" set PYTHON=python
set RUNNER=%PYTHON%

:start
:: Start analysis server from project root
start "analysis" cmd /c "%RUNNER% main.py server"

:: Start control API server from server/ dir
pushd "%~dp0server"
start "control" cmd /c "%RUNNER% -m uvicorn control.api:app --host 0.0.0.0 --port 8080"
popd

echo Started. Press Ctrl+C to stop.
pause
