Param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# Prefer the project's venv Python, else fall back to system python
$venvPython = Join-Path $PSScriptRoot 'server\\.venv\\Scripts\\python.exe'
if (Test-Path $venvPython) { $python = $venvPython } else {
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if ($cmd) { $python = $cmd.Source } else { Write-Error "python not found. Activate the project's venv or install Python."; exit 1 }
}

# Ensure uvicorn and dotenv are available in the venv; install if missing
Write-Host "Using Python: $python"
& $python -m pip --version 2>$null | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "pip missing - bootstrapping..."
    & $python -m ensurepip --upgrade
}
& $python -c "import uvicorn, dotenv" 2>$null | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Installing missing packages (uvicorn, python-dotenv) into venv..."
    & $python -m pip install uvicorn python-dotenv
}

$serverDir = Join-Path $PSScriptRoot 'server'

# Run analysis (main.py server) and control (uvicorn) using the chosen Python
$analysisArgs = @('main.py','server')
$controlArgs = @('-m','uvicorn','control.api:app','--host','0.0.0.0','--port','8080')

$analysisProc = Start-Process -FilePath $python -ArgumentList $analysisArgs -WorkingDirectory $PSScriptRoot -PassThru -NoNewWindow
$controlProc = Start-Process -FilePath $python -ArgumentList $controlArgs -WorkingDirectory $serverDir -PassThru -NoNewWindow

Write-Host "Started analysis (PID=$($analysisProc.Id)) and control (PID=$($controlProc.Id)). Press Ctrl+C to stop."

try {
    Wait-Process -Id $analysisProc.Id,$controlProc.Id
} finally {
    Write-Host "Stopping services..."
    try { Stop-Process -Id $analysisProc.Id -ErrorAction SilentlyContinue } catch {}
    try { Stop-Process -Id $controlProc.Id -ErrorAction SilentlyContinue } catch {}
}
