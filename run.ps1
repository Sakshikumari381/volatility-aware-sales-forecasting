# Start API + Streamlit for volatility-aware forecasting (Windows PowerShell)
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

Write-Host "Project: $ProjectRoot"
Write-Host ""
$apiUp = $false
try {
    $r = Invoke-WebRequest -Uri "http://127.0.0.1:8000/health" -UseBasicParsing -TimeoutSec 2
    if ($r.StatusCode -eq 200) { $apiUp = $true }
} catch { }

if ($apiUp) {
    Write-Host "FastAPI already running on http://127.0.0.1:8000 (skipping new API window)."
} else {
    Write-Host "Starting FastAPI on http://127.0.0.1:8000 ..."
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$ProjectRoot'; python main.py api"
}

Start-Sleep -Seconds 4

Write-Host "Starting Streamlit on http://127.0.0.1:8501 ..."
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$ProjectRoot'; python main.py ui"

Write-Host ""
Write-Host "Done. Open:"
Write-Host "  Dashboard: http://127.0.0.1:8501"
Write-Host "  API docs:  http://127.0.0.1:8000/docs"
Write-Host ""
Write-Host "Keep BOTH terminal windows open. In the UI: Upload CSV -> Run Pipeline."
