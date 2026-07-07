# AI Text Analyzer - Windows Runner
# powershell -ExecutionPolicy Bypass -File .\run_windows.ps1

$ErrorActionPreference = "Stop"
$appDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $appDir

try { Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -TimeoutSec 3 | Out-Null }
catch {
    Write-Host "Ollama API not reachable. Starting ollama serve..."
    Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Minimized
    Start-Sleep -Seconds 8
}

& ".\.venv\Scripts\streamlit.exe" run RunwayML.py --server.address localhost --server.port 8501
