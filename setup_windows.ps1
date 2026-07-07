# AI Text Analyzer - Windows Setup
# Run in PowerShell: Right click -> Run with PowerShell, or:
# powershell -ExecutionPolicy Bypass -File .\setup_windows.ps1

$ErrorActionPreference = "Stop"
$model = if ($env:MODEL) { $env:MODEL } else { "qwen2.5:1.5b" }
$appDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $appDir

Write-Host "[1/6] Checking Python..."
try { py -3 --version } catch { throw "Python 3 not found. Install Python 3.11+ from https://www.python.org/downloads/ and tick 'Add python.exe to PATH'." }

Write-Host "[2/6] Checking Ollama..."
$ollama = Get-Command ollama -ErrorAction SilentlyContinue
if (-not $ollama) {
    Write-Host "Ollama not found. Downloading installer..."
    $installer = "$env:TEMP\OllamaSetup.exe"
    Invoke-WebRequest -Uri "https://ollama.com/download/OllamaSetup.exe" -OutFile $installer
    Write-Host "Running Ollama installer. Complete the installer window, then return here."
    Start-Process -FilePath $installer -Wait
} else {
    Write-Host "Ollama found."
}

Write-Host "[3/6] Starting Ollama if needed..."
try { Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -TimeoutSec 3 | Out-Null }
catch {
    Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Minimized
    Start-Sleep -Seconds 8
}

Write-Host "[4/6] Pulling model: $model"
ollama pull $model

Write-Host "[5/6] Creating Python virtual environment..."
py -3 -m venv .venv
& ".\.venv\Scripts\python.exe" -m pip install --upgrade pip
& ".\.venv\Scripts\pip.exe" install -r requirements.txt

Write-Host "[6/6] Setup complete. Run: .\run_windows.ps1"
Write-Host "Open: http://localhost:8501"
