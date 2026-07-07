# AI Text Analyzer - Student Package

This package runs the same Streamlit + Ollama demo on Linux VPS, Windows, and macOS.

## Recommended model

Default: `qwen2.5:1.5b`

If your laptop is slow, use:

```bash
MODEL=llama3.2:1b ./setup_macos.sh
```

On Windows PowerShell:

```powershell
$env:MODEL="llama3.2:1b"; .\setup_windows.ps1
```

## Linux VPS

```bash
chmod +x setup.sh run.sh
./setup.sh
./run.sh
```

Open:

```text
http://YOUR_VPS_IP:8501
```

## macOS

Prerequisite: Homebrew installed from https://brew.sh

```bash
chmod +x setup_macos.sh run_macos.sh
./setup_macos.sh
./run_macos.sh
```

Open:

```text
http://localhost:8501
```

## Windows

Prerequisites:

1. Install Python 3.11+ from https://www.python.org/downloads/
2. Tick **Add python.exe to PATH** during install.
3. Run PowerShell in this folder.

```powershell
powershell -ExecutionPolicy Bypass -File .\setup_windows.ps1
powershell -ExecutionPolicy Bypass -File .\run_windows.ps1
```

Open:

```text
http://localhost:8501
```

## What this app demonstrates

- Streamlit = web interface
- Ollama = local AI runtime
- LLM model = AI brain
- Prompt = instruction sent to AI
