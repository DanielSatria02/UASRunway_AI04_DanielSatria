#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-qwen2.5:1.5b}"
APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$APP_DIR"

echo "[1/6] Checking Homebrew..."
if ! command -v brew >/dev/null 2>&1; then
  echo "Homebrew not found. Install from https://brew.sh then rerun this script."
  exit 1
fi

echo "[2/6] Installing Python if needed..."
if ! command -v python3 >/dev/null 2>&1; then
  brew install python
fi

echo "[3/6] Installing Ollama if needed..."
if ! command -v ollama >/dev/null 2>&1; then
  brew install --cask ollama
  echo "Ollama installed. If macOS asks, open Ollama once from Applications, then rerun this script."
fi

echo "[4/6] Starting Ollama if needed..."
if ! curl -fsS http://localhost:11434/api/tags >/dev/null 2>&1; then
  open -a Ollama || true
  sleep 8
fi
if ! curl -fsS http://localhost:11434/api/tags >/dev/null 2>&1; then
  echo "Ollama API still not reachable. Open the Ollama app manually, then rerun."
  exit 1
fi

echo "[5/6] Pulling model: $MODEL"
ollama pull "$MODEL"

echo "[6/6] Creating Python virtual environment and installing Streamlit..."
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

echo "Setup complete. Run: ./run_macos.sh"
echo "Open: http://localhost:8501"
