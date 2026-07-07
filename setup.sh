#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-qwen2.5:1.5b}"
APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[1/6] Updating package index and installing dependencies..."
sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y curl python3 python3-venv python3-pip

echo "[2/6] Installing Ollama if missing..."
if ! command -v ollama >/dev/null 2>&1; then
  curl -fsSL https://ollama.com/install.sh | sh
else
  echo "Ollama already installed: $(ollama --version || true)"
fi

echo "[3/6] Starting Ollama service if possible..."
if command -v systemctl >/dev/null 2>&1; then
  sudo systemctl enable --now ollama || true
fi
# Fallback for environments without a running service.
if ! curl -fsS http://localhost:11434/api/tags >/dev/null 2>&1; then
  echo "Starting temporary ollama serve in background..."
  nohup ollama serve > "$APP_DIR/ollama.log" 2>&1 &
  sleep 5
fi

echo "[4/6] Pulling recommended model: $MODEL"
ollama pull "$MODEL"

echo "[5/6] Creating Python virtual environment..."
cd "$APP_DIR"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

echo "[6/6] Setup complete. Run the app with:"
echo "  cd $APP_DIR"
echo "  source .venv/bin/activate"
echo "  streamlit run app.py --server.address 0.0.0.0 --server.port 8501"
echo
echo "If your VPS firewall allows port 8501, open: http://YOUR_VPS_IP:8501"
