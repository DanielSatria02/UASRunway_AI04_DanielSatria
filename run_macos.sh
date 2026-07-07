#!/usr/bin/env bash
set -euo pipefail
APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$APP_DIR"
if ! curl -fsS http://localhost:11434/api/tags >/dev/null 2>&1; then
  open -a Ollama || true
  sleep 8
fi
source .venv/bin/activate
streamlit run RunwayML.py --server.address 0.0.0.0 --server.port 8501
