#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if pgrep -fl "web_gui.py" >/dev/null 2>&1; then
  pkill -f "web_gui.py" || true
fi

exec .venv/bin/python web_gui.py --port 8765
