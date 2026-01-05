#!/usr/bin/env bash
set -euo pipefail

# FieldFlow local runner (macOS/Linux)
# Usage:
#   bash scripts/run_local.sh
#
# Prereqs: Python 3.11+ recommended

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi

source .venv/bin/activate
python -m pip install --upgrade pip

if [ -f "requirements.txt" ]; then
  pip install -r requirements.txt
else
  echo "ERROR: requirements.txt not found in repo root."
  exit 1
fi

# Run
streamlit run app.py
