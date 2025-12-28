#!/usr/bin/env bash
set -euo pipefail

# create venv if missing
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt

# sanity checks
python scripts/test_secrets.py || true

# run
streamlit run app.py

chmod +x scripts/run_local.sh
