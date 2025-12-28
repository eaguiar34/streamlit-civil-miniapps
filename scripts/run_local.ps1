$ErrorActionPreference = "Stop"

if (!(Test-Path -Path ".\.venv")) {
  python -m venv .venv
}
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt

# sanity check
python scripts\test_secrets.py

streamlit run app.py
