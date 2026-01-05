\
# FieldFlow local runner (Windows PowerShell)
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\run_local.ps1
#
# Prereqs: Python 3.11+ recommended

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

if (!(Test-Path ".venv")) {
    python -m venv .venv
}

.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

if (Test-Path "requirements.txt") {
    pip install -r requirements.txt
} else {
    Write-Host "ERROR: requirements.txt not found in repo root."
    exit 1
}

streamlit run app.py
