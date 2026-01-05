# FieldFlow — Local Run Files

Copy these folders into the ROOT of your FieldFlow repo (same folder that contains `app.py` and `requirements.txt`):

- `.streamlit/`
- `scripts/`

## 1) Add secrets
Create/modify: `.streamlit/secrets.toml`

- For local runs, set:
  - Google redirect_uri: `http://localhost:8501`
  - Microsoft redirect_uri: `http://localhost:8501`

Then paste your real client IDs/secrets.

## 2) Install + run
### macOS/Linux
```bash
bash scripts/run_local.sh
```

### Windows (PowerShell)
```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_local.ps1
```

## 3) Optional: quick sanity check
```bash
python scripts/test_secrets.py
```

## Common gotcha
In Google Cloud Console:
- **client_id** ends with `apps.googleusercontent.com`
- **client_secret** is the random secret string you generate/download
