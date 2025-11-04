# Streamlit Civil Engineering Mini‑Apps


Two tiny tools that run anywhere:
- **Submittal Checker** — Paste a spec section and a submittal. The app flags likely missing items with simple fuzzy matching.
- **Schedule What‑Ifs** — Enter tasks/dependencies; compute critical path; try fast‑track (overlap) and crash plans with rough cost impacts.


## Local Run
```bash
python -m pip install -r requirements.txt
streamlit run app.py


## Handling CUI / sensitive docs
This tool processes files **in memory** and avoids logging file contents, but compliance is a whole‑system property.


**Do not use public Streamlit Cloud for CUI.** Deploy on a controlled host that your organization has approved.


Minimum steps:
1. Set a passcode in `.streamlit/secrets.toml`:
[auth]
passcode = "generate-a-long-unique-passphrase"


2. Run locally or on an approved server (behind your VPN / reverse proxy). Prefer full‑disk encryption (BitLocker/FileVault/LUKS) and restricted access.
3. Add network and access controls at the platform level (SSO, CAC, firewall rules). If exposing externally, front Streamlit with a reverse proxy (Nginx) to enforce HTTPS, HSTS, and a Content Security Policy.
4. Pin dependencies and review changes (`pip-tools`, `requirements.txt` hashes) to reduce supply‑chain risk.
5. Purge sessions after use via the sidebar button.


References to consult with your security/compliance team:
- DoDI 5200.48 (CUI)
- NIST SP 800-171 / 800-53 controls
- DFARS 252.204-7012 (for contractors)
