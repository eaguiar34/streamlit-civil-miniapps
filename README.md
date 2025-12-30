# FieldFlow (Streamlit)

A lightweight, **single-file Streamlit app** that bundles a few practical construction / civil workflows:

- **Submittal Checker** — hybrid scoring of submittals vs. specs (lexical + semantic + keyword coverage), plus a run **Memory Bank**.
- **RFI Manager** — create / log RFIs, track status, and export your log.
- **Aging Dashboard** — quick “what’s late?” view for **Submittals + RFIs** with simple aging buckets.
- **Schedule What‑Ifs** — CPM with FS / SS / FF + lags, optional fast‑track overlap, floats breakdown, calendar mapping, and a simple crash-to-target loop.

Storage is **pluggable**:

- Local **SQLite** (default)
- **Google Sheets** (Service Account) *(optional)*
- **Google Sheets** (OAuth — per-user sign-in) *(optional)*
- **Microsoft 365 Excel** (OAuth — per-user sign-in) *(optional)*

> This repo includes examples for both **Streamlit Cloud** (use `st.secrets`) and **local runs** (use `.streamlit/secrets.toml` or environment variables).

---

## Project layout

- `app.py` — the Streamlit app
- `requirements.txt` — python deps
- `secrets.example.toml` — example Streamlit secrets (copy to `.streamlit/secrets.toml` for local)
- `.env.example` — optional environment-variable based secrets (local only)

If you received `app_updated.py` / `requirements_updated.txt`, you can:

```bash
mv app_updated.py app.py
mv requirements_updated.txt requirements.txt
```

---

## Run locally

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

pip install -r requirements.txt

# Create local secrets file
mkdir -p .streamlit
cp secrets.example.toml .streamlit/secrets.toml

streamlit run app.py
```

### Local secrets

For local runs, Streamlit reads:

- `.streamlit/secrets.toml` (recommended)

You *can* also supply environment variables. The app’s `get_secret("a.b.c")` helper looks for:

- `A__B__C` (dots become `__`, uppercase)

Example:

```bash
export GOOGLE_OAUTH__CLIENT_ID="..."
export GOOGLE_OAUTH__CLIENT_SECRET="..."
export GOOGLE_OAUTH__REDIRECT_URI="http://localhost:8501"
```

---

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub.
2. Create your app at https://share.streamlit.io/.
3. In the Streamlit app settings → **Secrets**, paste your TOML (same shape as `secrets.example.toml`).

**Important:** OAuth redirect URIs must match your deployed URL.

- Local dev redirect: `http://localhost:8501`
- Streamlit Cloud redirect: `https://YOUR-APP-NAME.streamlit.app`

(Use the exact URL shown by Streamlit for your deployed app.)

---

## OAuth: per-user Google + Microsoft sign-in (recommended)

If your goal is: **each user signs into their own Google/Microsoft and saves files into their own account**, you want **OAuth**.

You **do not** need to run your own server/service for this while staying on Streamlit.

What you *do* need as the app creator:

- A **Google Cloud OAuth client** (Web application)
- A **Microsoft Entra ID / Azure App registration**

These are just “app credentials” that identify *your* app to Google/Microsoft. Each user then authenticates *themselves*.

### Google OAuth setup (high level)

1. Google Cloud Console → APIs & Services
2. Enable:
   - Google Sheets API
   - Google Drive API
3. Create OAuth client:
   - Application type: **Web application**
   - Authorized redirect URIs:
     - `http://localhost:8501`
     - `https://YOUR-APP-NAME.streamlit.app`
4. Put the resulting `client_id`, `client_secret`, and your chosen `redirect_uri` into secrets under `[google_oauth]`.

### Microsoft OAuth setup (high level)

1. Microsoft Entra admin center / Azure portal → App registrations
2. Register a new app
3. Add a **Web** platform and set redirect URIs:
   - `http://localhost:8501`
   - `https://YOUR-APP-NAME.streamlit.app`
4. Create a **Client secret**
5. Put `client_id`, `client_secret`, `tenant_id` (or `common`) and `redirect_uri` into secrets under `[microsoft_oauth]`.

---

## Optional: Google Sheets Service Account backend

Service Accounts are **not per-user**. They are best when:

- you (the app owner) want a *single shared storage* spreadsheet
- you don’t need files saved into each user’s personal Drive

If you want per-user storage, stick to **Google OAuth**.

To use Service Account mode:

1. Create a service account in Google Cloud
2. Download its JSON key
3. Paste it under `[gcp_service_account]` in Streamlit secrets
4. (If needed) share the destination spreadsheet with the service account email

---

## Email notifications / monthly digest

The app can send a “feedback digest” email if you configure either:

- SendGrid: `[sendgrid] api_key=...`
- SMTP: `[smtp] host/user/password/from...`

If neither is configured, the digest still works for download/export, it just won’t email.

---

## Notes / troubleshooting

- **OAuth redirect mismatch** is the most common issue. The redirect URI in your provider console must exactly match the URL you are using.
- On Streamlit Cloud, the app restarts after secrets changes; refresh the page after editing secrets.
- For PDFs that are scans, enable the **OCR** checkbox (requires `pytesseract`, `pdf2image`, and system dependencies).

---

## Security basics

- Never commit real secrets. Use Streamlit Secrets or local `.streamlit/secrets.toml`.
- Prefer OAuth for user-owned data.
- Keep scopes minimal (Sheets/Drive file scope and basic profile/email).

