from __future__ import annotations

import sys
from pathlib import Path

SECRETS_PATH = Path(".streamlit/secrets.toml")

# Only validate presence of keys (not whether they're non-empty).
# Keep OAuth optional — your app can run fine with SQLite only.
REQUIRED_BASE = {
    "sheets": ["title"],
}

OPTIONAL_GROUPS = {
    "google_oauth": ["client_id", "client_secret", "redirect_uri"],
    "microsoft_oauth": ["client_id", "client_secret", "redirect_uri", "tenant_id"],
    "gcp_service_account": ["type", "project_id", "private_key_id", "private_key", "client_email", "token_uri"],
    "sendgrid": ["api_key", "from_email", "from_name"],
    "smtp": ["host", "port", "user", "password", "from"],
}


def _load_from_streamlit():
    """Try to load secrets via Streamlit runtime (works on Streamlit Cloud or when run with streamlit)."""
    try:
        import streamlit as st  # noqa
        # st.secrets acts like a mapping
        return dict(st.secrets)
    except Exception:
        return None


def _load_from_toml_file():
    """Load secrets from local .streamlit/secrets.toml (works from CLI)."""
    if not SECRETS_PATH.exists():
        return None
    try:
        import tomllib
    except Exception:
        print("Python 3.11+ required for tomllib, or install tomli for older Python.", file=sys.stderr)
        return None

    return tomllib.loads(SECRETS_PATH.read_text(encoding="utf-8"))


def _check_section(data: dict, section: str, keys: list[str]) -> list[str]:
    missing = []
    if section not in data:
        return [f"Missing section [{section}]"]
    for k in keys:
        if k not in data[section]:
            missing.append(f"Missing key: [{section}].{k}")
    return missing


def main() -> int:
    data = _load_from_streamlit()
    source = "streamlit.secrets"

    if data is None:
        data = _load_from_toml_file()
        source = str(SECRETS_PATH)

    if data is None:
        print(f"❌ No secrets found. Provide Streamlit secrets or create {SECRETS_PATH}.")
        return 2

    print(f"Reading secrets from: {source}")

    errors: list[str] = []

    # Base required
    for section, keys in REQUIRED_BASE.items():
        errors.extend(_check_section(data, section, keys))

    if errors:
        print("❌ Required secrets missing:")
        for e in errors:
            print(" -", e)
        return 1

    # Optional groups: only warn if partially present
    warnings: list[str] = []
    for section, keys in OPTIONAL_GROUPS.items():
        if section in data:
            # If the section exists, ensure it has all keys
            warnings.extend(_check_section(data, section, keys))

    if warnings:
        print("⚠️ Optional sections incomplete (only matters if you use those backends/features):")
        for w in warnings:
            print(" -", w)

    print("✅ Secrets structure looks OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
