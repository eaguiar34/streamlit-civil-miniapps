import os, sys
try:
    import streamlit as st
except Exception:
    print("Install streamlit first: pip install streamlit")
    sys.exit(0)

missing = []

def has(path):
    try:
        node = st.secrets
        for k in path.split("."):
            if k in node: node = node[k]
            else: return False
        return True
    except Exception:
        return False

checks = [
    "sheets.title",
    # optional groups — at least one of these can be present
    # "google_oauth.client_id",
    # "gcp_service_account.client_email",
    # "microsoft_oauth.client_id",
]

for c in checks:
    if not has(c): missing.append(c)

if missing:
    print("Note: some optional secrets are missing (only needed if you use those backends):")
    for m in missing:
        print("  -", m)
else:
    print("Secrets look OK (or you’re using the default SQLite backend).")
