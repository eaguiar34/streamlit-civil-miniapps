from __future__ import annotations

import io
import logging
import os
from typing import Optional

import pandas as pd
import streamlit as st

# Keep Streamlit logs quieter
logging.getLogger("streamlit").setLevel(logging.WARNING)

# -----------------------------------------------------------------------------
# Security / privacy helpers
#
# Rationale:
# - Streamlit apps rerun frequently; keep all processing in-memory by default.
# - You can optionally enable "secure mode" + passcode to discourage uploading
#   sensitive info to public deployments.
#
# Defaults:
# - SECURE_MODE defaults to *false* (so public Streamlit Cloud deployments work
#   without a blocking checkbox).
# - Turn it on explicitly via:
#     - .streamlit/secrets.toml: [security] secure_mode = true
#   or
#     - env var: SECURE_MODE=true
# -----------------------------------------------------------------------------

def _truthy(v: object) -> bool:
    return str(v or "").strip().lower() in {"1", "true", "yes", "y", "on"}

# Secrets take precedence over env. Default: False.
SECURE_MODE: bool = _truthy(
    st.secrets.get("security", {}).get("secure_mode", os.getenv("SECURE_MODE", "false"))
)

PASSCODE: Optional[str] = (
    (st.secrets.get("auth", {}).get("passcode")) if hasattr(st, "secrets") else None
) or os.getenv("APP_PASSCODE")


def secure_mode_banner() -> None:
    """Show a visible banner when secure mode is enabled."""
    if SECURE_MODE:
        st.warning(
            "Sensitive-data mode is ON. Files are processed in memory only. "
            "Avoid uploading CUI/PII to public services.",
            icon="⚠️",
        )


def require_environment_acknowledgement() -> None:
    """Optional user acknowledgement gate (only in SECURE_MODE)."""
    if not SECURE_MODE:
        return

    ok = st.checkbox(
        "I understand this is a public web app and I will not upload sensitive data.",
        value=False,
    )
    if not ok:
        st.stop()


def require_passcode() -> None:
    """Optional passcode gate (only in SECURE_MODE)."""
    if not SECURE_MODE:
        return

    if not PASSCODE:
        st.info(
            "Optional: set a passcode in `.streamlit/secrets.toml` under `[auth] passcode` "
            "or via env var `APP_PASSCODE`."
        )
        return

    if st.session_state.get("_auth_ok"):
        return

    st.subheader("Access")
    pw = st.text_input("Enter passcode", type="password")
    if st.button("Unlock", use_container_width=True):
        if pw == PASSCODE:
            st.session_state["_auth_ok"] = True
            try:
                st.rerun()
            except Exception:
                st.experimental_rerun()
        else:
            st.error("Wrong passcode")
            st.stop()
    else:
        st.stop()


def purge_button() -> None:
    """Sidebar helper to clear session state."""
    if st.sidebar.button(
        "Purge session data",
        help="Clears all cached variables, tokens, and uploaded file contents from this session.",
        use_container_width=True,
    ):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        try:
            st.rerun()
        except Exception:
            st.experimental_rerun()


def read_text_any(uploaded_file) -> str:
    """Read PDF/DOCX/CSV/TXT into text from memory.

    Notes:
    - No OCR is performed for scanned PDFs.
    - This function reads the uploaded file bytes once; callers should not reuse
      the stream object afterward.
    """
    if uploaded_file is None:
        return ""

    name = (getattr(uploaded_file, "name", "") or "").lower()
    mime = (getattr(uploaded_file, "type", "") or "").lower()
    data = uploaded_file.read()

    try:
        if name.endswith(".pdf") or "pdf" in mime:
            from pypdf import PdfReader

            reader = PdfReader(io.BytesIO(data))
            pages = [(p.extract_text() or "") for p in reader.pages]
            return "\n".join(pages).strip()

        if name.endswith(".docx") or "wordprocessingml.document" in mime:
            from docx import Document

            doc = Document(io.BytesIO(data))
            return "\n".join(p.text for p in doc.paragraphs).strip()

        if name.endswith(".csv") or "csv" in mime:
            df = pd.read_csv(io.BytesIO(data))
            return df.to_csv(index=False)

        return data.decode("utf-8", errors="ignore")
    except Exception as e:
        st.error(f"Couldn't read {getattr(uploaded_file, 'name', 'file')}: {e}")
        return ""
