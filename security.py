from __future__ import annotations
import os
import io
import logging
import streamlit as st
import pandas as pd

# Keep Streamlit logs quieter
logging.getLogger("streamlit").setLevel(logging.WARNING)

# Read-only switches
SECURE_MODE: bool = str(os.getenv("SECURE_MODE", "true")).lower() == "true"
PASSCODE = (
    (st.secrets.get("auth", {}).get("passcode")) if hasattr(st, "secrets") else None
) or os.getenv("APP_PASSCODE")

def secure_mode_banner():
    if SECURE_MODE:
        st.warning(
            "Sensitive-data mode is ON. Files are processed in memory. Do not upload CUI to public services.",
            icon="⚠️",
        )

def require_environment_acknowledgement():
    if SECURE_MODE:
        ok = st.checkbox(
            "I am running this on a locally controlled or organization-approved environment (not public cloud).",
            value=False,
        )
        if not ok:
            st.stop()

def require_passcode():
    if not SECURE_MODE:
        return
    if not PASSCODE:
        st.info("Set a passcode in .streamlit/secrets.toml under [auth] passcode or env APP_PASSCODE.")
        return
    if st.session_state.get("_auth_ok"):
        return
    st.subheader("Access")
    pw = st.text_input("Enter passcode", type="password")
    if st.button("Unlock"):
        if pw == PASSCODE:
            st.session_state["_auth_ok"] = True
            st.experimental_rerun()
        else:
            st.error("Wrong passcode")
            st.stop()
    else:
        st.stop()

def purge_button():
    if st.sidebar.button("Purge session data", help="Clears all text, tables, and cached variables"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.experimental_rerun()

def read_text_any(uploaded_file) -> str:
    """Read PDF/DOCX/CSV/TXT into text from memory. No OCR for scanned PDFs."""
    if uploaded_file is None:
        return ""
    name = (uploaded_file.name or "").lower()
    mime = (uploaded_file.type or "").lower()
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
        st.error(f"Couldn't read {uploaded_file.name}: {e}")
        return ""
