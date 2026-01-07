import streamlit as st
import fieldflow_core as core

st.set_page_config(page_title="FieldFlow", page_icon="🛠️", layout="wide")

core.render_sidebar("__Home__")

st.title("FieldFlow")
st.caption("Welcome. Use the left navigation to open pages (Submittal Checker, Schedule What-Ifs, RFI Manager, Aging Dashboard, Settings & Examples).")

st.markdown("### Getting started")
st.markdown(
        "- Pick a **Storage backend** in the sidebar (Local SQLite, Google Sheets, Microsoft 365 Excel).
"
        "- If you choose Google/Microsoft, click **Sign in** in the sidebar.
"
        "- If you don't see **Settings & Examples**, it usually means the `pages/05_Settings_and_Examples.py` file isn't in your repo *or* the app didn't deploy the latest commit."
    )

st.markdown("### Sample files")
st.markdown("Open **Settings & Examples** to download sample spec/submittal/schedule files bundled in the repo (`sample_data/`).")
