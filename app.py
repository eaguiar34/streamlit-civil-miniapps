import streamlit as st
import fieldflow_core as core

st.set_page_config(page_title="FieldFlow", page_icon="🦺", layout="wide")
core.render_sidebar("Home")

st.title("FieldFlow")
st.markdown("Welcome. Use the left navigation to open pages (Submittal Checker, Schedule What-Ifs, RFI Manager, etc.).")

st.markdown("### Quick start")
st.markdown("- Pick a **Storage** backend in the sidebar (Local SQLite, Google Sheets OAuth, Microsoft 365 Excel OAuth).")
st.markdown("- If you choose Google/Microsoft, click **Sign in** in the sidebar.")
st.markdown("- For Microsoft: open **Settings & Examples** and click **Initialize OneDrive workbook** once (prevents Graph quota spam on reruns).")

st.markdown("### Sample files")
st.markdown("Open **Settings & Examples** to download sample spec/submittal/schedule files bundled in the repo under `sample_data/`.")
