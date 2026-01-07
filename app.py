import streamlit as st
import fieldflow_core as core

st.set_page_config(page_title="FieldFlow", page_icon="🦺", layout="wide")

# Sidebar (storage + OAuth + feedback)
core.render_sidebar("Home")

st.title("FieldFlow")
st.caption("Welcome. Use the left navigation to open pages (Submittal Checker, Schedule What-Ifs, RFI Manager, etc.).")

st.markdown("""
### Getting started
- Pick a **Storage** backend in the sidebar (Local SQLite, Google Sheets, Microsoft 365 Excel).
- If you choose Google/Microsoft, click **Sign in** in the sidebar.
- For Microsoft: go to **Settings & Examples** and click **Initialize OneDrive workbook** once (this avoids Graph quota errors on reruns).

### Sample files
Open **Settings & Examples** to download sample spec/submittal/schedule files bundled in the repo (`sample_data/`).
""")
