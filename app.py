import streamlit as st
import fieldflow_core as core

st.set_page_config(page_title="FieldFlow", page_icon="\U0001f6e0", layout="wide")
core.render_sidebar("Home")

st.title("FieldFlow")
st.write("Use the left navigation (pages) to open each tool.")

st.markdown("""
### What\'s here
- Submittal Checker
- Schedule What-Ifs
- RFI Manager
- Aging Dashboard
- Settings & Examples
- Saved Results

### Saving results
This build does **not** connect to Google Drive or OneDrive.

When you click **Save this result**, FieldFlow stores it in a small local SQLite database (inside Streamlit\'s app container).
You can browse and download later in **Saved Results**.
""")
