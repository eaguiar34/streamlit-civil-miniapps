import streamlit as st
from security import (
secure_mode_banner,
require_environment_acknowledgement,
require_passcode,
purge_button,
)


st.set_page_config(page_title="Civil Mini-Apps", page_icon="🛠️", layout="wide")


# --- Security guardrails (local/approved envs only) ---
secure_mode_banner()
require_environment_acknowledgement()
require_passcode()
purge_button()


st.title("Civil Engineering Mini-Apps")
st.caption("Submittal Checker • Schedule What-Ifs")


st.markdown(
"""
These are lightweight tools for day-to-day checks. Use the sidebar to switch pages.


- **Submittal Checker** — Paste text or upload PDF/DOCX/CSV; get a fuzzy-match coverage score and a Found/Weak/Missing table.
- **Schedule What-Ifs** — Edit/Upload tasks CSV; compute CPM; try fast-track overlap and a greedy crash plan with $/day slopes.


**Note:** Uploaded files are read in memory and not written to disk. Use on a locally controlled or organization‑approved host, not public cloud.
"""
)


st.divider()


st.markdown(
"""
**Local run**


```bash
python -m pip install -r requirements.txt
streamlit run app.py
