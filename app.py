import streamlit as st


st.set_page_config(page_title="Civil Mini‑Apps", page_icon="🛠️", layout="wide")


st.title("Civil Engineering Mini‑Apps")
st.caption("Submittal Checker • Schedule What‑Ifs")


st.markdown(
"""
These are lightweight tools for day‑to‑day checks. Use the sidebar to switch pages:


- **Submittal Checker**: Paste a spec and a submittal narrative; get a coverage score and missing items.
- **Schedule What‑Ifs**: Enter tasks with durations, dependencies, and cost data; see critical path, then explore crash/fast‑track options.


> Tip: File → Download as `.py` from this repo, or `git clone` and run locally.
"""
)


st.divider()


st.markdown(
"""
**Local run**
```bash
pip install -r requirements.txt
streamlit run app.py


