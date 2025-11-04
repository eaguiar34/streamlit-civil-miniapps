import streamlit as st
from security import secure_mode_banner, require_environment_acknowledgement, require_passcode, purge_button


st.set_page_config(page_title="Civil Mini‑Apps", page_icon="🛠️", layout="wide")
secure_mode_banner()
require_environment_acknowledgement()
require_passcode()
purge_button()


st.title("Civil Engineering Mini‑Apps")
st.caption("Submittal Checker • Schedule What‑Ifs")


st.markdown(
"""
These are lightweight tools for day‑to‑day checks. Use the sidebar to switch pages.
No documents are written to disk; uploaded files are read in‑memory and discarded when you purge the session.
"""
)


import streamlit as st

st.set_page_config(page_title="Civil Mini‑Apps", page_icon="🛠️", layout="wide")

st.title("Civil Engineering Mini‑Apps")
st.caption("Submittal Checker • Schedule What‑Ifs")

# Keep markdown blocks simple and make sure every triple quote closes.
st.markdown(
    """
These are lightweight tools for day-to-day checks. Use the sidebar to switch pages:

- **Submittal Checker**: Paste a spec and a submittal narrative; get a coverage score and missing items.
- **Schedule What‑Ifs**: Enter tasks with durations, dependencies, and cost data; see critical path, then explore crash/fast‑track options.
    """
)

st.divider()

st.markdown(
    """
**Local run**

```
pip install -r requirements.txt
streamlit run app.py
```

**Deploy free (Streamlit Community Cloud)**

1. Push this folder to a public GitHub repo.
2. On share.streamlit.io, create a new app and choose `app.py`.
3. (Optional) Set Python to 3.11+. No secrets needed.
    """
)
