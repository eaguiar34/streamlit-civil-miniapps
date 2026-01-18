import streamlit as st
import fieldflow_core as core

st.set_page_config(page_title="FieldFlow", page_icon="🦺", layout="wide")
core.render_sidebar("Home")

st.title("FieldFlow")
st.caption("Use the left navigation (pages) to open each tool.")

st.markdown("### What's here")
st.markdown("- Submittal Checker\n- Schedule What-Ifs\n- RFI Manager\n- Aging Dashboard\n- Settings & Examples\n- Saved Results")
