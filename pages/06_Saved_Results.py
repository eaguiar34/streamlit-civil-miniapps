import streamlit as st
import fieldflow_core as core

st.set_page_config(page_title="Saved Results", page_icon="🗂️", layout="wide")
core.render_sidebar("Saved Results")
core.saved_results_page()
