import streamlit as st
import fieldflow_core as core

st.set_page_config(page_title="RFI Manager", page_icon="❓", layout="wide")
core.render_sidebar("RFI Manager")
core.rfi_manager_page()
