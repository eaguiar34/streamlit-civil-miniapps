import streamlit as st
import fieldflow_core as core

st.set_page_config(page_title="Settings & Examples", page_icon="🧰", layout="wide")
core.render_sidebar("Settings & Examples")
core.settings_examples_page()
