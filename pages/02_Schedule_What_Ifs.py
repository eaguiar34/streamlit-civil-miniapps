import streamlit as st
import fieldflow_core as core

st.set_page_config(page_title="Schedule What-Ifs", page_icon="📅", layout="wide")
core.render_sidebar("Schedule What-Ifs")
core.schedule_whatifs_page()
