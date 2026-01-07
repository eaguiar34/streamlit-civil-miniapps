import streamlit as st
import fieldflow_core as core

st.set_page_config(page_title="Submittal Checker", page_icon="🧾", layout="wide")
core.render_sidebar("Submittal Checker")
core.submittal_checker_page()
