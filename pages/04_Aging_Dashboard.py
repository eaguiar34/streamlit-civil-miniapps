import streamlit as st
import fieldflow_core as core

st.set_page_config(page_title="Aging Dashboard", page_icon="🦺", layout="wide")
core.render_sidebar("Aging Dashboard")

core.aging_dashboard_page()
