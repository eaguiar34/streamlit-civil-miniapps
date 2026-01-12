import streamlit as st
import pandas as pd
import fieldflow_core as core

st.set_page_config(page_title="Saved Results", page_icon="🗂️", layout="wide")
core.render_sidebar("Saved Results")

st.title("Saved Results")
backend = core.get_backend_choice()

st.markdown("### Saved Submittal Checks")
subs = pd.DataFrame()
try:
    subs = backend.list_submittals()
except Exception as e:
    st.exception(e)

if subs is None or subs.empty:
    st.info("No saved submittal checks in the currently selected backend.")
else:
    core.df_fullwidth(subs, hide_index=True, height=core.rows_to_height(len(subs)+6))
    ids = [int(x) for x in subs["id"].tolist()] if "id" in subs.columns else []
    pick = st.selectbox("Open submittal run", options=[None] + ids)
    if pick is not None:
        rec = backend.get_submittal(int(pick))
        st.markdown("**Metadata**")
        st.json(rec.get("meta", {}))
        csv_bytes = rec.get("result_csv_bytes") or b""
        if csv_bytes:
            st.download_button(
                "Download submittal results CSV",
                data=csv_bytes,
                file_name=f"fieldflow_submittal_{pick}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        hint = backend.open_url_hint(rec) if hasattr(backend,"open_url_hint") else None
        if hint:
            st.markdown(f"Open stored result: {hint}")

st.markdown("---")
st.markdown("### Saved Schedule What-If Runs")
runs = pd.DataFrame()
try:
    runs = backend.list_schedule_runs()
except Exception as e:
    st.exception(e)

if runs is None or runs.empty:
    st.info("No saved schedule runs in the currently selected backend.")
else:
    core.df_fullwidth(runs, hide_index=True, height=core.rows_to_height(len(runs)+6))
    run_ids = [str(x) for x in runs["run_id"].tolist()] if "run_id" in runs.columns else []
    pick2 = st.selectbox("Open schedule run", options=[""] + run_ids)
    if pick2:
        meta, df_run = backend.load_schedule_run(pick2)
        st.markdown("**Metadata**")
        st.json(meta)
        st.markdown("**Schedule**")
        core.df_fullwidth(df_run, hide_index=True, height=core.rows_to_height(len(df_run)+8))
        st.download_button(
            "Download schedule CSV",
            data=df_run.to_csv(index=False).encode("utf-8"),
            file_name=f"fieldflow_schedule_{pick2}.csv",
            mime="text/csv",
            use_container_width=True,
        )
