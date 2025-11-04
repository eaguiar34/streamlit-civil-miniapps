from __future__ import annotations
key="task_editor",
use_container_width=True,
num_rows="dynamic",
hide_index=True,
)


col1, col2, col3 = st.columns([1, 1, 1])
with col1:
run_cpm = st.button("Compute CPM")
with col2:
target_days = st.number_input("Target project duration (days)", min_value=1, value=30)
with col3:
overlap_fraction = st.slider("Fast-track overlap fraction", 0.0, 0.9, 0.0, 0.1,
help="Allow successors to start this fraction of predecessor duration early.")


if run_cpm:
try:
df_clean = df.copy()
df_clean["Duration"] = df_clean["Duration"].astype(int)
df_clean["Min_Duration"] = df_clean["Min_Duration"].astype(int)
for i, row in df_clean.iterrows():
if row["Min_Duration"] > row["Duration"]:
st.error(f"Row {i + 1} Min_Duration exceeds Duration.")
st.stop()


base = cpm(df_clean, overlap_fraction=0.0)
st.subheader("Base Schedule (no overlap)")
base_tbl = pd.DataFrame({
"Task": list(base.ES.keys()),
"ES": [base.ES[t] for t in base.ES],
"EF": [base.EF[t] for t in base.EF],
"LS": [base.LS[t] for t in base.LS],
"LF": [base.LF[t] for t in base.LF],
"On Critical Path": [t in base.critical_path for t in base.ES],
}).sort_values("ES")
st.dataframe(base_tbl, use_container_width=True, hide_index=True)
st.info(f"Base project duration: {base.total_duration} days. Critical path: {', '.join(base.critical_path)}")


if overlap_fraction > 0.0:
ft = cpm(df_clean, overlap_fraction=overlap_fraction)
ft_tbl = pd.DataFrame({
"Task": list(ft.ES.keys()),
"ES": [ft.ES[t] for t in ft.ES],
"EF": [ft.EF[t] for t in ft.EF],
"LS": [ft.LS[t] for t in ft.LS],
"LF": [ft.LF[t] for t in ft.LF],
"On Critical Path": [t in ft.critical_path for t in ft.ES],
}).sort_values("ES")
st.subheader("Fast-Tracked Schedule (overlap applied globally)")
st.dataframe(ft_tbl, use_container_width=True, hide_index=True)
st.success(f"Fast-tracked duration: {ft.total_duration} days (with {int(overlap_fraction * 100)}% overlap).")


st.subheader("Crash Plan (greedy, $/day slope)")
crashed_df, extra_cost = suggest_crash(df_clean, target_duration=target_days)
crashed_res = cpm(crashed_df, overlap_fraction=0.0)
st.dataframe(crashed_df, use_container_width=True, hide_index=True)
if crashed_res.total_duration <= target_days:
st.success(f"Target met: {crashed_res.total_duration} days. Est. extra cost ~ ${extra_cost:,.0f}.")
else:
st.warning(f"Target not fully met. Best via crashing: {crashed_res.total_duration} days. Est. extra cost ~ ${extra_cost:,.0f}.")


st.download_button(
label="Download current task table (CSV)",
data=df_clean.to_csv(index=False).encode("utf-8"),
file_name="schedule_tasks.csv",
mime="text/csv",
)
st.download_button(
label="Download crashed plan (CSV)",
data=crashed_df.to_csv(index=False).encode("utf-8"),
file_name="schedule_crashed.csv",
mime="text/csv",
)


st.caption("Heuristics only; not a substitute for a full risk/resource analysis.")


except Exception as e:
st.exception(e)
else:
st.markdown("Click Compute CPM to calculate schedule and explore scenarios.")
