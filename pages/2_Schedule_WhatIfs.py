# 2_Schedule_WhatIfs.py
with left:
target_days = st.number_input("Target project duration (days)", min_value=1, value=30, step=1)
with right:
overlap_frac = st.number_input("Fast‑track overlap fraction", min_value=0.0, max_value=0.9, value=0.0, step=0.05, help="Portion of predecessor duration you can overlap when successor has Overlap OK.")


c1, c2 = st.columns([1,1])
clamp_ff = c1.checkbox("Clamp Free Float at ≥ 0", value=True, help="Classic CPM clamps Free Float to zero; uncheck to see negative Free Float under fast‑track overlaps.")
c1.caption("Tables show both **Free_Float** (affected by the toggle) and **Free_Float_Raw** (always raw) so you can see hidden overlaps.")
cal_mode = c2.checkbox("Calendar mode (map to dates)", value=True)


if cal_mode:
cw1, cw2 = st.columns([1,1])
with cw1:
proj_start = st.date_input("Project start date", value=date.today())
with cw2:
workdays = st.multiselect(
"Workdays",
options=["Mon","Tue","Wed","Thu","Fri","Sat","Sun"],
default=["Mon","Tue","Wed","Thu","Fri"],
help="Business days used to convert ES/EF into dates."
)
holidays_text = st.text_area("Holidays (YYYY‑MM‑DD, one per line)", height=80, placeholder="2025-11-27
2025-12-25")
holidays = [ln.strip() for ln in holidays_text.splitlines() if ln.strip()]
cbd = make_cbd(workdays, holidays)


cA, cB, cC, cD = st.columns([1,1,1,1])
compute = cA.button("Compute CPM", type="secondary")
run_crash = cB.button("Crash to Target", type="primary")
btn_dl_edited = cC.button("Download Edited CSV")


if btn_dl_edited:
st.download_button("Download edited CSV", edited_df.to_csv(index=False).encode(), "tasks_edited.csv", "text/csv")


if compute or run_crash:
try:
missing = [c for c in REQUIRED if c not in edited_df.columns]
if missing:
st.error(f"Missing required columns: {missing}")
st.stop()


base_schedule, base_days = cpm_schedule(edited_df, overlap_frac, clamp_ff)
if cal_mode:
base_schedule = calendarize(base_schedule, proj_start, cbd)


st.success(f"Baseline project duration: {base_days} days")
df_fullwidth(base_schedule, hide_index=True, height=rows_to_height(len(base_schedule)))
st.download_button("Download baseline schedule CSV", base_schedule.to_csv(index=False).encode(), "baseline_schedule.csv", "text/csv")
st.subheader("Gantt (Baseline)")
chart_fullwidth(gantt_chart(base_schedule))


if run_crash:
if target_days >= base_days:
st.info("Target ≥ baseline; nothing to crash.")
else:
crashed_df, log, final_days = crash_to_target(edited_df, target_days, overlap_frac, clamp_ff)
crashed_schedule, _ = cpm_schedule(crashed_df, overlap_frac, clamp_ff)
if cal_mode:
crashed_schedule = calendarize(crashed_schedule, proj_start, cbd)


edited_df["_baseline_duration"] = edited_df["Duration"]
base_cost = total_cost(edited_df)
crashed_df["_baseline_duration"] = edited_df["Duration"]
crash_cost = total_cost(crashed_df)


st.markdown("### Crashed Scenario")
st.success(f"New duration: {final_days} days (target: {target_days})")
st.metric("Added cost (approx)", f"${crash_cost - base_cost:,.0f}")


st.subheader("Revised Durations")
show_cols = ["Task","Duration","Min_Duration","Normal_Cost_per_day","Crash_Cost_per_day"]
df_fullwidth(crashed_df[show_cols], hide_index=True, height=rows_to_height(len(crashed_df)))
st.download_button("Download crashed durations CSV", crashed_df.to_csv(index=False).encode(), "crashed_durations.csv", "text/csv")


st.subheader("Crashed Schedule")
df_fullwidth(crashed_schedule, hide_index=True, height=rows_to_height(len(crashed_schedule)))
st.download_button("Download crashed schedule CSV", crashed_schedule.to_csv(index=False).encode(), "crashed_schedule.csv", "text/csv")
st.subheader("Gantt (Crashed)")
chart_fullwidth(gantt_chart(crashed_schedule))


st.subheader("Crash Log")
if log:
for line in log:
st.write("• ", line)
else:
st.write("No feasible crashes — target may be below theoretical minimum.")


except Exception as e:
st.exception(e)
