from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from security import require_passcode

st.set_page_config(page_title="Schedule What-Ifs", page_icon="📅", layout="wide")
require_passcode()

st.title("Schedule What-Ifs")
st.caption("Compute critical path, then explore crash/fast-track scenarios and cost impacts.")

SAMPLE = pd.DataFrame([
    {"Task": "A - Site Prep", "Duration": 5, "Predecessors": "", "Normal_Cost_per_day": 1200.0, "Crash_Cost_per_day": 1800.0, "Min_Duration": 3, "Overlap_OK": True},
    {"Task": "B - Foundations", "Duration": 10, "Predecessors": "A - Site Prep", "Normal_Cost_per_day": 1600.0, "Crash_Cost_per_day": 2600.0, "Min_Duration": 7, "Overlap_OK": True},
    {"Task": "C - Structure", "Duration": 12, "Predecessors": "B - Foundations", "Normal_Cost_per_day": 1900.0, "Crash_Cost_per_day": 3000.0, "Min_Duration": 9, "Overlap_OK": True},
    {"Task": "D - MEP Rough-In", "Duration": 8, "Predecessors": "C - Structure", "Normal_Cost_per_day": 1500.0, "Crash_Cost_per_day": 2400.0, "Min_Duration": 6, "Overlap_OK": True},
    {"Task": "E - Enclosure", "Duration": 9, "Predecessors": "C - Structure", "Normal_Cost_per_day": 1400.0, "Crash_Cost_per_day": 2300.0, "Min_Duration": 7, "Overlap_OK": False},
    {"Task": "F - Finishes", "Duration": 10, "Predecessors": "D - MEP Rough-In, E - Enclosure", "Normal_Cost_per_day": 1550.0, "Crash_Cost_per_day": 2550.0, "Min_Duration": 8, "Overlap_OK": False},
])

def parse_preds(cell: str) -> List[str]:
    if not isinstance(cell, str) or not cell.strip():
        return []
    return [p.strip() for p in cell.split(",") if p.strip()]

@dataclass
class CPMResult:
    ES: Dict[str, int]
    EF: Dict[str, int]
    LS: Dict[str, int]
    LF: Dict[str, int]
    total_duration: int
    critical_path: List[str]

def topological_order(tasks: List[str], preds: Dict[str, List[str]]) -> List[str]:
    incoming = {t: set(preds.get(t, [])) for t in tasks}
    order: List[str] = []
    no_incoming = [t for t in tasks if not incoming[t]]
    while no_incoming:
        n = no_incoming.pop()
        order.append(n)
        for m in tasks:
            if n in incoming[m]:
                incoming[m].remove(n)
                if not incoming[m] and m not in order and m not in no_incoming:
                    no_incoming.append(m)
    if len(order) != len(tasks):
        raise ValueError("Cycle detected in dependencies. Check Predecessors column.")
    return order

def cpm(df: pd.DataFrame, overlap_fraction: float = 0.0) -> CPMResult:
    durations = {row.Task: int(row.Duration) for row in df.itertuples(index=False)}
    preds = {row.Task: parse_preds(row.Predecessors) for row in df.itertuples(index=False)}
    tasks = list(durations.keys())
    order = topological_order(tasks, preds)

    ES: Dict[str, int] = {}
    EF: Dict[str, int] = {}
    for t in order:
        if not preds[t]:
            ES[t] = 0
        else:
            candidate_starts = []
            for p in preds[t]:
                pdur = durations[p]
                allowed_overlap = int(round(overlap_fraction * pdur))
                candidate_starts.append(EF[p] - allowed_overlap)
            ES[t] = max(candidate_starts)
        EF[t] = ES[t] + durations[t]

    total = max(EF.values()) if EF else 0
    LS: Dict[str, int] = {}
    LF: Dict[str, int] = {}
    for t in reversed(order):
        succs = [s for s in tasks if t in preds[s]]
        if not succs:
            LF[t] = total
        else:
            candidate_finishes = []
            for s in succs:
                sdur = durations[t]
                allowed_overlap = int(round(overlap_fraction * sdur))
                candidate_finishes.append(LS[s] + allowed_overlap)
            LF[t] = min(candidate_finishes)
        LS[t] = LF[t] - durations[t]

    crit = [t for t in tasks if (LS[t] - ES[t]) == 0]
    return CPMResult(ES, EF, LS, LF, total_duration=total, critical_path=crit)

def suggest_crash(df: pd.DataFrame, target_duration: int):
    work = df.copy().reset_index(drop=True)
    work.index = work["Task"]

    def slope(row):
        return float(row["Crash_Cost_per_day"]) - float(row["Normal_Cost_per_day"])

    total_extra_cost = 0.0
    iteration = 0
    while True:
        res = cpm(work)
        if res.total_duration <= target_duration:
            break
        crit = [t for t in res.critical_path]
        candidates: List[str] = []
        for t in crit:
            dur = int(work.loc[t, "Duration"])
            min_d = int(work.loc[t, "Min_Duration"])
            if dur > min_d:
                candidates.append(t)
        if not candidates:
            break
        best_t = min(candidates, key=lambda t: slope(work.loc[t]))
        work.loc[best_t, "Duration"] -= 1
        total_extra_cost += slope(work.loc[best_t])
        iteration += 1
        if iteration > 5000:
            break
    return work.reset_index(drop=True), float(total_extra_cost)

st.subheader("Task Table")
st.markdown("Upload a CSV to load tasks, or edit the grid below. Use commas in Predecessors. Min_Duration <= Duration.")

uploaded = st.file_uploader("Upload tasks CSV", type=["csv"], key="tasks_csv")

if "task_df" not in st.session_state:
    st.session_state.task_df = SAMPLE.copy()

required_cols = ["Task","Duration","Predecessors","Normal_Cost_per_day","Crash_Cost_per_day","Min_Duration","Overlap_OK"]

if uploaded is not None:
    try:
        df_up = pd.read_csv(uploaded)
        missing = [c for c in required_cols if c not in df_up.columns]
        if missing:
            st.error(f"CSV missing required columns: {missing}")
        else:
            for col in ["Duration", "Min_Duration"]:
                df_up[col] = pd.to_numeric(df_up[col], errors="coerce").fillna(0).astype(int)
            for col in ["Normal_Cost_per_day", "Crash_Cost_per_day"]:
                df_up[col] = pd.to_numeric(df_up[col], errors="coerce").astype(float)
            if df_up["Overlap_OK"].dtype == object:
                df_up["Overlap_OK"] = (
                    df_up["Overlap_OK"].astype(str).str.strip().str.lower().map({
                        "true": True, "t": True, "yes": True, "y": True, "1": True,
                        "false": False, "f": False, "no": False, "n": False, "0": False,
                    }).fillna(False)
                )
            st.session_state.task_df = df_up
            st.success("Loaded tasks from CSV.")
    except Exception as e:
        st.error(f"Couldn't read CSV: {e}")

df = st.data_editor(
    st.session_state.task_df,
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
    overlap_fraction = st.slider("Fast-track overlap fraction", 0.0, 0.9, 0.0, 0.1, help="Allow successors to start this fraction of predecessor duration early.")

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
    except Exception as e:
        st.exception(e)
else:
    st.markdown("Click Compute CPM to calculate schedule and explore scenarios.")
