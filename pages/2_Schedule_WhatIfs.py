# 2_Schedule_WhatIfs.py
# Streamlit "Schedule What‑Ifs" — Floats + Calendar Mode
# - Flexible CSV ingest (aliases; optional Min_Duration & Overlap_OK)
# - CPM with Slack (Total Float), Free/Independent/Interfering Float
# - Optional clamping of floats ≥ 0 (toggle)
# - Fast‑track overlap
# - Calendar mode: map ES/EF to real dates via working‑day calendar (custom workweek + holidays)
# - Mini Gantt (numeric or date‑based) + CSV exports

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from datetime import date

import pandas as pd
import streamlit as st
import altair as alt
from pandas.tseries.offsets import CustomBusinessDay

# -------------------- Config -------------------- #
REQUIRED = [
    "Task",
    "Duration",
    "Predecessors",
    "Normal_Cost_per_day",
    "Crash_Cost_per_day",
]
# Optional: Min_Duration (defaults to Duration) and Overlap_OK (defaults False)

ALIASES: Dict[str, str] = {
    # Task
    "task": "Task", "activity": "Task", "name": "Task",
    # Duration
    "duration": "Duration", "duration_days": "Duration", "duration_(days)": "Duration", "dur": "Duration",
    # Predecessors
    "predecessors": "Predecessors", "pred": "Predecessors", "predecessor": "Predecessors",
    # Normal cost/day
    "normal_cost_per_day": "Normal_Cost_per_day", "normal/day": "Normal_Cost_per_day",
    "normal_cost/day": "Normal_Cost_per_day", "normal_cost": "Normal_Cost_per_day", "normal_cost_usd": "Normal_Cost_per_day",
    # Crash cost/day
    "crash_cost_per_day": "Crash_Cost_per_day", "crash/day": "Crash_Cost_per_day",
    "crash_cost/day": "Crash_Cost_per_day", "crash_cost": "Crash_Cost_per_day", "crash_cost_usd": "Crash_Cost_per_day",
    # Min duration (aka crash duration)
    "min_duration": "Min_Duration", "min_dur": "Min_Duration", "min": "Min_Duration",
    "crash_duration": "Min_Duration", "crash_duration_days": "Min_Duration", "crash_dur": "Min_Duration",
    # Overlap flag
    "overlap_ok": "Overlap_OK", "overlap?": "Overlap_OK", "allow_overlap": "Overlap_OK",
}

# -------------------- Utilities -------------------- #
def norm_col(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return re.sub(r"_+", "_", s).strip("_")


def canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    raw_to_norm = {c: norm_col(c) for c in df.columns}
    rename_map = {}
    for raw, norm in raw_to_norm.items():
        if norm in ALIASES:
            rename_map[raw] = ALIASES[norm]
    return df.rename(columns=rename_map)


def load_csv(uploaded) -> Tuple[pd.DataFrame, List[str]]:
    warnings: List[str] = []
    raw = uploaded.getvalue()
    df = pd.read_csv(io.BytesIO(raw))
    df = canonicalize_columns(df)

    if "Min_Duration" not in df.columns and "Duration" in df.columns:
        df["Min_Duration"] = df["Duration"]
        warnings.append("Min_Duration not provided; defaulting to Duration.")
    if "Overlap_OK" not in df.columns:
        df["Overlap_OK"] = False
        warnings.append("Overlap_OK not provided; defaulting to False for all tasks.")

    for col in ["Duration", "Min_Duration", "Normal_Cost_per_day", "Crash_Cost_per_day"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "Overlap_OK" in df.columns:
        df["Overlap_OK"] = (
            df["Overlap_OK"].astype(str).str.strip().str.lower().map(
                {"1": True, "0": False, "true": True, "false": False, "yes": True, "no": False, "y": True, "n": False}
            ).fillna(False)
        )

    if "Task" in df.columns:
        df["Task"] = df["Task"].astype(str)
    if "Predecessors" in df.columns:
        df["Predecessors"] = df["Predecessors"].fillna("").astype(str)

    if {"Duration", "Min_Duration"} <= set(df.columns):
        mask = df["Min_Duration"] > df["Duration"]
        if mask.any():
            warnings.append("Some rows have Min_Duration > Duration; clamping to Duration.")
            df.loc[mask, "Min_Duration"] = df.loc[mask, "Duration"]

    return df, warnings

# -------------------- CPM Core -------------------- #
@dataclass
class CPMNode:
    task: str
    dur: int
    preds: List[str]
    es: int = 0
    ef: int = 0
    ls: int = 0
    lf: int = 0
    tf: int = 0


def parse_predecessors(s: str) -> List[str]:
    if not s or not isinstance(s, str):
        return []
    return [p.strip() for p in re.split(r"[,;]", s) if p.strip()]


def topological_order(nodes: Dict[str, CPMNode]) -> List[str]:
    indeg = {k: 0 for k in nodes}
    for n in nodes.values():
        for p in n.preds:
            if p in indeg:
                indeg[n.task] += 1
    Q = [k for k, d in indeg.items() if d == 0]
    order = []
    while Q:
        v = Q.pop(0)
        order.append(v)
        for w in nodes:
            if v in nodes[w].preds:
                indeg[w] -= 1
                if indeg[w] == 0:
                    Q.append(w)
    if len(order) != len(nodes):
        raise ValueError("Dependency cycle detected. Check Predecessors.")
    return order


def cpm_schedule(df: pd.DataFrame, overlap_frac: float = 0.0, clamp_free_float: bool = True) -> Tuple[pd.DataFrame, int]:
    # Build nodes
    nodes: Dict[str, CPMNode] = {}
    for _, r in df.iterrows():
        nodes[r["Task"]] = CPMNode(
            task=r["Task"], dur=int(max(0, r["Duration"])), preds=parse_predecessors(r.get("Predecessors", ""))
        )

    order = topological_order(nodes)

    # Forward pass with optional fast‑track overlap
    for name in order:
        node = nodes[name]
        if not node.preds:
            node.es = 0
        else:
            starts = []
            for p in node.preds:
                if p not in nodes:
                    raise ValueError(f"Unknown predecessor '{p}' for task '{name}'.")
                pred = nodes[p]
                overlap_allow = bool(df.loc[df["Task"] == name, "Overlap_OK"].iloc[0]) if "Overlap_OK" in df.columns else False
                if overlap_allow and overlap_frac > 0:
                    starts.append(max(0, pred.ef - int(round(overlap_frac * pred.dur))))
                else:
                    starts.append(pred.ef)
            node.es = max(starts)
        node.ef = node.es + node.dur

    project_duration = max((n.ef for n in nodes.values()), default=0)

    # Successors map
    succs: Dict[str, List[str]] = {k: [] for k in nodes}
    for t, n in nodes.items():
        for p in n.preds:
            if p in succs:
                succs[p].append(t)

    # Backward pass
    for name in reversed(order):
        node = nodes[name]
        succ_ls = [nodes[s].ls for s in succs[name]]
        node.lf = min(succ_ls) if succ_ls else project_duration
        node.ls = node.lf - node.dur
        node.tf = node.ls - node.es  # total float (aka slack)

    # Floats
    rows = []
    for name in order:
        n = nodes[name]
        succ_es_min = min([nodes[s].es for s in succs[name]], default=project_duration)
        free_raw = succ_es_min - n.ef
        free_float = max(0, free_raw) if clamp_free_float else free_raw
        max_pred_ef = max([nodes[p].ef for p in n.preds], default=0)
        indep = max(0, succ_es_min - max_pred_ef - n.dur)  # conventional definition clamps at 0
        interfering = n.tf - free_float
        rows.append({
            "Task": n.task,
            "Duration": n.dur,
            "ES": n.es,
            "EF": n.ef,
            "LS": n.ls,
            "LF": n.lf,
            "Total_Float": n.tf,
            "Slack": n.tf,
            "Free_Float": free_float,
            "Free_Float_Raw": free_raw,
            "Independent_Float": indep,
            "Interfering_Float": interfering,
            "Critical": n.tf == 0,
        })

    out = pd.DataFrame(rows).sort_values("ES", kind="stable")
    return out, project_duration

# -------------------- Crashing -------------------- #

def crash_once(df: pd.DataFrame, schedule: pd.DataFrame) -> Optional[str]:
    crit = schedule[schedule["Critical"]]
    if crit.empty:
        return None
    merged = crit.merge(df[["Task", "Duration", "Min_Duration", "Normal_Cost_per_day", "Crash_Cost_per_day"]], on="Task")
    merged["slope"] = (merged["Crash_Cost_per_day"] - merged["Normal_Cost_per_day"]).astype(float)
    can = merged[merged["Duration"] > merged["Min_Duration"]]
    if can.empty:
        return None
    can = can.merge(schedule[["Task", "ES"]], on="Task").sort_values(["slope", "ES"], kind="stable")
    return can.iloc[0]["Task"]


def apply_crash(df: pd.DataFrame, task: str) -> pd.DataFrame:
    new = df.copy()
    new.loc[new["Task"] == task, "Duration"] = new.loc[new["Task"] == task, "Duration"] - 1
    return new


def total_cost(df: pd.DataFrame) -> float:
    baseline = (df["Normal_Cost_per_day"] * df["Duration"].round(0)).sum()
    crashed_days = (df.get("_baseline_duration", df["Duration"]) - df["Duration"]).clip(lower=0)
    slope = (df["Crash_Cost_per_day"] - df["Normal_Cost_per_day"]).clip(lower=0)
    return float(baseline + (crashed_days * slope).sum())


def crash_to_target(df: pd.DataFrame, target_days: int, overlap_frac: float, clamp_free_float: bool) -> Tuple[pd.DataFrame, List[str], int]:
    df = df.copy()
    df["_baseline_duration"] = df["Duration"]
    log: List[str] = []
    schedule, cur = cpm_schedule(df, overlap_frac, clamp_free_float)
    while cur > target_days:
        pick = crash_once(df, schedule)
        if pick is None:
            break
        df = apply_crash(df, pick)
        log.append(f"Shortened '{pick}' by 1 day → new durations applied.")
        schedule, cur = cpm_schedule(df, overlap_frac, clamp_free_float)
    return df.drop(columns=["_baseline_duration"]), log, cur

# -------------------- Calendar helpers -------------------- #

def make_cbd(workdays: List[str], holidays: List[str]) -> CustomBusinessDay:
    weekmask = " ".join(workdays)  # e.g., "Mon Tue Wed Thu Fri"
    hols = [pd.to_datetime(h).date() for h in holidays if h.strip()]
    return CustomBusinessDay(weekmask=weekmask, holidays=hols)


def calendarize(schedule_df: pd.DataFrame, project_start: date, cbd: CustomBusinessDay) -> pd.DataFrame:
    if schedule_df is None or schedule_df.empty:
        return schedule_df
    out = schedule_df.copy()
    start_ts = pd.to_datetime(project_start)
    out["ES_date"] = start_ts + out["ES"].astype(int) * cbd
    out["EF_date_excl"] = start_ts + out["EF"].astype(int) * cbd  # exclusive bound
    out["Start_Date"] = out["ES_date"]
    out["Finish_Date"] = out["EF_date_excl"] - 1 * cbd  # inclusive last working day
    return out

# -------------------- Gantt -------------------- #

def gantt_chart(schedule_df: pd.DataFrame) -> alt.Chart:
    if schedule_df is None or schedule_df.empty:
        return alt.Chart(pd.DataFrame({"ES": [0], "EF": [0], "Task": ["No tasks"]})).mark_bar()

    data = schedule_df.copy()
    data["Task"] = data["Task"].astype(str)
    height = max(140, min(32 * len(data), 640))

    if "ES_date" in data.columns and "EF_date_excl" in data.columns:
        x = alt.X("ES_date:T", title="Start")
        x2 = "EF_date_excl:T"
    else:
        x = alt.X("ES:Q", title="Day (project time)")
        x2 = "EF:Q"

    return (
        alt.Chart(data)
        .mark_bar()
        .encode(
            x=x,
            x2=x2,
            y=alt.Y("Task:N", sort=alt.SortField("ES", order="ascending")),
            color=alt.condition("datum.Critical", alt.value("#d62728"), alt.value("#1f77b4")),
            tooltip=[
                "Task", "Duration", "ES", "EF", "LS", "LF",
                "Total_Float", "Free_Float", "Independent_Float", "Interfering_Float",
                alt.Tooltip("Start_Date:T", title="Start Date", format="%Y-%m-%d"),
                alt.Tooltip("Finish_Date:T", title="Finish Date", format="%Y-%m-%d"),
            ],
        )
        .properties(height=height)
    )

# -------------------- UI -------------------- #
st.set_page_config(page_title="Schedule What‑Ifs", layout="wide")
st.title("Schedule What‑Ifs — Floats + Calendar")

st.write("Compute critical path, explore fast‑track/crash scenarios, see floats, and map to real dates.")

st.subheader("Task Table")
uploaded = st.file_uploader("Upload tasks CSV", type=["csv"], accept_multiple_files=False)

with st.expander("CSV column requirements & example", expanded=False):
    st.code(
        "Task,Duration,Predecessors,Normal_Cost_per_day,Crash_Cost_per_day,Min_Duration,Overlap_OK\n"
        "A - Site Prep,5,,1200,1800,3,TRUE\n"
        "B - Foundations,10,A - Site Prep,1600,2600,7,TRUE\n",
        language="csv",
    )
    st.caption("Required: Task, Duration, Predecessors, Normal_Cost_per_day, Crash_Cost_per_day · Optional: Min_Duration, Overlap_OK. Aliases like Duration_days, Crash_Cost_USD, Crash_Duration_days are accepted.")

base_df = pd.DataFrame({
    "Task": ["A - Site Prep", "B - Foundations", "C - Structure", "D - MEP Rough‑In", "E - Enclosure", "F - Finishes"],
    "Duration": [5, 10, 12, 8, 9, 10],
    "Predecessors": ["", "A - Site Prep", "B - Foundations", "C - Structure", "C - Structure", "D - MEP Rough‑In, E - Enclosure"],
    "Normal_Cost_per_day": [1200, 1600, 1900, 1500, 1400, 1550],
    "Crash_Cost_per_day": [1800, 2600, 3000, 2400, 2300, 2550],
    "Min_Duration": [3, 7, 9, 6, 7, 8],
    "Overlap_OK": [True, True, True, True, False, False],
})

if uploaded:
    df, warns = load_csv(uploaded)
else:
    df, warns = base_df.copy(), ["Using example table — upload your CSV to replace it."]

for w in warns:
    st.warning(w)

edited_df = st.data_editor(
    df,
    use_container_width=True,
    hide_index=True,
    num_rows="dynamic",
    column_config={
        "Task": st.column_config.TextColumn("Task", required=True),
        "Duration": st.column_config.NumberColumn("Duration", min_value=0, step=1),
        "Predecessors": st.column_config.TextColumn("Predecessors", help="Comma‑separated predecessor task names"),
        "Normal_Cost_per_day": st.column_config.NumberColumn("Normal Cost / day", min_value=0),
        "Crash_Cost_per_day": st.column_config.NumberColumn("Crash Cost / day", min_value=0),
        "Min_Duration": st.column_config.NumberColumn("Min Duration", min_value=0, step=1),
        "Overlap_OK": st.column_config.CheckboxColumn("Overlap OK"),
    },
    key="task_table",
)

st.markdown("---")
left, right = st.columns([1,1])
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
    holidays_text = st.text_area("Holidays (YYYY‑MM‑DD, one per line)", height=80, placeholder="2025-11-27\n2025-12-25")
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
        st.dataframe(base_schedule, use_container_width=True, hide_index=True)
        st.download_button("Download baseline schedule CSV", base_schedule.to_csv(index=False).encode(), "baseline_schedule.csv", "text/csv")
        st.subheader("Gantt (Baseline)")
        st.altair_chart(gantt_chart(base_schedule), use_container_width=True)

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
                st.dataframe(crashed_df[show_cols], use_container_width=True, hide_index=True)
                st.download_button("Download crashed durations CSV", crashed_df.to_csv(index=False).encode(), "crashed_durations.csv", "text/csv")

                st.subheader("Crashed Schedule")
                st.dataframe(crashed_schedule, use_container_width=True, hide_index=True)
                st.download_button("Download crashed schedule CSV", crashed_schedule.to_csv(index=False).encode(), "crashed_schedule.csv", "text/csv")
                st.subheader("Gantt (Crashed)")
                st.altair_chart(gantt_chart(crashed_schedule), use_container_width=True)

                st.subheader("Crash Log")
                if log:
                    for line in log:
                        st.write("• ", line)
                else:
                    st.write("No feasible crashes — target may be below theoretical minimum.")

    except Exception as e:
        st.exception(e)
