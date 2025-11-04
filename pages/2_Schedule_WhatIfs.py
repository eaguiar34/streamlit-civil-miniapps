# 2_Schedule_WhatIfs.py
# Streamlit page with tolerant CSV ingest, inline preview, and clear validation messages.
# It accepts common header variants and fixes the "CSV missing required columns" false-negative.

from __future__ import annotations

import io
import re
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st

REQUIRED = [
    "Task",
    "Duration",
    "Predecessors",
    "Normal_Cost_per_day",
    "Crash_Cost_per_day",
    "Min_Duration",
    "Overlap_OK",
]

# Map flexible header variants to canonical names
ALIASES: Dict[str, str] = {
    # task
    "task": "Task",
    "activity": "Task",
    "name": "Task",
    # duration
    "duration": "Duration",
    "duration_(days)": "Duration",
    "dur": "Duration",
    # predecessors
    "predecessors": "Predecessors",
    "pred": "Predecessors",
    "predecessor": "Predecessors",
    # cost per day (normal)
    "normal_cost_per_day": "Normal_Cost_per_day",
    "normal/day": "Normal_Cost_per_day",
    "normal_cost/day": "Normal_Cost_per_day",
    "normal_cost": "Normal_Cost_per_day",
    # cost per day (crash)
    "crash_cost_per_day": "Crash_Cost_per_day",
    "crash/day": "Crash_Cost_per_day",
    "crash_cost/day": "Crash_Cost_per_day",
    "crash_cost": "Crash_Cost_per_day",
    # min duration
    "min_duration": "Min_Duration",
    "min_dur": "Min_Duration",
    "min": "Min_Duration",
    # overlap
    "overlap_ok": "Overlap_OK",
    "overlap?": "Overlap_OK",
    "allow_overlap": "Overlap_OK",
}


def norm_col(s: str) -> str:
    # lowercase, remove non-alphanum, collapse underscores
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def canonicalize_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Rename columns to canonical names and return (df, missing_required)."""
    raw_to_norm = {c: norm_col(str(c)) for c in df.columns}
    rename_map = {}
    for raw, norm in raw_to_norm.items():
        if norm in ALIASES:
            rename_map[raw] = ALIASES[norm]
        elif norm.capitalize() in REQUIRED:
            # e.g., "Duration" when norm_col returned "Duration" ignoring case
            rename_map[raw] = norm.capitalize()
        elif norm == "task":
            rename_map[raw] = "Task"
        elif norm == "predecessors":
            rename_map[raw] = "Predecessors"
        elif norm == "overlap_ok":
            rename_map[raw] = "Overlap_OK"

    df = df.rename(columns=rename_map)

    missing = [c for c in REQUIRED if c not in df.columns]
    return df, missing


def load_csv(uploaded) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Read CSV, try to canonicalize headers, coerce dtypes, and report issues.
    Returns df, missing_required, warnings
    """
    warnings: List[str] = []
    try:
        raw = uploaded.getvalue()
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        raise

    df, missing = canonicalize_columns(df)

    # Coerce types if possible
    for col in ["Duration", "Normal_Cost_per_day", "Crash_Cost_per_day", "Min_Duration"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "Overlap_OK" in df.columns:
        # Accept 1/0, true/false, yes/no, y/n
        df["Overlap_OK"] = (
            df["Overlap_OK"].astype(str).str.strip().str.lower().map(
                {"1": True, "0": False, "true": True, "false": False, "yes": True, "no": False, "y": True, "n": False}
            ).fillna(False)
        )

    if "Predecessors" in df.columns:
        df["Predecessors"] = df["Predecessors"].fillna("").astype(str)

    # Sanity checks
    if set(["Duration", "Min_Duration"]).issubset(df.columns):
        bad = df[df["Min_Duration"] > df["Duration"]]
        if not bad.empty:
            warnings.append("Some rows have Min_Duration > Duration; clamping to Duration.")
            df.loc[df["Min_Duration"] > df["Duration"], "Min_Duration"] = df["Duration"]

    return df, missing, warnings


# --- UI --- #
st.set_page_config(page_title="Schedule What-Ifs", layout="wide")
st.title("Schedule What-Ifs")

st.write("Compute critical path, then explore crash/fast-track scenarios and cost impacts.")

st.subheader("Task Table")
uploaded = st.file_uploader("Upload tasks CSV", type=["csv"], accept_multiple_files=False)

example_help = "Required columns: " + ", ".join(REQUIRED)
with st.expander("CSV column requirements & example", expanded=False):
    st.code(
        "Task,Duration,Predecessors,Normal_Cost_per_day,Crash_Cost_per_day,Min_Duration,Overlap_OK\n"
        "A - Site Prep,5,,1200,1800,3,TRUE\n"
        "B - Foundations,10,A - Site Prep,1600,2600,7,TRUE\n",
        language="csv",
    )
    st.caption(example_help)

edited_df = None
issues: List[str] = []

if uploaded:
    df, missing, warns = load_csv(uploaded)

    if missing:
        st.error(f"CSV missing required columns: {missing}")
    if warns:
        for w in warns:
            st.warning(w)

    st.markdown("**Preview / Edit**")
    edited_df = st.data_editor(
        df,
        use_container_width=True,
        hide_index=True,
        num_rows="dynamic",
        column_config={
            "Task": st.column_config.TextColumn("Task", required=True),
            "Duration": st.column_config.NumberColumn("Duration", min_value=0, step=1),
            "Predecessors": st.column_config.TextColumn(
                "Predecessors", help="Comma-separated predecessor task names"
            ),
            "Normal_Cost_per_day": st.column_config.NumberColumn("Normal Cost / day", min_value=0),
            "Crash_Cost_per_day": st.column_config.NumberColumn("Crash Cost / day", min_value=0),
            "Min_Duration": st.column_config.NumberColumn("Min Duration", min_value=0, step=1),
            "Overlap_OK": st.column_config.CheckboxColumn("Overlap OK"),
        },
        key="task_table",
    )

# Controls
left, right = st.columns([1,1])
with left:
    if edited_df is not None:
        st.download_button("Download edited CSV", edited_df.to_csv(index=False).encode(), "tasks_edited.csv", "text/csv")

with right:
    target_days = st.number_input("Target project duration (days)", min_value=1, value=30, step=1)
    overlap_frac = st.number_input("Fast-track overlap fraction", min_value=0.0, max_value=0.9, value=0.0, step=0.05, help="Portion of successor that can overlap predecessors when Overlap OK")

# Compute button (placeholder for your CPM/crash logic)
compute = st.button("Compute CPM")

if compute:
    if edited_df is None:
        st.warning("Please upload a CSV first.")
    elif any(col not in edited_df.columns for col in REQUIRED):
        missing = [c for c in REQUIRED if c not in edited_df.columns]
        st.error(f"Cannot compute: still missing {missing}")
    else:
        st.success("CSV looks good. Plug this table into your CPM + crash logic.")
        st.dataframe(edited_df, use_container_width=True, hide_index=True)
        st.caption(f"Target = {target_days} days · Overlap fraction = {overlap_frac:.2f}")

# NOTE: Wire this edited_df into your existing CPM engine.  This page focuses on robust ingestion/preview.
