from __future__ import annotations

import io
import json
import os
import re
import sqlite3
import time
import uuid
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------
# UI helpers
# -----------------------------

APP_NAME = "FieldFlow"
DB_PATH = Path(".fieldflow") / "fieldflow.sqlite"
ASSETS_DIR = Path(__file__).parent / "assets"
LOGO_CANDIDATES = [
    ASSETS_DIR / "FieldFlow_logo.png",
    ASSETS_DIR / "FieldFlow_logo.jpg",
    ASSETS_DIR / "logo.png",
]


def _ensure_db_dir() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def _db() -> sqlite3.Connection:
    _ensure_db_dir()
    conn = sqlite3.connect(DB_PATH.as_posix(), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _init_db() -> None:
    conn = _db()
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS schedule_runs (
            id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            name TEXT NOT NULL,
            baseline_csv TEXT NOT NULL,
            crashed_csv TEXT,
            meta_json TEXT
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS submittal_checks (
            id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            name TEXT NOT NULL,
            result_json TEXT NOT NULL
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS rfis (
            id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            project TEXT,
            subject TEXT,
            discipline TEXT,
            priority TEXT,
            question TEXT,
            response TEXT,
            status TEXT,
            due_date TEXT
        )
        """
    )

    conn.commit()
    conn.close()


_init_db()


def render_sidebar(active_page: str) -> None:
    # Logo at top
    for p in LOGO_CANDIDATES:
        if p.exists():
            st.sidebar.image(str(p), use_container_width=True)
            break

    st.sidebar.title(APP_NAME)

    st.sidebar.caption("This build saves locally (no Google/Microsoft login).")

    if st.sidebar.button("Purge session data"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Pages**")
    st.sidebar.write("Use the left nav (Streamlit pages) to switch tools.")

    st.sidebar.markdown("---")
    st.sidebar.caption(f"Page: {active_page}")


# -----------------------------
# Storage API
# -----------------------------


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def save_schedule_run(name: str, baseline_df: pd.DataFrame, crashed_df: Optional[pd.DataFrame], meta: dict) -> str:
    run_id = str(uuid.uuid4())
    conn = _db()
    cur = conn.cursor()

    baseline_csv = baseline_df.to_csv(index=False)
    crashed_csv = crashed_df.to_csv(index=False) if crashed_df is not None else None

    cur.execute(
        """
        INSERT INTO schedule_runs (id, created_at, name, baseline_csv, crashed_csv, meta_json)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (run_id, _utc_now_iso(), name, baseline_csv, crashed_csv, json.dumps(meta)),
    )
    conn.commit()
    conn.close()
    return run_id


def list_schedule_runs() -> List[sqlite3.Row]:
    conn = _db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM schedule_runs ORDER BY created_at DESC")
    rows = cur.fetchall()
    conn.close()
    return rows


def delete_schedule_run(run_id: str) -> None:
    conn = _db()
    cur = conn.cursor()
    cur.execute("DELETE FROM schedule_runs WHERE id = ?", (run_id,))
    conn.commit()
    conn.close()


def save_submittal_check(name: str, payload: dict) -> str:
    check_id = str(uuid.uuid4())
    conn = _db()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO submittal_checks (id, created_at, name, result_json)
        VALUES (?, ?, ?, ?)
        """,
        (check_id, _utc_now_iso(), name, json.dumps(payload)),
    )
    conn.commit()
    conn.close()
    return check_id


def list_submittal_checks() -> List[sqlite3.Row]:
    conn = _db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM submittal_checks ORDER BY created_at DESC")
    rows = cur.fetchall()
    conn.close()
    return rows


def delete_submittal_check(check_id: str) -> None:
    conn = _db()
    cur = conn.cursor()
    cur.execute("DELETE FROM submittal_checks WHERE id = ?", (check_id,))
    conn.commit()
    conn.close()


def save_rfi(payload: dict) -> str:
    rfi_id = str(uuid.uuid4())
    conn = _db()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO rfis (id, created_at, project, subject, discipline, priority, question, response, status, due_date)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            rfi_id,
            _utc_now_iso(),
            payload.get("project"),
            payload.get("subject"),
            payload.get("discipline"),
            payload.get("priority"),
            payload.get("question"),
            payload.get("response"),
            payload.get("status"),
            payload.get("due_date"),
        ),
    )
    conn.commit()
    conn.close()
    return rfi_id


def list_rfis() -> List[sqlite3.Row]:
    conn = _db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM rfis ORDER BY created_at DESC")
    rows = cur.fetchall()
    conn.close()
    return rows


def update_rfi(rfi_id: str, updates: dict) -> None:
    if not updates:
        return
    cols = []
    vals = []
    for k, v in updates.items():
        cols.append(f"{k} = ?")
        vals.append(v)
    vals.append(rfi_id)
    conn = _db()
    cur = conn.cursor()
    cur.execute(f"UPDATE rfis SET {', '.join(cols)} WHERE id = ?", tuple(vals))
    conn.commit()
    conn.close()


def delete_rfi(rfi_id: str) -> None:
    conn = _db()
    cur = conn.cursor()
    cur.execute("DELETE FROM rfis WHERE id = ?", (rfi_id,))
    conn.commit()
    conn.close()


# -----------------------------
# Submittal Checker
# -----------------------------


def _read_text_upload(upload) -> str:
    if upload is None:
        return ""
    data = upload.read()
    try:
        return data.decode("utf-8", errors="ignore")
    except Exception:
        return ""


def _extract_bullets(text: str) -> List[str]:
    # Grab simple bullet-like lines
    lines = [ln.strip() for ln in text.splitlines()]
    out = []
    for ln in lines:
        if not ln:
            continue
        if ln.startswith(("-", "*")):
            out.append(ln.lstrip("-* ").strip())
        elif re.match(r"^[A-Z]\.|^\d+\.", ln):
            out.append(ln)
    return out


def _keyword_set(text: str) -> set:
    words = re.findall(r"[A-Za-z][A-Za-z\-]{2,}", text.lower())
    stop = {
        "the",
        "and",
        "for",
        "with",
        "per",
        "shall",
        "from",
        "that",
        "this",
        "into",
        "your",
        "are",
        "not",
        "use",
        "useful",
        "submittal",
        "section",
    }
    return {w for w in words if w not in stop}


def submittal_checker_page() -> None:
    st.title("Submittal Checker")
    st.caption("Lightweight checker (no OCR / no external deps). Upload text files or paste content.")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Spec Source")
        spec_up = st.file_uploader("Upload spec (TXT)", type=["txt"], key="spec_up")
        spec_txt = st.text_area("Or paste spec text", height=200, key="spec_txt")
    with c2:
        st.subheader("Submittal Source")
        sub_up = st.file_uploader("Upload submittal (TXT)", type=["txt"], key="sub_up")
        sub_txt = st.text_area("Or paste submittal text", height=200, key="sub_txt")

    spec = spec_txt.strip() or _read_text_upload(spec_up)
    subm = sub_txt.strip() or _read_text_upload(sub_up)

    analyze = st.button("Analyze")

    if analyze:
        if not spec or not subm:
            st.error("Provide both spec and submittal text.")
            return

        spec_bul = _extract_bullets(spec)
        sub_bul = _extract_bullets(subm)

        spec_kw = _keyword_set(spec)
        sub_kw = _keyword_set(subm)

        missing = sorted(list(spec_kw - sub_kw))
        extra = sorted(list(sub_kw - spec_kw))
        overlap = sorted(list(spec_kw & sub_kw))

        st.session_state["__submittal_last__"] = {
            "spec_len": len(spec),
            "submittal_len": len(subm),
            "spec_bullets": spec_bul[:200],
            "submittal_bullets": sub_bul[:200],
            "missing_keywords": missing[:200],
            "extra_keywords": extra[:200],
            "overlap_keywords": overlap[:200],
        }

    last = st.session_state.get("__submittal_last__")
    if last:
        st.markdown("---")
        st.subheader("Results")

        m1, m2, m3 = st.columns(3)
        m1.metric("Spec words (approx)", str(max(1, last["spec_len"] // 5)))
        m2.metric("Submittal words (approx)", str(max(1, last["submittal_len"] // 5)))
        m3.metric("Keyword overlap", str(len(last["overlap_keywords"])))

        with st.expander("Spec bullets (detected)"):
            st.write(last["spec_bullets"] or "(none detected)")

        with st.expander("Submittal bullets (detected)"):
            st.write(last["submittal_bullets"] or "(none detected)")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Possibly missing in submittal**")
            st.write(last["missing_keywords"][:50] or "(none)")
        with c2:
            st.markdown("**Extra terms in submittal**")
            st.write(last["extra_keywords"][:50] or "(none)")

        st.markdown("---")
        st.subheader("Save")
        name = st.text_input("Name this submittal check", value=f"Submittal check {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        if st.button("Save this result"):
            payload = {
                "created_at": _utc_now_iso(),
                "name": name,
                "result": last,
            }
            cid = save_submittal_check(name=name, payload=payload)
            st.success(f"Saved. ID: {cid}")


# -----------------------------
# Schedule What-Ifs (simple CPM + crash)
# -----------------------------


def _load_schedule_csv(upload) -> pd.DataFrame:
    if upload is None:
        # Load sample
        sample = Path(__file__).parent / "sample_data" / "schedule_sample.csv"
        return pd.read_csv(sample)
    return pd.read_csv(upload)


def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # allow some variations
    col_map = {}
    for c in out.columns:
        c2 = c.strip()
        col_map[c] = c2
    out = out.rename(columns=col_map)

    required = ["Task", "Duration"]
    for r in required:
        if r not in out.columns:
            raise ValueError(f"Missing required column: {r}")

    # defaults
    for c in ["Predecessors", "Normal Cost/day", "Crash Cost/day", "Min Duration"]:
        if c not in out.columns:
            out[c] = "" if c == "Predecessors" else np.nan

    out["Duration"] = pd.to_numeric(out["Duration"], errors="coerce").fillna(0).astype(int)
    out["Min Duration"] = pd.to_numeric(out["Min Duration"], errors="coerce").fillna(out["Duration"]).astype(int)
    out["Crash Cost/day"] = pd.to_numeric(out["Crash Cost/day"], errors="coerce").fillna(np.inf)
    out["Normal Cost/day"] = pd.to_numeric(out["Normal Cost/day"], errors="coerce").fillna(np.nan)

    # task IDs: use first token before ' - ' if present, else use row index
    def _task_id(t: str, i: int) -> str:
        if isinstance(t, str) and " - " in t:
            return t.split(" - ", 1)[0].strip()
        if isinstance(t, str) and t.strip():
            return t.strip().split()[0]
        return f"T{i+1}"

    out["TaskID"] = [_task_id(str(t), i) for i, t in enumerate(out["Task"].astype(str).tolist())]
    return out


def _parse_pred_token(tok: str) -> Tuple[str, str, int]:
    # Examples:
    # "A - Site Prep FS+0" -> ("A", "FS", 0)
    # "B SS+2" -> ("B", "SS", 2)
    tok = tok.strip()
    if not tok:
        raise ValueError("empty predecessor")

    # Find relationship at end
    m = re.search(r"\b(FS|SS|FF|SF)\s*([\+\-]\s*\d+)?\s*$", tok, flags=re.IGNORECASE)
    rel = "FS"
    lag = 0
    head = tok
    if m:
        rel = m.group(1).upper()
        if m.group(2):
            lag = int(m.group(2).replace(" ", ""))
        head = tok[: m.start()].strip()

    # Extract ID from head
    if " - " in head:
        pid = head.split(" - ", 1)[0].strip()
    else:
        pid = head.split()[0].strip()

    return pid, rel, lag


def _edges(df: pd.DataFrame) -> List[Tuple[str, str, str, int]]:
    edges = []
    for _, row in df.iterrows():
        tid = row["TaskID"]
        preds = str(row.get("Predecessors", "") or "").strip()
        if not preds:
            continue
        parts = [p.strip() for p in preds.split(",") if p.strip()]
        for p in parts:
            pid, rel, lag = _parse_pred_token(p)
            edges.append((pid, tid, rel, lag))
    return edges


def _toposort(nodes: List[str], edges: List[Tuple[str, str, str, int]]) -> List[str]:
    # Kahn
    indeg = {n: 0 for n in nodes}
    adj = {n: [] for n in nodes}
    for u, v, rel, lag in edges:
        if u not in indeg:
            indeg[u] = 0
            adj[u] = []
        if v not in indeg:
            indeg[v] = 0
            adj[v] = []
        adj[u].append((v, rel, lag))
        indeg[v] += 1

    q = [n for n in indeg if indeg[n] == 0]
    out = []
    while q:
        n = q.pop(0)
        out.append(n)
        for v, _, _ in adj.get(n, []):
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)

    # If cycle, append remaining in arbitrary order
    if len(out) != len(indeg):
        remaining = [n for n in indeg if n not in out]
        out.extend(remaining)

    return [n for n in out if n in nodes]


def _compute_schedule(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    nodes = df["TaskID"].tolist()
    dur = dict(zip(df["TaskID"], df["Duration"]))
    edges = _edges(df)
    order = _toposort(nodes, edges)

    es = {n: 0 for n in nodes}
    ef = {n: dur.get(n, 0) for n in nodes}

    # Forward pass
    for n in order:
        # compute constraints from predecessors
        for u, v, rel, lag in edges:
            if v != n:
                continue
            if u not in es:
                continue
            if rel == "SS":
                es[n] = max(es[n], es[u] + lag)
            elif rel == "FS":
                es[n] = max(es[n], ef[u] + lag)
            elif rel == "FF":
                # finish-to-finish: EF(v) >= EF(u)+lag => ES(v) >= EF(u)+lag - dur(v)
                es[n] = max(es[n], ef[u] + lag - dur.get(n, 0))
            elif rel == "SF":
                # rare: ES(v)+dur(v) >= ES(u)+lag => ES(v) >= ES(u)+lag - dur(v)
                es[n] = max(es[n], es[u] + lag - dur.get(n, 0))

        ef[n] = es[n] + dur.get(n, 0)

    proj = max(ef.values()) if ef else 0

    # Backward pass
    ls = {n: proj - dur.get(n, 0) for n in nodes}
    lf = {n: proj for n in nodes}

    rev = list(reversed(order))
    for n in rev:
        # constraints from successors
        for u, v, rel, lag in edges:
            if u != n:
                continue
            if v not in ls:
                continue
            if rel == "SS":
                # ES(v) >= ES(u)+lag => LS(u) <= LS(v)-lag
                ls[u] = min(ls[u], ls[v] - lag)
                lf[u] = ls[u] + dur.get(u, 0)
            elif rel == "FS":
                # ES(v) >= EF(u)+lag => LF(u) <= LS(v)-lag
                lf[u] = min(lf[u], ls[v] - lag)
                ls[u] = lf[u] - dur.get(u, 0)
            elif rel == "FF":
                # EF(v) >= EF(u)+lag => LF(u) <= LF(v)-lag
                lf[u] = min(lf[u], lf[v] - lag)
                ls[u] = lf[u] - dur.get(u, 0)
            elif rel == "SF":
                # EF(v) >= ES(u)+lag => LS(u) <= LF(v)-lag
                ls[u] = min(ls[u], lf[v] - lag)
                lf[u] = ls[u] + dur.get(u, 0)

    out = df.copy()
    out["ES"] = out["TaskID"].map(es)
    out["EF"] = out["TaskID"].map(ef)
    out["LS"] = out["TaskID"].map(ls)
    out["LF"] = out["TaskID"].map(lf)
    out["Float"] = out["LS"] - out["ES"]
    out["Critical"] = out["Float"].fillna(0).astype(float).abs() < 1e-9
    return out.sort_values(["ES", "TaskID"]).reset_index(drop=True)


def _crash_to_target(df: pd.DataFrame, target: int) -> Tuple[pd.DataFrame, Dict[str, int]]:
    df = df.copy()
    reductions = {tid: 0 for tid in df["TaskID"].tolist()}

    def project_duration(dfx: pd.DataFrame) -> int:
        sch = _compute_schedule(dfx)
        return int(sch["EF"].max() if len(sch) else 0)

    # Safety limit to avoid infinite loops
    max_steps = int(df['Duration'].sum()) + 100

    steps = 0
    while project_duration(df) > target and steps < max_steps:
        sch = _compute_schedule(df)
        proj = int(sch["EF"].max())
        if proj <= target:
            break

        crit = sch[sch["Critical"] == True].copy()
        if crit.empty:
            break

        # candidate tasks that can still be reduced
        cand = crit.merge(
            df[["TaskID", "Duration", "Min Duration", "Crash Cost/day"]],
            on="TaskID",
            how="left",
        )
        cand = cand[cand["Duration"] > cand["Min Duration"]]
        if cand.empty:
            break

        cand = cand.sort_values(["Crash Cost/day", "Duration"], ascending=[True, False])
        pick = cand.iloc[0]["TaskID"]

        # reduce by 1 day
        df.loc[df["TaskID"] == pick, "Duration"] = df.loc[df["TaskID"] == pick, "Duration"].iloc[0] - 1
        reductions[pick] += 1
        steps += 1

    return df, reductions


# NOTE: The line above uses a fancy quote if pasted from rich text.
# Fix it defensively:



def schedule_whatifs_page() -> None:
    st.title("Schedule What-Ifs")
    st.caption("Upload a schedule CSV, compute a basic CPM schedule, optionally crash to a target duration, then save / download.")

    upload = st.file_uploader("Upload tasks CSV", type=["csv"], key="sched_up")
    try:
        base_raw = _load_schedule_csv(upload)
        base = _normalize_cols(base_raw)
    except Exception as e:
        st.error(f"Could not load schedule: {e}")
        return

    with st.expander("CSV columns & example"):
        st.markdown(
            """
Required columns:
- **Task** (name)
- **Duration** (days)

Optional columns:
- **Predecessors** (comma-separated, e.g. `A FS+0`, `B SS+2`)
- **Normal Cost/day**, **Crash Cost/day**, **Min Duration**

A sample CSV is available in **Settings & Examples**.
"""
        )

    st.subheader("Task Table")
    edited = st.data_editor(
        base[["Task", "TaskID", "Duration", "Predecessors", "Normal Cost/day", "Crash Cost/day", "Min Duration"]],
        use_container_width=True,
        num_rows="dynamic",
        key="sched_editor",
    )

    try:
        edited = _normalize_cols(edited)
    except Exception as e:
        st.error(f"Invalid table: {e}")
        return

    # Compute baseline
    try:
        baseline = _compute_schedule(edited)
    except Exception as e:
        st.error(f"Could not compute schedule: {e}")
        return

    proj = int(baseline["EF"].max() if len(baseline) else 0)
    st.markdown("---")
    st.subheader("Baseline")
    st.success(f"Baseline project duration: {proj} days")
    st.dataframe(baseline, use_container_width=True)

    # Crash
    st.markdown("---")
    st.subheader("Crash to target")
    target = st.number_input("Target project duration (days)", min_value=0, value=max(0, proj), step=1)
    do_crash = st.button("Crash to target")

    crashed = None
    reductions = None
    if do_crash:
        crashed_df, reductions = _crash_to_target(edited, int(target))
        crashed = _compute_schedule(crashed_df)
        proj2 = int(crashed["EF"].max() if len(crashed) else 0)
        st.info(f"Crashed project duration: {proj2} days (target: {int(target)})")
        if reductions:
            red_df = pd.DataFrame(
                [{"TaskID": k, "Days crashed": v} for k, v in reductions.items() if v > 0]
            ).sort_values("Days crashed", ascending=False)
            st.write("Crash summary:")
            st.dataframe(red_df, use_container_width=True)
        st.dataframe(crashed, use_container_width=True)

    # Downloads
    st.markdown("---")
    st.subheader("Save, download, and browse schedule runs")

    base_csv = baseline.to_csv(index=False).encode("utf-8")
    st.download_button("Download baseline CSV", data=base_csv, file_name="schedule_baseline.csv", mime="text/csv", use_container_width=True)

    if crashed is not None:
        crash_csv = crashed.to_csv(index=False).encode("utf-8")
        st.download_button("Download crashed CSV", data=crash_csv, file_name="schedule_crashed.csv", mime="text/csv", use_container_width=True)

    with st.expander("Save this schedule run (to local SQLite)"):
        run_name = st.text_input("Run name", value=f"Schedule run {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        if st.button("Save this result"):
            meta = {
                "baseline_duration_days": proj,
                "target_duration_days": int(target),
                "crashed": crashed is not None,
            }
            rid = save_schedule_run(run_name, baseline, crashed, meta)
            st.success(f"Saved schedule run. ID: {rid}")

    with st.expander("Saved schedule runs"):
        runs = list_schedule_runs()
        if not runs:
            st.caption("No saved runs yet.")
        for r in runs:
            st.markdown(f"**{r['name']}**  ")
            st.caption(f"Saved: {r['created_at']}  |  ID: {r['id']}")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.download_button(
                    "Download baseline",
                    data=r["baseline_csv"].encode("utf-8"),
                    file_name=f"{r['name']}_baseline.csv".replace(" ", "_"),
                    mime="text/csv",
                    key=f"dlb_{r['id']}",
                    use_container_width=True,
                )
            with c2:
                if r["crashed_csv"]:
                    st.download_button(
                        "Download crashed",
                        data=r["crashed_csv"].encode("utf-8"),
                        file_name=f"{r['name']}_crashed.csv".replace(" ", "_"),
                        mime="text/csv",
                        key=f"dlc_{r['id']}",
                        use_container_width=True,
                    )
                else:
                    st.button("No crashed", disabled=True, key=f"noc_{r['id']}")
            with c3:
                if st.button("Delete", key=f"del_{r['id']}"):
                    delete_schedule_run(r["id"])
                    st.rerun()
            st.markdown("---")


# -----------------------------
# RFI Manager
# -----------------------------


def rfi_manager_page() -> None:
    st.title("RFI Manager")

    with st.expander("Create new RFI", expanded=True):
        c1, c2, c3 = st.columns(3)
        project = c1.text_input("Project")
        discipline = c2.selectbox("Discipline", ["General", "Civil", "Structural", "MEP", "Architectural", "Other"], index=0)
        priority = c3.selectbox("Priority", ["Normal", "High", "Critical"], index=0)

        subject = st.text_input("Subject")
        question = st.text_area("Question / Clarification needed")
        due_date = st.date_input("Due date (optional)")

        status = st.selectbox("Status", ["Open", "Answered", "Closed"], index=0)
        response = st.text_area("Response (optional)")

        if st.button("Save RFI"):
            payload = {
                "project": project,
                "discipline": discipline,
                "priority": priority,
                "subject": subject,
                "question": question,
                "response": response,
                "status": status,
                "due_date": due_date.isoformat() if due_date else None,
            }
            rid = save_rfi(payload)
            st.success(f"Saved RFI. ID: {rid}")

    st.markdown("---")
    st.subheader("RFIs")
    rows = list_rfis()
    if not rows:
        st.caption("No RFIs yet.")
        return

    df = pd.DataFrame([dict(r) for r in rows])
    st.dataframe(df, use_container_width=True)
    st.download_button("Download RFIs CSV", data=df.to_csv(index=False).encode("utf-8"), file_name="rfis.csv", mime="text/csv")


# -----------------------------
# Aging Dashboard
# -----------------------------


def aging_dashboard_page() -> None:
    st.title("Aging Dashboard")
    rows = list_rfis()
    if not rows:
        st.info("No RFIs to chart yet. Create some in RFI Manager.")
        return

    df = pd.DataFrame([dict(r) for r in rows])
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
    df["age_days"] = (pd.Timestamp.now(tz="UTC") - df["created_at"]).dt.total_seconds() / 86400.0

    st.subheader("RFI age (days)")
    st.bar_chart(df["age_days"])

    st.subheader("By status")
    st.bar_chart(df["status"].fillna("Unknown").value_counts())


# -----------------------------
# Settings & Examples
# -----------------------------


def settings_examples_page() -> None:
    st.title("Settings & Examples")

    st.markdown("### Sample files (download)")
    base = Path(__file__).parent / "sample_data"
    samples = [
        ("spec_sample.txt", "Spec sample (text)"),
        ("submittal_sample.txt", "Submittal sample (text)"),
        ("schedule_sample.csv", "Schedule sample (CSV)"),
    ]

    for fn, label in samples:
        fp = base / fn
        if fp.exists():
            st.download_button(
                label=f"Download: {label}",
                data=fp.read_bytes(),
                file_name=fn,
                mime="application/octet-stream",
                use_container_width=True,
            )
        else:
            st.warning(f"Missing file: {fp}")

    st.markdown("---")
    st.markdown(
        """
### About storage
- FieldFlow stores saved items in a local SQLite DB at `.fieldflow/fieldflow.sqlite`.
- On Streamlit Cloud, local disk is **not guaranteed** to be permanent across rebuilds.

If you need durable storage later, we can plug in a real database (Postgres/Supabase) without changing the UI much.
"""
    )


# -----------------------------
# Saved Results
# -----------------------------




# -----------------------------
# Export helpers
# -----------------------------

def _slug(s: str) -> str:
    s = (s or '').strip()
    s = re.sub(r'[^A-Za-z0-9._-]+', '_', s)
    s = re.sub(r'_+', '_', s).strip('_')
    return s or 'item'


def _build_all_saved_results_zip() -> bytes:
    """Build a ZIP containing all saved results from the local SQLite DB.

    This is computed only when the user clicks the button on the Saved Results page.
    """
    buf = io.BytesIO()
    created = _utc_now_iso()

    runs = list_schedule_runs()
    checks = list_submittal_checks()
    rfis = list_rfis()

    manifest = {
        'created_at': created,
        'counts': {
            'schedule_runs': len(runs),
            'submittal_checks': len(checks),
            'rfis': len(rfis),
        },
        'notes': 'Export generated by FieldFlow (local-only).',
    }

    with zipfile.ZipFile(buf, mode='w', compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr('manifest.json', json.dumps(manifest, indent=2))

        # Schedule runs
        for r in runs:
            base = f"schedule_runs/{r['created_at']}_{_slug(r['name'])}_{r['id']}"
            z.writestr(f"{base}/baseline.csv", r['baseline_csv'] or '')
            if r['crashed_csv']:
                z.writestr(f"{base}/crashed.csv", r['crashed_csv'])
            meta = {}
            try:
                meta = json.loads(r['meta_json'] or '{}')
            except Exception:
                meta = {}
            meta_out = {
                'id': r['id'],
                'created_at': r['created_at'],
                'name': r['name'],
                'meta': meta,
            }
            z.writestr(f"{base}/meta.json", json.dumps(meta_out, indent=2))

        # Submittal checks
        for c in checks:
            base = f"submittal_checks/{c['created_at']}_{_slug(c['name'])}_{c['id']}"
            z.writestr(f"{base}/result.json", c['result_json'] or '{}')

        # RFIs
        if rfis:
            df = pd.DataFrame([dict(r) for r in rfis])
            z.writestr('rfis/rfis.csv', df.to_csv(index=False))

    return buf.getvalue()

def saved_results_page() -> None:
    st.title("Saved Results")

    st.markdown("### Export")
    if st.button("Build ZIP of all saved results", use_container_width=True):
        try:
            st.session_state["__export_zip__"] = _build_all_saved_results_zip()
            st.session_state["__export_zip_built_at__"] = _utc_now_iso()
            st.toast("ZIP is ready — download below.")
        except Exception as e:
            st.error(f"Failed to build ZIP: {e}")

    if st.session_state.get("__export_zip__"):
        ts = st.session_state.get("__export_zip_built_at__", "")
        st.download_button(
            label=f"Download all saved results (ZIP){' — ' + ts if ts else ''}",
            data=st.session_state["__export_zip__"],
            file_name=f"fieldflow_saved_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
            mime="application/zip",
            use_container_width=True,
        )


    tab1, tab2, tab3 = st.tabs(["Schedule runs", "Submittal checks", "RFIs"])

    with tab1:
        runs = list_schedule_runs()
        if not runs:
            st.caption("No saved schedule runs yet.")
        for r in runs:
            st.markdown(f"**{r['name']}**")
            st.caption(f"Saved: {r['created_at']}  |  ID: {r['id']}")
            meta_str = r.get("meta_json") or "{}"
            try:
                meta_obj = json.loads(meta_str)
            except Exception:
                meta_obj = {}
            meta_out = {"id": r.get('id'), "created_at": r.get('created_at'), "name": r.get('name'), "meta": meta_obj}
            meta_bytes = json.dumps(meta_out, indent=2).encode('utf-8')
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, mode='w', compression=zipfile.ZIP_DEFLATED) as z:
                z.writestr('baseline.csv', (r.get('baseline_csv') or ''))
                if r.get('crashed_csv'):
                    z.writestr('crashed.csv', r.get('crashed_csv') or '')
                z.writestr('meta.json', meta_bytes.decode('utf-8'))
            bundle_bytes = buf.getvalue()
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.download_button(
                    'Download baseline',
                    data=(r.get('baseline_csv') or '').encode('utf-8'),
                    file_name=f"{r['name']}_baseline.csv".replace(' ', '_'),
                    mime='text/csv',
                    key=f"sr_dlb_{r['id']}",
                    use_container_width=True,
                )
            with c2:
                if r.get('crashed_csv'):
                    st.download_button(
                        'Download crashed',
                        data=r.get('crashed_csv').encode('utf-8'),
                        file_name=f"{r['name']}_crashed.csv".replace(' ', '_'),
                        mime='text/csv',
                        key=f"sr_dlc_{r['id']}",
                        use_container_width=True,
                    )
                else:
                    st.button('No crashed', disabled=True, key=f"sr_noc_{r['id']}", use_container_width=True)
            with c3:
                st.download_button(
                    'Download meta',
                    data=meta_bytes,
                    file_name=f"{r['name']}_meta.json".replace(' ', '_'),
                    mime='application/json',
                    key=f"sr_meta_{r['id']}",
                    use_container_width=True,
                )
            with c4:
                if st.button('Delete', key=f"sr_del_{r['id']}", use_container_width=True):
                    delete_schedule_run(r['id'])
                    st.rerun()
            st.download_button(
                'Download run bundle (ZIP)',
                data=bundle_bytes,
                file_name=f"{r['name']}_bundle.zip".replace(' ', '_'),
                mime='application/zip',
                key=f"sr_zip_{r['id']}",
                use_container_width=True,
            )
            st.markdown('---')

    with tab2:
        checks = list_submittal_checks()
        if not checks:
            st.caption("No saved submittal checks yet.")
        for c in checks:
            st.markdown(f"**{c['name']}**")
            st.caption(f"Saved: {c['created_at']}  |  ID: {c['id']}")
            payload = json.loads(c["result_json"])
            with st.expander("View details"):
                st.json(payload)
            json_bytes = json.dumps(payload, indent=2).encode('utf-8')
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, mode='w', compression=zipfile.ZIP_DEFLATED) as z:
                z.writestr('result.json', json_bytes.decode('utf-8'))
            bundle_bytes = buf.getvalue()
            c1, c2, c3 = st.columns(3)
            with c1:
                st.download_button(
                    'Download JSON',
                    data=json_bytes,
                    file_name=f"{c['name']}".replace(' ', '_') + '.json',
                    mime='application/json',
                    key=f"sc_dl_{c['id']}",
                    use_container_width=True,
                )
            with c2:
                st.download_button(
                    'Download bundle (ZIP)',
                    data=bundle_bytes,
                    file_name=f"{c['name']}".replace(' ', '_') + '_bundle.zip',
                    mime='application/zip',
                    key=f"sc_zip_{c['id']}",
                    use_container_width=True,
                )
            with c3:
                if st.button('Delete', key=f"sc_del_{c['id']}", use_container_width=True):
                    delete_submittal_check(c['id'])
                    st.rerun()
            st.markdown('---')

    with tab3:
        rows = list_rfis()
        if not rows:
            st.caption("No RFIs yet.")
        else:
            df = pd.DataFrame([dict(r) for r in rows])
            st.dataframe(df, use_container_width=True)
            st.download_button("Download RFIs CSV", data=df.to_csv(index=False).encode("utf-8"), file_name="rfis.csv", mime="text/csv")
