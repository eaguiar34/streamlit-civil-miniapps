# app.py
# FieldFlow: Submittal Checker + Schedule What-Ifs
# - Sidebar pages
# - Hybrid submittal scoring (lexical + semantic + coverage + penalties + section boost)
# - Reviewer keyword presets + memory bank
# - Storage backends: SQLite (local), Google Sheets (Service/OAuth), Microsoft Excel (OAuth)
# - Schedule What-Ifs: CPM with FS/SS/FF lags, overlap, floats, calendar mode, crash-to-target
# - Width shims + auto-heights; safe secrets loader; OAuth helpers
# - Make sure you set secrets (see README snippet at end of this file)

from __future__ import annotations

import os, io, re, json, time, urllib.parse, sqlite3
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from datetime import datetime, date

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import requests

# ==== Robust PDF/OCR/Table extraction helpers (add after imports) ====
import hashlib

def _hash_list(xs: list[str]) -> str:
    h = hashlib.sha256()
    for s in xs:
        h.update((s or "").encode("utf-8"))
    return h.hexdigest()

def _try_import(modname):
    try:
        return __import__(modname)
    except Exception:
        return None

_pytesseract = _try_import("pytesseract")
_pdf2image  = _try_import("pdf2image")
_PIL        = _try_import("PIL")
_tabula     = _try_import("tabula")  # tabula-py

def ocr_pdf_to_text(uploaded_file) -> str:
    """Fallback: rasterize pages and OCR."""
    if not (_pytesseract and _pdf2image and _PIL):
        return ""
    try:
        # pdf2image convert_from_bytes
        pages = _pdf2image.convert_from_bytes(uploaded_file.getvalue(), dpi=200)
        texts = []
        for im in pages:
            texts.append(_pytesseract.image_to_string(im))
        return "\n".join(texts)
    except Exception:
        return ""

def tabula_tables_to_text(uploaded_file) -> str:
    """Extract tables via tabula and linearize as text."""
    if not _tabula:
        return ""
    try:
        dfs = _tabula.read_pdf(uploaded_file, pages="all", multiple_tables=True, lattice=True)
        parts = []
        for df in dfs:
            parts.append("\n".join([" ".join(map(str, row)) for row in df.fillna("").values.tolist()]))
        return "\n\n".join(parts)
    except Exception:
        return ""

@st.cache_resource
def get_embedder():
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def cached_embeddings(chunks: list[str]):
    model = get_embedder()
    if not model:
        return None
    # normalize embeddings so cosine is a dot product
    emb = model.encode(chunks, show_progress_bar=False, normalize_embeddings=True)
    return emb

def bm25_scores(query: str, docs: list[str]) -> list[float]:
    try:
        from rank_bm25 import BM25Okapi
    except Exception:
        return [0.0]*len(docs)
    # simple tokenization
    def tok(s): return re.findall(r"[a-z0-9]+", s.lower())
    corpus = [tok(d) for d in docs]
    bm = BM25Okapi(corpus)
    q = tok(query)
    scores = bm.get_scores(q)
    if not scores.size:
        return [0.0]*len(docs)
    # min-max normalize to [0,1]
    mn, mx = float(scores.min()), float(scores.max())
    if mx <= mn:
        return [0.0]*len(docs)
    return [(float(s)-mn)/(mx-mn) for s in scores]


# =========================
# App bootstrap
# =========================
# ---------- Width shims (handles Streamlit deprecations gracefully)
def df_fullwidth(df, **kwargs):
    try:
        return st.dataframe(df, width='stretch', **kwargs)
    except TypeError:
        return st.dataframe(df, **kwargs)

def editor_fullwidth(df, **kwargs):
    try:
        return st.data_editor(df, width='stretch', **kwargs)
    except TypeError:
        return st.data_editor(df, **kwargs)

def chart_fullwidth(chart, **kwargs):
    try:
        return st.altair_chart(chart, width='stretch', **kwargs)
    except TypeError:
        return st.altair_chart(chart, **kwargs)

_DEF_ROW = 32
_DEF_HEADER = 38
def rows_to_height(n_rows: int, row_px: int=_DEF_ROW, header_px: int=_DEF_HEADER, max_height: int=1400) -> int:
    n = max(1, int(n_rows))
    return min(max_height, header_px + row_px * n)

# ---------- Secrets loader (works with st.secrets or env vars)
def get_secret(path: str, default=None):
    # Streamlit secrets
    try:
        node = st.secrets
        for key in path.split("."):
            if key in node:
                node = node[key]
            else:
                node = None; break
        if node is not None:
            return node
    except Exception:
        pass
    # Env fallback (supports JSON or plain)
    env_key = path.upper().replace(".", "__")
    val = os.getenv(env_key)
    if not val:
        return default
    try:
        return json.loads(val)
    except Exception:
        return val

# =========================
# Pluggable Storage Backends
# =========================
BACKEND_SQLITE = "Local (SQLite)"
BACKEND_GS_SERVICE = "Google Sheets (Service Account)"
BACKEND_GS_OAUTH = "Google Sheets (OAuth)"
BACKEND_MS_OAUTH = "Microsoft 365 Excel (OAuth)"
BACKEND_CHOICES = [BACKEND_SQLITE, BACKEND_GS_SERVICE, BACKEND_GS_OAUTH, BACKEND_MS_OAUTH]

def _ensure_ss(key, default):
    if key not in st.session_state:
        st.session_state[key] = default
    return st.session_state[key]

class StorageBackend:
    def save_preset(self, name: str, payload: dict) -> None: ...
    def load_presets(self) -> dict: ...
    def delete_preset(self, name: str) -> None: ...
    def save_submittal(self, meta: dict, result_csv_bytes: bytes, spec_excerpt: str, sub_excerpt: str) -> int: ...
    def list_submittals(self) -> pd.DataFrame: ...
    def get_submittal(self, id_: int) -> dict: ...
    def delete_submittal(self, id_: int) -> None: ...
    def open_url_hint(self, rec: dict) -> str | None: ...

    # NEW
    def save_feedback(self, row: dict) -> None: ...
    def list_feedback(self, limit: int = 50) -> pd.DataFrame: ...


    # RFI Manager
    def upsert_rfi(self, row: dict) -> int: ...
    def list_rfis(self, limit: int = 5000) -> 'pd.DataFrame': ...
    def get_rfi(self, id_: int) -> dict: ...
    def delete_rfi(self, id_: int) -> None: ...


# ---- SQLite Backend
class SQLiteBackend(StorageBackend):
    def __init__(self, db_path: str):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        con = sqlite3.connect(db_path, check_same_thread=False)
        con.execute("PRAGMA journal_mode=WAL;")
        con.execute("""
        CREATE TABLE IF NOT EXISTS presets (
            name TEXT PRIMARY KEY,
            json TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )""")
        con.execute("""
        CREATE TABLE IF NOT EXISTS submittals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            company TEXT, client TEXT, project TEXT,
            date_submitted TEXT, quote NUMERIC, notes TEXT,
            threshold INT, weights_json TEXT,
            must TEXT, nice TEXT, forbid TEXT,
            pass_count INT, review_count INT, pass_rate REAL,
            result_csv BLOB, spec_excerpt TEXT, submittal_excerpt TEXT,
            created_at TEXT NOT NULL
        )""")
        con.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            user_id TEXT,
            page TEXT,
            rating INT,
            categories TEXT,
            email TEXT,
            message TEXT
        )""")
        con.execute("""
        CREATE TABLE IF NOT EXISTS rfis (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            user_id TEXT,
            project TEXT,
            subject TEXT,
            question TEXT,
            discipline TEXT,
            spec_section TEXT,
            priority TEXT,
            status TEXT,
            due_date TEXT,
            assignee_email TEXT,
            recipient_emails TEXT,
            cc_emails TEXT,
            related_tasks TEXT,
            schedule_impact_days INT,
            cost_impact REAL,
            last_sent_at TEXT,
            last_reminded_at TEXT,
            last_response_at TEXT,
            response_text TEXT,
            notes TEXT
        )""")
        # RFI links + attachments (stored in DB for Local/Streamlit Cloud).
        con.execute("""CREATE TABLE IF NOT EXISTS rfi_links (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            rfi_id INTEGER NOT NULL,
            url TEXT NOT NULL,
            created_at TEXT,
            FOREIGN KEY (rfi_id) REFERENCES rfis(id) ON DELETE CASCADE
        )""")
        con.execute("""CREATE TABLE IF NOT EXISTS rfi_attachments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            rfi_id INTEGER NOT NULL,
            filename TEXT NOT NULL,
            mime TEXT,
            data BLOB NOT NULL,
            uploaded_at TEXT,
            FOREIGN KEY (rfi_id) REFERENCES rfis(id) ON DELETE CASCADE
        )""")

        con.commit()
        self.con = con

    def save_preset(self, name: str, payload: dict) -> None:
        self.con.execute(
            "REPLACE INTO presets(name, json, updated_at) VALUES(?,?,?)",
            (name, json.dumps(payload), datetime.utcnow().isoformat()),
        )
        self.con.commit()

    def load_presets(self) -> dict:
        rows = self.con.execute("SELECT name, json FROM presets").fetchall()
        return {name: json.loads(js) for (name, js) in rows}

    def delete_preset(self, name: str) -> None:
        self.con.execute("DELETE FROM presets WHERE name=?", (name,))
        self.con.commit()

    def save_submittal(self, meta: dict, result_csv_bytes: bytes, spec_excerpt: str, sub_excerpt: str) -> int:
        cur = self.con.cursor()
        cur.execute("""
        INSERT INTO submittals
        (company, client, project, date_submitted, quote, notes,
         threshold, weights_json, must, nice, forbid,
         pass_count, review_count, pass_rate,
         result_csv, spec_excerpt, submittal_excerpt, created_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", (
            meta.get("company"), meta.get("client"), meta.get("project"),
            meta.get("date_submitted"), meta.get("quote"), meta.get("notes"),
            meta.get("threshold"), json.dumps(meta.get("weights", {})),
            meta.get("must"), meta.get("nice"), meta.get("forbid"),
            meta.get("pass_count"), meta.get("review_count"), meta.get("pass_rate"),
            result_csv_bytes, spec_excerpt, sub_excerpt, datetime.utcnow().isoformat()
        ))
        self.con.commit()
        return int(cur.lastrowid)

    def list_submittals(self) -> pd.DataFrame:
        return pd.read_sql_query("""
            SELECT id, company, client, project, date_submitted, quote,
                   pass_count, review_count, pass_rate, threshold, created_at
            FROM submittals
            ORDER BY company, project, date_submitted DESC, created_at DESC
        """, self.con)

    def get_submittal(self, id_: int) -> dict:
        cur = self.con.execute("SELECT * FROM submittals WHERE id=?", (id_,))
        row = cur.fetchone()
        if not row: return {}
        cols = [d[1] for d in self.con.execute("PRAGMA table_info(submittals)")]
        return dict(zip(cols, row))

    def delete_submittal(self, id_: int) -> None:
        self.con.execute("DELETE FROM submittals WHERE id=?", (id_,))
        self.con.commit()

    def save_feedback(self, row: dict) -> None:
        self.con.execute(
            "INSERT INTO feedback(created_at,user_id,page,rating,categories,email,message) VALUES(?,?,?,?,?,?,?)",
            (
                row.get("created_at"),
                row.get("user_id"),
                row.get("page"),
                int(row.get("rating") or 0),
                ",".join(row.get("categories") or []),
                row.get("email"),
                row.get("message"),
            ),
        )
        self.con.commit()

    def list_feedback(self, limit: int = 50) -> pd.DataFrame:
        return pd.read_sql_query(
            "SELECT * FROM feedback ORDER BY id DESC LIMIT ?",
            self.con,
            params=(int(limit),),
        )

    # ---- RFIs
    def upsert_rfi(self, row: dict) -> int:
        """Insert or update an RFI. Returns the RFI id."""
        now = row.get("updated_at") or __import__("datetime").datetime.utcnow().isoformat()
        rid = row.get("id")
        fields = [
            "user_id","project","subject","question","discipline","spec_section","priority",
            "status","due_date","assignee_email","to_emails","cc_emails",
            "related_tasks","schedule_impact_days","cost_impact",
            "last_sent_at","last_reminded_at","last_response_at","thread_notes",
        ]
        vals = [row.get(f) for f in fields]
        if rid:
            self.con.execute(
                """UPDATE rfis SET
                    updated_at=?,
                    user_id=?, project=?, subject=?, question=?, discipline=?, spec_section=?, priority=?,
                    status=?, due_date=?, assignee_email=?, to_emails=?, cc_emails=?,
                    related_tasks=?, schedule_impact_days=?, cost_impact=?,
                    last_sent_at=?, last_reminded_at=?, last_response_at=?, thread_notes=?
                WHERE id=?""",
                [now] + vals + [int(rid)],
            )
            self.con.commit()
            return int(rid)

        cur = self.con.cursor()
        cur.execute(
            """INSERT INTO rfis(
                created_at, updated_at,
                user_id, project, subject, question, discipline, spec_section, priority,
                status, due_date, assignee_email, to_emails, cc_emails,
                related_tasks, schedule_impact_days, cost_impact,
                last_sent_at, last_reminded_at, last_response_at, thread_notes
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            [row.get("created_at") or __import__("datetime").datetime.utcnow().isoformat(), now] + vals,
        )
        self.con.commit()
        return int(cur.lastrowid)

    def list_rfis(self, limit: int = 5000) -> pd.DataFrame:
        return pd.read_sql_query(
            "SELECT * FROM rfis ORDER BY id DESC LIMIT ?",
            self.con,
            params=(int(limit),),
        )

    def get_rfi(self, id_: int) -> dict:
        cur = self.con.execute("SELECT * FROM rfis WHERE id=?", (int(id_),))
        row = cur.fetchone()
        if not row:
            return {}
        cols = [d[1] for d in self.con.execute("PRAGMA table_info(rfis)")]
        return dict(zip(cols, row))

    def delete_rfi(self, id_: int) -> None:
        self.con.execute("DELETE FROM rfis WHERE id=?", (int(id_),))
        self.con.commit()


    def add_rfi_links(self, rfi_id: int, urls: list[str]):
        urls = [u.strip() for u in urls if str(u).strip()]
        if not urls:
            return
        con = self._conn()
        now = datetime.utcnow().isoformat()
        for u in urls:
            con.execute("INSERT INTO rfi_links (rfi_id, url, created_at) VALUES (?, ?, ?)", (rfi_id, u, now))
        con.commit()

    def list_rfi_links(self, rfi_id: int) -> list[dict]:
        con = self._conn()
        rows = con.execute("SELECT id, url, created_at FROM rfi_links WHERE rfi_id=? ORDER BY id DESC", (rfi_id,)).fetchall()
        return [{"id": r[0], "url": r[1], "created_at": r[2]} for r in rows]

    def delete_rfi_link(self, link_id: int):
        con = self._conn()
        con.execute("DELETE FROM rfi_links WHERE id=?", (link_id,))
        con.commit()

    def add_rfi_attachments(self, rfi_id: int, files: list[dict]):
        """files: [{"filename": str, "mime": str|None, "data": bytes}]"""
        if not files:
            return
        con = self._conn()
        now = datetime.utcnow().isoformat()
        for f in files:
            con.execute("INSERT INTO rfi_attachments (rfi_id, filename, mime, data, uploaded_at) VALUES (?, ?, ?, ?, ?)", (rfi_id, f.get('filename') or 'attachment', f.get('mime'), f.get('data') or b'', now))
        con.commit()

    def list_rfi_attachments(self, rfi_id: int) -> list[dict]:
        con = self._conn()
        rows = con.execute("SELECT id, filename, mime, uploaded_at, length(data) FROM rfi_attachments WHERE rfi_id=? ORDER BY id DESC", (rfi_id,)).fetchall()
        return [{"id": r[0], "filename": r[1], "mime": r[2], "uploaded_at": r[3], "size": r[4]} for r in rows]

    def get_rfi_attachment_data(self, attachment_id: int) -> tuple[str, str, bytes]:
        con = self._conn()
        row = con.execute("SELECT filename, COALESCE(mime, ''), data FROM rfi_attachments WHERE id=?", (attachment_id,)).fetchone()
        if not row:
            raise KeyError("Attachment not found")
        return row[0], row[1], row[2]

    def delete_rfi_attachment(self, attachment_id: int):
        con = self._conn()
        con.execute("DELETE FROM rfi_attachments WHERE id=?", (attachment_id,))
        con.commit()

    def open_url_hint(self, rec: dict) -> str | None:
        return None

# ---- Google Sheets (Service Account) Backend
class GoogleSheetsSA(StorageBackend):
    def __init__(self, title: str):
        try:
            import gspread
            from google.oauth2.service_account import Credentials
        except Exception:
            st.error("Missing Google Sheets deps. Install: gspread, google-auth")
            raise
        self.gspread = gspread
        self.Credentials = Credentials
        self.title = title
        self.scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
        self.client = self._client()
        self.ss = self._open_or_create_spreadsheet(self.client)
        self._ensure_tabs()

    @st.cache_resource
    def _client(_self=None):
        creds_info = get_secret("gcp_service_account")
        if not creds_info:
            raise RuntimeError("No service account in secrets.")
        creds = _self.Credentials.from_service_account_info(creds_info, scopes=_self.scopes)
        return _self.gspread.authorize(creds)

    def _open_or_create_spreadsheet(self, client):
        try:
            return client.open(self.title)
        except self.gspread.SpreadsheetNotFound:
            return client.create(self.title)

    def _ensure_ws(self, title, headers):
        try:
            ws = self.ss.worksheet(title)
        except self.gspread.WorksheetNotFound:
            ws = self.ss.add_worksheet(title=title, rows=1000, cols=max(20, len(headers)))
            ws.append_row(headers)
            return ws
        if ws.row_values(1) != headers:
            ws.resize(rows=1)
            ws.update([headers])
        return ws

    def _ensure_tabs(self):
        self._ensure_ws("presets", ["name","json","updated_at"])
        self._ensure_ws("submittals", [
            "id","company","client","project","date_submitted","quote","notes",
            "threshold","weights_json","must","nice","forbid",
            "pass_count","review_count","pass_rate",
            "result_sheet_name","spec_excerpt","submittal_excerpt","created_at"
        ])
        self._ensure_ws("feedback", ["created_at","user_id","page","rating","categories","email","message"])
        self._ensure_ws("rfis", ["id","created_at","updated_at","user_id","project","subject","question","discipline","spec_section","priority","status","due_date","assignee_email","to_emails","cc_emails","related_tasks","schedule_impact_days","cost_impact","last_sent_at","last_reminded_at","last_response_at","thread_notes"])


    def load_presets(self) -> dict:
        rows = self.ss.worksheet("presets").get_all_records()
        out = {}
        for r in rows:
            try: out[r["name"]] = json.loads(r["json"])
            except: pass
        return out

    def save_preset(self, name: str, payload: dict) -> None:
        ws = self.ss.worksheet("presets")
        rows = ws.get_all_records()
        names = [r.get("name") for r in rows]
        now = datetime.utcnow().isoformat()
        if name in names:
            idx = names.index(name) + 2
            ws.update(f"A{idx}:C{idx}", [[name, json.dumps(payload), now]])
        else:
            ws.append_row([name, json.dumps(payload), now])

    def delete_preset(self, name: str) -> None:
        ws = self.ss.worksheet("presets")
        rows = ws.get_all_records()
        for i, r in enumerate(rows, start=2):
            if r.get("name") == name:
                ws.delete_rows(i); return

    def _next_id(self) -> int:
        rows = self.ss.worksheet("submittals").get_all_records()
        if not rows: return 1
        try: return max(int(r.get("id",0) or 0) for r in rows) + 1
        except: return len(rows)+1

    def _sanitize_tab(self, s: str) -> str:
        return re.sub(r"[\[\]\:\*\?\/\\]", "_", s)[:90]

    def save_submittal(self, meta: dict, result_csv_bytes: bytes, spec_excerpt: str, sub_excerpt: str) -> int:
        ws = self.ss.worksheet("submittals")
        run_id = self._next_id()
        # results in new tab
        try:
            df = pd.read_csv(pd.io.common.BytesIO(result_csv_bytes))
            tab = self._sanitize_tab(f"run_{run_id}_{meta.get('company','')}_{meta.get('project','')}")
            try:
                old = self.ss.worksheet(tab); self.ss.del_worksheet(old)
            except: pass
            w = self.ss.add_worksheet(title=tab, rows=max(1000, len(df)+10), cols=max(20, len(df.columns)+2))
            values = [list(df.columns)] + df.astype(object).where(pd.notna(df), "").values.tolist()
            w.update("A1", values)
        except Exception:
            tab = ""
        now = datetime.utcnow().isoformat()
        row = [
            run_id, meta.get("company"), meta.get("client"), meta.get("project"),
            meta.get("date_submitted"), meta.get("quote"), meta.get("notes"),
            meta.get("threshold"), json.dumps(meta.get("weights", {})),
            meta.get("must"), meta.get("nice"), meta.get("forbid"),
            meta.get("pass_count"), meta.get("review_count"), float(meta.get("pass_rate",0.0)),
            tab, spec_excerpt, sub_excerpt, now
        ]
        ws.append_row(row, value_input_option="RAW")
        return run_id

    def list_submittals(self) -> pd.DataFrame:
        rows = self.ss.worksheet("submittals").get_all_records()
        if not rows:
            return pd.DataFrame(columns=["id","company","client","project","date_submitted","quote",
                                         "pass_count","review_count","pass_rate","threshold","created_at"])
        df = pd.DataFrame(rows)
        for c in ["id","pass_count","review_count","threshold"]:
            if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
        if "pass_rate" in df.columns: df["pass_rate"] = pd.to_numeric(df["pass_rate"], errors="coerce").fillna(0.0)
        return df[["id","company","client","project","date_submitted","quote",
                   "pass_count","review_count","pass_rate","threshold","created_at"]]

    def get_submittal(self, id_: int) -> dict:
        for r in self.ss.worksheet("submittals").get_all_records():
            try:
                if int(r.get("id",0) or 0) == int(id_): return r
            except: pass
        return {}

    def delete_submittal(self, id_: int) -> None:
        ws = self.ss.worksheet("submittals")
        rows = ws.get_all_records()
        for i, r in enumerate(rows, start=2):
            try:
                if int(r.get("id",0) or 0) == int(id_):
                    tab = r.get("result_sheet_name") or ""
                    if tab:
                        try: self.ss.del_worksheet(self.ss.worksheet(tab))
                        except: pass
                    ws.delete_rows(i); return
            except: continue

    def save_feedback(self, row: dict) -> None:
        ws = self.ss.worksheet("feedback")
        ws.append_row([
            row.get("created_at"),
            row.get("user_id"),
            row.get("page"),
            int(row.get("rating") or 0),
            ",".join(row.get("categories") or []),
            row.get("email"),
            row.get("message"),
        ], value_input_option="RAW")

    def list_feedback(self, limit: int = 50) -> pd.DataFrame:
        rows = self.ss.worksheet("feedback").get_all_records()
        if not rows:
            return pd.DataFrame(columns=["created_at","user_id","page","rating","categories","email","message"])
        df = pd.DataFrame(rows)
        df["rating"] = pd.to_numeric(df.get("rating", 0), errors="coerce").fillna(0).astype(int)
        return df.sort_values("created_at", ascending=False).head(limit)

    # ---- RFIs
    def _next_rfi_id(self) -> int:
        rows = self.ss.worksheet("rfis").get_all_records()
        if not rows:
            return 1
        try:
            return max(int(r.get("id", 0) or 0) for r in rows) + 1
        except Exception:
            return len(rows) + 1

    def upsert_rfi(self, row: dict) -> int:
        ws = self.ss.worksheet("rfis")
        headers = ws.row_values(1)
        # Ensure header shape
        expected = ["id","created_at","updated_at","user_id","project","subject","question","discipline","spec_section","priority","status","due_date","assignee_email","to_emails","cc_emails","related_tasks","schedule_impact_days","cost_impact","last_sent_at","last_reminded_at","last_response_at","thread_notes"]
        if headers != expected:
            ws.resize(rows=1)
            ws.update([expected])

        rows = ws.get_all_records()
        rid = row.get("id")
        if not rid:
            rid = self._next_rfi_id()
            row["id"] = rid
            row.setdefault("created_at", datetime.utcnow().isoformat())

        row.setdefault("updated_at", datetime.utcnow().isoformat())

        out = [
            row.get("id"), row.get("created_at"), row.get("updated_at"), row.get("user_id"),
            row.get("project"), row.get("subject"), row.get("question"), row.get("discipline"),
            row.get("spec_section"), row.get("priority"), row.get("status"), row.get("due_date"),
            row.get("assignee_email"), row.get("to_emails"), row.get("cc_emails"),
            row.get("related_tasks"), row.get("schedule_impact_days"), row.get("cost_impact"),
            row.get("last_sent_at"), row.get("last_reminded_at"), row.get("last_response_at"), row.get("thread_notes"),
        ]

        # Update if exists
        ids = []
        for r in rows:
            try:
                ids.append(int(r.get("id", 0) or 0))
            except Exception:
                ids.append(0)
        if int(rid) in ids:
            idx = ids.index(int(rid)) + 2
            ws.update(f"A{idx}:V{idx}", [out], value_input_option="RAW")
        else:
            ws.append_row(out, value_input_option="RAW")
        return int(rid)

    def list_rfis(self, limit: int = 5000) -> pd.DataFrame:
        rows = self.ss.worksheet("rfis").get_all_records()
        if not rows:
            return pd.DataFrame(columns=["id","created_at","updated_at","user_id","project","subject","question","discipline","spec_section","priority","status","due_date","assignee_email","to_emails","cc_emails","related_tasks","schedule_impact_days","cost_impact","last_sent_at","last_reminded_at","last_response_at","thread_notes"])
        df = pd.DataFrame(rows)
        for c in ["id","schedule_impact_days"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
        for c in ["cost_impact"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        return df.sort_values("id", ascending=False).head(int(limit))

    def get_rfi(self, id_: int) -> dict:
        rows = self.ss.worksheet("rfis").get_all_records()
        for r in rows:
            try:
                if int(r.get("id", 0) or 0) == int(id_):
                    return r
            except Exception:
                continue
        return {}

    def delete_rfi(self, id_: int) -> None:
        ws = self.ss.worksheet("rfis")
        rows = ws.get_all_records()
        for i, r in enumerate(rows, start=2):
            try:
                if int(r.get("id", 0) or 0) == int(id_):
                    ws.delete_rows(i)
                    return
            except Exception:
                continue


    def open_url_hint(self, rec: dict) -> str | None:
        try: return self.ss.url
        except: return None

# ---- Google OAuth helpers + Sheets backend
def _qp_value(params, key):
    """Return a single query-param value, handling Streamlit's list values."""
    v = params.get(key)
    if isinstance(v, list):
        return v[0] if v else None
    return v

def google_oauth_start():
    import secrets as _secrets
    from google_auth_oauthlib.flow import Flow

    cfg = get_secret("google_oauth", {})
    if not cfg:
        st.error("Google OAuth not configured in secrets.")
        return

    state = "g_" + _secrets.token_urlsafe(16)

    flow = Flow.from_client_config(
        {
            "web": {
                "client_id": cfg["client_id"],
                "client_secret": cfg["client_secret"],
                "redirect_uris": [cfg["redirect_uri"]],
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
            }
        },
        scopes=[
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive.file",
            "openid", "email", "profile",
        ],
        state=state,
    )
    flow.redirect_uri = cfg["redirect_uri"]

    auth_url, _ = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        prompt="consent",
    )

    # Best-effort CSRF protection (may be lost across redirects on Streamlit Cloud).
    st.session_state["__google_state__"] = state

    st.markdown(f"[Continue to Google]({auth_url})")
def google_oauth_callback():
    """Handle the OAuth redirect back from Google."""
    from google_auth_oauthlib.flow import Flow

    cfg = get_secret("google_oauth", {})
    params = st.query_params

    code = _qp_value(params, "code")
    state = _qp_value(params, "state")
    if not code or not state:
        return False

    # Ignore if this callback isn't for Google (helps when multiple providers share ?code=...).
    if isinstance(state, str) and not state.startswith("g_"):
        return False

    # Guard against double-processing the same authorization code (refresh/back button can trigger this)
    used = st.session_state.setdefault("__oauth_used_codes__", set())
    if code in used:
        st.query_params.clear()
        return False
    used.add(code)

    # If we still have the original state, enforce it. (Streamlit Cloud may lose it.)
    saved_state = st.session_state.get("__google_state__")
    if saved_state is not None and state != saved_state:
        return False

    # Ignore callbacks that aren't for Google (we prefix state with 'g_').
    if isinstance(state, str) and not state.startswith("g_"):
        return False

    flow = Flow.from_client_config(
        {
            "web": {
                "client_id": cfg["client_id"],
                "client_secret": cfg["client_secret"],
                "redirect_uris": [cfg["redirect_uri"]],
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
            }
        },
        scopes=[
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive.file",
            "openid", "email", "profile",
        ],
        state=state,
    )
    flow.redirect_uri = cfg["redirect_uri"]
    flow.fetch_token(code=code)

    creds = flow.credentials
    st.session_state["__google_token__"] = {
        "token": creds.token,
        "refresh_token": getattr(creds, "refresh_token", None),
        "token_uri": creds.token_uri,
        "client_id": creds.client_id,
        "client_secret": creds.client_secret,
        "scopes": creds.scopes,
    }

    ui = requests.get(
        "https://www.googleapis.com/oauth2/v3/userinfo",
        headers={"Authorization": f"Bearer {creds.token}"},
        timeout=15,
    ).json()
    st.session_state["__google_user__"] = ui.get("email", "google-user")

    st.query_params.clear()
    return True
def google_credentials():
    """Return Google Credentials (OAuth), refreshing if needed."""
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request

    tok = st.session_state.get("__google_token__")
    if not tok:
        return None

    creds = Credentials(**tok)

    # Best-effort refresh if expired and we have a refresh token.
    try:
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
            st.session_state["__google_token__"] = {
                "token": creds.token,
                "refresh_token": creds.refresh_token,
                "token_uri": creds.token_uri,
                "client_id": creds.client_id,
                "client_secret": creds.client_secret,
                "scopes": creds.scopes,
            }
    except Exception:
        # If refresh fails, fall back to the existing creds; caller can re-auth.
        pass

    return creds
class GoogleSheetsOAuth(StorageBackend):
    def __init__(self, title: str):
        import gspread
        creds = google_credentials()
        if creds is None:
            raise RuntimeError("Google not authenticated.")
        self.client = gspread.authorize(creds)
        self.title = title
        self.ss = self._open_or_create(self.client)
        self._ensure_tabs()

    def _open_or_create(self, client):
        try:
            return client.open(self.title)
        except Exception:
            return client.create(self.title)

    def _ensure_ws(self, title, headers):
        try:
            ws = self.ss.worksheet(title)
        except Exception:
            ws = self.ss.add_worksheet(title=title, rows=1000, cols=max(20, len(headers)))
            ws.append_row(headers)
            return ws
        if ws.row_values(1) != headers:
            ws.resize(rows=1)
            ws.update([headers])
        return ws

    def _ensure_tabs(self):
        self._ensure_ws("presets", ["name","json","updated_at"])
        self._ensure_ws("submittals", [
            "id","company","client","project","date_submitted","quote","notes",
            "threshold","weights_json","must","nice","forbid",
            "pass_count","review_count","pass_rate",
            "result_sheet_name","spec_excerpt","submittal_excerpt","created_at"
        ])
        self._ensure_ws("feedback", ["created_at","user_id","page","rating","categories","email","message"])
        self._ensure_ws("rfis", [
            "id","created_at","updated_at","user_id","project","subject","question","discipline","spec_section",
            "priority","status","due_date","assignee_email","to_emails","cc_emails","related_tasks",
            "schedule_impact_days","cost_impact","last_sent_at","last_reminded_at","last_response_at","thread_notes"
        ])

    def load_presets(self) -> dict:
        rows = self.ss.worksheet("presets").get_all_records()
        out = {}
        for r in rows:
            try: out[r["name"]] = json.loads(r["json"])
            except: pass
        return out

    def save_preset(self, name, payload):
        ws = self.ss.worksheet("presets")
        rows = ws.get_all_records()
        names = [r.get("name") for r in rows]
        now = datetime.utcnow().isoformat()
        if name in names:
            idx = names.index(name) + 2
            ws.update(f"A{idx}:C{idx}", [[name, json.dumps(payload), now]])
        else:
            ws.append_row([name, json.dumps(payload), now])

    def delete_preset(self, name):
        ws = self.ss.worksheet("presets")
        rows = ws.get_all_records()
        for i, r in enumerate(rows, start=2):
            if r.get("name")==name:
                ws.delete_rows(i); return

    def _next_id(self):
        rows = self.ss.worksheet("submittals").get_all_records()
        if not rows: return 1
        try: return max(int(r.get("id",0) or 0) for r in rows) + 1
        except: return len(rows)+1

    def save_submittal(self, meta, result_csv_bytes, spec_excerpt, sub_excerpt):
        ws = self.ss.worksheet("submittals")
        run_id = self._next_id()
        # results tab
        try:
            df = pd.read_csv(pd.io.common.BytesIO(result_csv_bytes))
            tab = re.sub(r"[\[\]\:\*\?\/\\]", "_", f"run_{run_id}_{meta.get('company','')}_{meta.get('project','')}")[:90]
            try:
                old = self.ss.worksheet(tab); self.ss.del_worksheet(old)
            except: pass
            w = self.ss.add_worksheet(title=tab, rows=max(1000,len(df)+10), cols=max(20,len(df.columns)+2))
            values = [list(df.columns)] + df.astype(object).where(pd.notna(df), "").values.tolist()
            w.update("A1", values)
        except Exception:
            tab = ""
        now = datetime.utcnow().isoformat()
        row = [
            run_id, meta.get("company"), meta.get("client"), meta.get("project"),
            meta.get("date_submitted"), meta.get("quote"), meta.get("notes"),
            meta.get("threshold"), json.dumps(meta.get("weights", {})),
            meta.get("must"), meta.get("nice"), meta.get("forbid"),
            meta.get("pass_count"), meta.get("review_count"), float(meta.get("pass_rate",0.0)),
            tab, spec_excerpt, sub_excerpt, now
        ]
        ws.append_row(row); return run_id

    def list_submittals(self) -> pd.DataFrame:
        rows = self.ss.worksheet("submittals").get_all_records()
        if not rows:
            return pd.DataFrame(columns=["id","company","client","project","date_submitted","quote",
                                         "pass_count","review_count","pass_rate","threshold","created_at"])
        df = pd.DataFrame(rows)
        for c in ["id","pass_count","review_count","threshold"]:
            if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
        if "pass_rate" in df.columns: df["pass_rate"] = pd.to_numeric(df["pass_rate"], errors="coerce").fillna(0.0)
        return df[["id","company","client","project","date_submitted","quote",
                   "pass_count","review_count","pass_rate","threshold","created_at"]]

    def get_submittal(self, id_):
        for r in self.ss.worksheet("submittals").get_all_records():
            try:
                if int(r.get("id",0) or 0)==int(id_): return r
            except: pass
        return {}

    def delete_submittal(self, id_):
        ws = self.ss.worksheet("submittals")
        rows = ws.get_all_records()
        for i, r in enumerate(rows, start=2):
            if r.get("id")==id_:
                ws.delete_rows(i); return

    def save_feedback(self, row: dict) -> None:
        ws = self.ss.worksheet("feedback")
        ws.append_row([
            row.get("created_at"),
            row.get("user_id"),
            row.get("page"),
            int(row.get("rating") or 0),
            ",".join(row.get("categories") or []),
            row.get("email"),
            row.get("message"),
        ], value_input_option="RAW")

    def list_feedback(self, limit: int = 50) -> pd.DataFrame:
        rows = self.ss.worksheet("feedback").get_all_records()
        if not rows:
            return pd.DataFrame(columns=["created_at","user_id","page","rating","categories","email","message"])
        df = pd.DataFrame(rows)
        df["rating"] = pd.to_numeric(df.get("rating", 0), errors="coerce").fillna(0).astype(int)
        return df.sort_values("created_at", ascending=False).head(limit)


    # ---- RFIs
    def _next_rfi_id(self) -> int:
        rows = self.ss.worksheet("rfis").get_all_records()
        if not rows:
            return 1
        try:
            return max(int(r.get("id", 0) or 0) for r in rows) + 1
        except Exception:
            return len(rows) + 1

    def upsert_rfi(self, row: dict) -> int:
        ws = self.ss.worksheet("rfis")
        rows = ws.get_all_records()
        headers = ws.row_values(1)
        rid = int(row.get("id") or 0)
        if rid <= 0:
            rid = self._next_rfi_id()
            row["id"] = rid
            row.setdefault("created_at", datetime.utcnow().isoformat())
        row.setdefault("updated_at", datetime.utcnow().isoformat())

        def as_cells(hs: list[str]):
            out = []
            for h in hs:
                v = row.get(h)
                if isinstance(v, list):
                    v = ",".join(map(str, v))
                out.append("" if v is None else v)
            return out

        ids = [int(r.get("id", 0) or 0) for r in rows]
        if rid in ids:
            idx = ids.index(rid) + 2
            ws.update(f"A{idx}:{chr(64+len(headers))}{idx}", [as_cells(headers)])
        else:
            ws.append_row(as_cells(headers), value_input_option="RAW")
        return rid

    def list_rfis(self, limit: int = 5000) -> pd.DataFrame:
        rows = self.ss.worksheet("rfis").get_all_records()
        if not rows:
            return pd.DataFrame(columns=[
                "id","created_at","updated_at","user_id","project","subject","question","discipline","spec_section",
                "priority","status","due_date","assignee_email","to_emails","cc_emails","related_tasks",
                "schedule_impact_days","cost_impact","last_sent_at","last_reminded_at","last_response_at","thread_notes"
            ])
        df = pd.DataFrame(rows)
        for c in ["id","schedule_impact_days"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
        if "cost_impact" in df.columns:
            df["cost_impact"] = pd.to_numeric(df["cost_impact"], errors="coerce").fillna(0.0)
        return df.sort_values("id", ascending=False).head(int(limit))

    def get_rfi(self, id_: int) -> dict:
        for r in self.ss.worksheet("rfis").get_all_records():
            try:
                if int(r.get("id", 0) or 0) == int(id_):
                    return r
            except Exception:
                continue
        return {}

    def delete_rfi(self, id_: int) -> None:
        ws = self.ss.worksheet("rfis")
        rows = ws.get_all_records()
        for i, r in enumerate(rows, start=2):
            try:
                if int(r.get("id", 0) or 0) == int(id_):
                    ws.delete_rows(i)
                    return
            except Exception:
                continue

    def open_url_hint(self, rec): 
        try: return self.ss.url
        except: return None

# ---- Microsoft OAuth helpers + Excel backend
def ms_oauth_start():
    import secrets as _secrets

    cfg = get_secret("microsoft_oauth", {})
    if not cfg:
        st.error("Microsoft OAuth not configured in secrets.")
        return

    state = "m_" + _secrets.token_urlsafe(16)

    params = {
        "client_id": cfg["client_id"],
        "response_type": "code",
        "redirect_uri": cfg["redirect_uri"],
        "response_mode": "query",
        "scope": " ".join(["openid", "profile", "email", "offline_access", "Files.ReadWrite", "User.Read"]),
        "state": state,
    }

    st.session_state["__ms_state__"] = state

    auth_url = "https://login.microsoftonline.com/{tenant}/oauth2/v2.0/authorize?{q}".format(
        tenant=cfg.get("tenant_id", "common"),
        q=urllib.parse.urlencode(params),
    )
    st.markdown(f"[Continue to Microsoft]({auth_url})")
def ms_oauth_callback():
    cfg = get_secret("microsoft_oauth", {})
    params = st.query_params

    code = _qp_value(params, "code")
    state = _qp_value(params, "state")
    if not code or not state:
        return False

    used = st.session_state.setdefault("__oauth_used_codes__", set())
    if code in used:
        st.query_params.clear()
        return False
    used.add(code)

    saved_state = st.session_state.get("__ms_state__")
    if saved_state is not None and state != saved_state:
        return False

    if isinstance(state, str) and not state.startswith("m_"):
        return False

    token_url = f"https://login.microsoftonline.com/{cfg.get('tenant_id','common')}/oauth2/v2.0/token"
    data = {
        "client_id": cfg["client_id"],
        "client_secret": cfg["client_secret"],
        "code": code,
        "redirect_uri": cfg["redirect_uri"],
        "grant_type": "authorization_code",
        "scope": "openid profile email offline_access Files.ReadWrite User.Read",
    }

    tok = requests.post(token_url, data=data, timeout=20).json()
    if "access_token" not in tok:
        return False

    tok["_obtained"] = int(time.time())
    st.session_state["__ms_token__"] = tok

    me = requests.get(
        "https://graph.microsoft.com/v1.0/me",
        headers={"Authorization": f"Bearer {tok['access_token']}"},
        timeout=20,
    ).json()
    st.session_state["__ms_user__"] = me.get("userPrincipalName", "microsoft-user")

    st.query_params.clear()
    return True
def ms_access_token():
    """Return Microsoft Graph access token, refreshing if needed."""
    tok = st.session_state.get("__ms_token__")
    if not tok:
        return None

    access = tok.get("access_token")
    if not access:
        return None

    # Refresh if we know it's expired and we have a refresh token.
    try:
        obtained = int(tok.get("_obtained", 0))
        expires_in = int(tok.get("expires_in", 0))
        # Refresh a little early (60s).
        if tok.get("refresh_token") and expires_in and obtained and (time.time() > obtained + expires_in - 60):
            cfg = get_secret("microsoft_oauth", {})
            token_url = f"https://login.microsoftonline.com/{cfg.get('tenant_id','common')}/oauth2/v2.0/token"
            data = {
                "client_id": cfg["client_id"],
                "client_secret": cfg["client_secret"],
                "grant_type": "refresh_token",
                "refresh_token": tok["refresh_token"],
                "redirect_uri": cfg["redirect_uri"],
                "scope": "openid profile email offline_access Files.ReadWrite User.Read",
            }
            new_tok = requests.post(token_url, data=data, timeout=20).json()
            if "access_token" in new_tok:
                new_tok["_obtained"] = int(time.time())
                st.session_state["__ms_token__"] = new_tok
                return new_tok["access_token"]
    except Exception:
        pass

    return access
class MSExcelOAuth(StorageBackend):
    def __init__(self, title: str):
        self.base = "https://graph.microsoft.com/v1.0"
        self.token = ms_access_token()
        if not self.token:
            raise RuntimeError("Microsoft not authenticated.")
        self.title = title
        # Don't create/find the workbook on every rerun (can trip Graph quota).
        # We lazily ensure it when we actually need to write, and cache the id in st.session_state.
        self._workbook_id = st.session_state.get("__ms_workbook_id__") if hasattr(st, "session_state") else None

    def _hed(self): return {"Authorization": f"Bearer {self.token}", "Content-Type":"application/json"}

    def is_initialized(self) -> bool:
        return bool(self._workbook_id)

    def init_workbook(self) -> str:
        """Create/find the workbook once and cache its id in session_state."""
        self._workbook_id = self._ensure_workbook()
        try:
            st.session_state["__ms_workbook_id__"] = self._workbook_id
        except Exception:
            pass
        return self._workbook_id

    def _get_workbook_id(self) -> str:
        """Return workbook id if initialized; otherwise raise a friendly error."""
        if self._workbook_id:
            return self._workbook_id
        raise RuntimeError(
            "OneDrive workbook not initialized. Open **Settings & Examples** and click "
            "**Initialize OneDrive workbook** once."
        )

    def _ensure_workbook(self):
        r_resp = requests.get(f"{self.base}/me/drive/root/children", headers=self._hed(), timeout=20)
        try:
            r = r_resp.json()
        except Exception:
            raise RuntimeError(f"Microsoft Graph returned non-JSON when listing Drive root (status {r_resp.status_code}).")

        if isinstance(r, dict) and "error" in r:
            err = r.get("error", {})
            raise RuntimeError(f"Microsoft Graph error listing Drive root: {err.get('code','')} {err.get('message','')}")

        wid = None
        for item in r.get("value", []):
            if item.get("name") == f"{self.title}.xlsx":
                wid = item.get("id")
                break

        if not wid:
            payload = {"name": f"{self.title}.xlsx", "file": {}}
            c_resp = requests.post(
                f"{self.base}/me/drive/root/children",
                headers=self._hed(),
                data=json.dumps(payload),
                timeout=20,
            )
            try:
                created = c_resp.json()
            except Exception:
                raise RuntimeError(f"Microsoft Graph returned non-JSON when creating workbook (status {c_resp.status_code}).")

            if isinstance(created, dict) and "error" in created:
                err = created.get("error", {})
                raise RuntimeError(f"Microsoft Graph error creating workbook: {err.get('code','')} {err.get('message','')}")

            wid = created.get("id")
            if not wid:
                raise RuntimeError(f"Workbook creation did not return an id (status {c_resp.status_code}). Response keys: {list(created.keys()) if isinstance(created, dict) else type(created)}")

        return wid

    def _ensure_worksheet(self, name: str, headers: list[str]):
        url_ws = f"{self.base}/me/drive/items/{self._get_workbook_id()}/workbook/worksheets"
        r = requests.get(url_ws, headers=self._hed()).json()
        names = [w["name"] for w in r.get("value",[])]
        if name not in names:
            _ = requests.post(url_ws + "/add", headers=self._hed(), data=json.dumps({"name": name})).json()
            self._update_range(name, "A1", [headers])
        # ensure feedback sheet exists for feedback operations
        if name != "feedback":
            self._ensure_worksheet("feedback", ["created_at","user_id","page","rating","categories","email","message"])


    @staticmethod
    def _col_name(n: int) -> str:
        """1 -> A, 2 -> B, 27 -> AA"""
        name = ""
        while n > 0:
            n, r = divmod(n - 1, 26)
            name = chr(65 + r) + name
        return name

    def _update_range(self, ws_name: str, start_cell: str, values_2d: list[list]):
        """Patch a 2D array into a worksheet starting at start_cell (e.g., A1)."""
        m2 = re.match(r"^([A-Z]+)(\d+)$", str(start_cell).upper().strip())
        if not m2:
            raise ValueError(f"Bad start_cell: {start_cell}")
        col_letters, row_str = m2.group(1), m2.group(2)
        start_row = int(row_str)

        nrows = max(1, len(values_2d))
        ncols = max(1, max((len(r) for r in values_2d), default=1))
        start_col_num = 0
        for ch in col_letters:
            start_col_num = start_col_num * 26 + (ord(ch) - 64)
        end_col_num = start_col_num + ncols - 1
        end_row = start_row + nrows - 1
        end_cell = f"{self._col_name(end_col_num)}{end_row}"
        address = f"{col_letters}{start_row}:{end_cell}"

        url = f"{self.base}/me/drive/items/{self._get_workbook_id()}/workbook/worksheets/{urllib.parse.quote(ws_name)}/range(address='{address}')"
        requests.patch(url, headers=self._hed(), data=json.dumps({"values": values_2d}))

    def _append_rows(self, ws_name: str, values_2d: list[list]):
        url = f"{self.base}/me/drive/items/{self._get_workbook_id()}/workbook/worksheets/{urllib.parse.quote(ws_name)}/usedRange(valuesOnly=true)"
        r = requests.get(url, headers=self._hed()).json()
        row_count = r.get("rowCount", 0) or 0
        start_cell = f"A{row_count+1}" if row_count>0 else "A1"
        self._update_range(ws_name, start_cell, values_2d)

    def save_feedback(self, row: dict) -> None:
        self._ensure_worksheet("feedback", ["created_at","user_id","page","rating","categories","email","message"])
        self._append_rows("feedback", [[
            row.get("created_at"),
            row.get("user_id"),
            row.get("page"),
            int(row.get("rating") or 0),
            ",".join(row.get("categories") or []),
            row.get("email"),
            row.get("message"),
        ]])

    def list_feedback(self, limit: int = 50) -> pd.DataFrame:
        self._ensure_worksheet("feedback", ["created_at","user_id","page","rating","categories","email","message"])
        url = f"{self.base}/me/drive/items/{self._get_workbook_id()}/workbook/worksheets/{urllib.parse.quote('feedback')}/usedRange(valuesOnly=true)"
        r = requests.get(url, headers=self._hed()).json()
        vals = r.get("values", [])
        if len(vals) <= 1:
            return pd.DataFrame(columns=["created_at","user_id","page","rating","categories","email","message"])
        df = pd.DataFrame(vals[1:], columns=vals[0])
        df["rating"] = pd.to_numeric(df.get("rating", 0), errors="coerce").fillna(0).astype(int)
        return df.sort_values("created_at", ascending=False).head(limit)


    # ---- RFIs
    def upsert_rfi(self, row: dict) -> int:
        ws = "rfis"
        headers = [
            "id","created_at","updated_at","user_id","project","subject","question","discipline","spec_section",
            "priority","status","due_date","assignee_email","to_emails","cc_emails","related_tasks",
            "schedule_impact_days","cost_impact","last_sent_at","last_reminded_at","last_response_at","thread_notes"
        ]
        self._ensure_worksheet(ws, headers)
        # Load current
        url = f"{self.base}/me/drive/items/{self._get_workbook_id()}/workbook/worksheets/{urllib.parse.quote(ws)}/usedRange(valuesOnly=true)"
        r = requests.get(url, headers=self._hed()).json()
        vals = r.get("values", [])
        existing = vals[1:] if len(vals) > 1 else []
        id_idx = 0
        rid = row.get("id")
        if rid is not None:
            try:
                rid = int(rid)
            except Exception:
                rid = None
        if rid is None:
            # next id
            mx = 0
            for rr in existing:
                try:
                    mx = max(mx, int(rr[id_idx]))
                except Exception:
                    pass
            rid = mx + 1
            row["id"] = rid
        # Build output row values in header order
        def _get(k):
            v = row.get(k)
            if v is None:
                return ""
            if isinstance(v, (list, tuple)):
                return ",".join(str(x) for x in v)
            return str(v)
        out_row = [_get(h) for h in headers]
        # Update if exists
        found_idx = None
        for i, rr in enumerate(existing, start=2):
            try:
                if int(rr[id_idx]) == int(rid):
                    found_idx = i
                    break
            except Exception:
                continue
        if found_idx is None:
            self._append_rows(ws, [out_row])
        else:
            # Update entire row
            self._update_range(ws, f"A{found_idx}", [out_row])
        return int(rid)

    def list_rfis(self, limit: int = 5000) -> pd.DataFrame:
        ws = "rfis"
        headers = [
            "id","created_at","updated_at","user_id","project","subject","question","discipline","spec_section",
            "priority","status","due_date","assignee_email","to_emails","cc_emails","related_tasks",
            "schedule_impact_days","cost_impact","last_sent_at","last_reminded_at","last_response_at","thread_notes"
        ]
        self._ensure_worksheet(ws, headers)
        url = f"{self.base}/me/drive/items/{self._get_workbook_id()}/workbook/worksheets/{urllib.parse.quote(ws)}/usedRange(valuesOnly=true)"
        r = requests.get(url, headers=self._hed()).json()
        vals = r.get("values", [])
        if len(vals) <= 1:
            return pd.DataFrame(columns=headers)
        df = pd.DataFrame(vals[1:], columns=vals[0])
        if "id" in df.columns:
            df["id"] = pd.to_numeric(df["id"], errors="coerce").fillna(0).astype(int)
        if "schedule_impact_days" in df.columns:
            df["schedule_impact_days"] = pd.to_numeric(df["schedule_impact_days"], errors="coerce").fillna(0).astype(int)
        if "cost_impact" in df.columns:
            df["cost_impact"] = pd.to_numeric(df["cost_impact"], errors="coerce").fillna(0.0)
        df = df.sort_values(["status","due_date","created_at"], ascending=[True, True, False], kind="stable")
        return df.head(int(limit))

    def get_rfi(self, id_: int) -> dict:
        df = self.list_rfis(limit=10000)
        try:
            return df[df["id"] == int(id_)].iloc[0].to_dict()
        except Exception:
            return {}

    def delete_rfi(self, id_: int) -> None:
        # Minimal viable (Excel row delete via Graph is non-trivial). Mark as Closed instead.
        rec = self.get_rfi(id_)
        if not rec:
            return
        rec["status"] = "Closed"
        rec["updated_at"] = __import__("datetime").datetime.utcnow().isoformat()
        self.upsert_rfi(rec)

    # Presets
    def load_presets(self) -> dict:
        ws = "presets"; headers = ["name","json","updated_at"]
        self._ensure_worksheet(ws, headers)
        url = f"{self.base}/me/drive/items/{self._get_workbook_id()}/workbook/worksheets/{urllib.parse.quote(ws)}/usedRange(valuesOnly=true)"
        r = requests.get(url, headers=self._hed()).json()
        vals = r.get("values", [])
        out = {}
        for row in vals[1:]:
            if len(row)>=2:
                try: out[row[0]] = json.loads(row[1])
                except: pass
        return out

    def save_preset(self, name: str, payload: dict) -> None:
        ws = "presets"; headers = ["name","json","updated_at"]
        self._ensure_worksheet(ws, headers)
        self._append_rows(ws, [[name, json.dumps(payload), datetime.utcnow().isoformat()]])

    def delete_preset(self, name: str) -> None:
        pass  # minimal viable; range delete omitted

    # Submittals
    def save_submittal(self, meta: dict, result_csv_bytes: bytes, spec_excerpt: str, sub_excerpt: str) -> int:
        ws = "submittals"; headers = ["id","company","client","project","date_submitted","quote","notes",
                                      "threshold","weights_json","must","nice","forbid",
                                      "pass_count","review_count","pass_rate",
                                      "result_sheet_name","spec_excerpt","submittal_excerpt","created_at"]
        self._ensure_worksheet(ws, headers)
        url = f"{self.base}/me/drive/items/{self._get_workbook_id()}/workbook/worksheets/{urllib.parse.quote(ws)}/usedRange(valuesOnly=true)"
        r = requests.get(url, headers=self._hed()).json()
        vals = r.get("values", [])
        run_id = max(1, len(vals))  # naive id

        tab = f"run_{run_id}_{(meta.get('company') or '')[:30]}_{(meta.get('project') or '')[:30]}"
        self._ensure_worksheet(tab, ["Results"])
        try:
            df = pd.read_csv(pd.io.common.BytesIO(result_csv_bytes))
            vals2 = [list(df.columns)] + df.astype(object).where(pd.notna(df), "").values.tolist()
            self._update_range(tab, "A1", vals2)
        except Exception:
            tab = ""

        row = [
            run_id, meta.get("company"), meta.get("client"), meta.get("project"),
            meta.get("date_submitted"), meta.get("quote"), meta.get("notes"),
            meta.get("threshold"), json.dumps(meta.get("weights", {})),
            meta.get("must"), meta.get("nice"), meta.get("forbid"),
            meta.get("pass_count"), meta.get("review_count"), float(meta.get("pass_rate",0.0)),
            tab, spec_excerpt, sub_excerpt, datetime.utcnow().isoformat()
        ]
        self._append_rows(ws, [row])
        return run_id

    def list_submittals(self) -> pd.DataFrame:
        ws = "submittals"; headers = ["id","company","client","project","date_submitted","quote","notes",
                                      "threshold","weights_json","must","nice","forbid",
                                      "pass_count","review_count","pass_rate",
                                      "result_sheet_name","spec_excerpt","submittal_excerpt","created_at"]
        self._ensure_worksheet(ws, headers)
        url = f"{self.base}/me/drive/items/{self._get_workbook_id()}/workbook/worksheets/{urllib.parse.quote(ws)}/usedRange(valuesOnly=true)"
        r = requests.get(url, headers=self._hed()).json()
        vals = r.get("values", [])
        if len(vals) <= 1:
            return pd.DataFrame(columns=["id","company","client","project","date_submitted","quote",
                                         "pass_count","review_count","pass_rate","threshold","created_at"])
        df = pd.DataFrame(vals[1:], columns=vals[0])
        for c in ["id","pass_count","review_count","threshold"]:
            if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
        if "pass_rate" in df.columns: df["pass_rate"] = pd.to_numeric(df["pass_rate"], errors="coerce").fillna(0.0)
        return df[["id","company","client","project","date_submitted","quote",
                   "pass_count","review_count","pass_rate","threshold","created_at"]]

    def get_submittal(self, id_: int) -> dict:
        df = self.list_submittals()
        try:
            return df[df["id"]==int(id_)].iloc[0].to_dict()
        except Exception:
            return {}

    def delete_submittal(self, id_: int) -> None:
        pass

    def open_url_hint(self, rec: dict) -> str | None:
        return "https://www.office.com/launch/excel"

def get_backend(kind: str) -> StorageBackend:
    """Create a backend instance.

    NOTE: OAuth-backed backends are intentionally NOT cached because their behavior depends
    on per-session OAuth tokens stored in st.session_state.
    """
    if kind in (BACKEND_GS_OAUTH, BACKEND_MS_OAUTH):
        title = get_secret("sheets.title", "SubmittalCheckerData")
        if kind == BACKEND_GS_OAUTH:
            return GoogleSheetsOAuth(title)
        return MSExcelOAuth(title)
    return _get_backend_cached(kind)

@st.cache_resource
def _get_backend_cached(kind: str) -> StorageBackend:
    title = get_secret("sheets.title", "SubmittalCheckerData")
    if kind == BACKEND_GS_SERVICE:
        return GoogleSheetsSA(title)
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(data_dir, exist_ok=True)
    return SQLiteBackend(os.path.join(data_dir, "checker.db"))

# Thin wrappers

def get_backend_choice() -> StorageBackend:
    """Return the backend instance for the current selection.

    OAuth backends require the user to sign in first. If the user hasn't signed in yet,
    we keep the app usable by temporarily falling back to SQLite. Once sign-in completes,
    this will automatically switch to the selected OAuth backend.
    """
    selected_key = st.session_state.get("__backend_choice__", BACKEND_SQLITE)

    # Decide what we can actually instantiate right now.
    effective_key = selected_key
    if selected_key == BACKEND_GS_OAUTH and google_credentials() is None:
        st.sidebar.warning("Google not signed in. Click **Sign in** above, then return here.")
        effective_key = BACKEND_SQLITE
    elif selected_key == BACKEND_MS_OAUTH and not ms_access_token():
        st.sidebar.warning("Microsoft not signed in. Click **Sign in** above, then return here.")
        effective_key = BACKEND_SQLITE

    # Reuse existing instance if it matches what we'd instantiate now.
    inst = st.session_state.get("__backend_instance__")
    inst_effective = st.session_state.get("__backend_effective_kind__")
    if inst is not None and inst_effective == effective_key:
        return inst

    try:
        inst = get_backend(effective_key)
    except Exception as e:
        st.sidebar.error(f"Backend error: {e}")
        inst = get_backend(BACKEND_SQLITE)
        effective_key = BACKEND_SQLITE

    st.session_state["__backend_instance__"] = inst
    st.session_state["__backend_effective_kind__"] = effective_key
    return inst

def _backend_call(label: str, fn, default=None):
    try:
        return fn()
    except Exception as e:
        # Show in sidebar so it doesn't interrupt the main layout
        try:
            st.sidebar.error(f"Backend error ({label}): {e}")
        except Exception:
            st.error(f"Backend error ({label}): {e}")
        return default

def db_save_preset(b: StorageBackend, name: str, payload: dict):
    return _backend_call("save_preset", lambda: b.save_preset(name, payload), None)

def db_load_presets(b: StorageBackend) -> dict:
    return _backend_call("load_presets", lambda: b.load_presets(), {})

def db_delete_preset(b: StorageBackend, name: str):
    return _backend_call("delete_preset", lambda: b.delete_preset(name), None)

def db_save_submittal(b: StorageBackend, meta: dict, csv_bytes: bytes, spec_excerpt: str, sub_excerpt: str) -> int:
    return int(_backend_call("save_submittal", lambda: b.save_submittal(meta, csv_bytes, spec_excerpt, sub_excerpt), 0) or 0)

def db_list_submittals(b: StorageBackend) -> pd.DataFrame:
    return _backend_call("list_submittals", lambda: b.list_submittals(), pd.DataFrame())

def db_get_submittal(b: StorageBackend, id_: int) -> dict:
    return _backend_call("get_submittal", lambda: b.get_submittal(id_), {})

def db_delete_submittal(b: StorageBackend, id_: int):
    return _backend_call("delete_submittal", lambda: b.delete_submittal(id_), None)

def db_open_url_hint(b: StorageBackend, rec: dict) -> str | None:
    return _backend_call("open_url_hint", lambda: b.open_url_hint(rec), None)

# RFI wrappers

def db_upsert_rfi(b: StorageBackend, row: dict) -> int:
    return int(_backend_call("upsert_rfi", lambda: b.upsert_rfi(row), 0) or 0)

def db_list_rfis(b: StorageBackend, limit: int = 5000) -> pd.DataFrame:
    return _backend_call("list_rfis", lambda: b.list_rfis(limit=limit), pd.DataFrame())

def db_get_rfi(b: StorageBackend, id_: int) -> dict:
    return _backend_call("get_rfi", lambda: b.get_rfi(id_), {})

def db_delete_rfi(b: StorageBackend, id_: int) -> None:
    _backend_call("delete_rfi", lambda: b.delete_rfi(id_), None)
    return None
# -------------------------
# Tiny settings helper (piggyback on presets table)
# -------------------------
_APP_SETTINGS_KEY = "__app_settings__"


def db_add_rfi_links(backend: StorageBackend, rfi_id: int, urls: list[str]):
    fn = getattr(backend, "add_rfi_links", None)
    if not callable(fn):
        return
    fn(rfi_id, urls)

def db_list_rfi_links(backend: StorageBackend, rfi_id: int) -> list[dict]:
    fn = getattr(backend, "list_rfi_links", None)
    if not callable(fn):
        return []
    return fn(rfi_id)

def db_delete_rfi_link(backend: StorageBackend, link_id: int):
    fn = getattr(backend, "delete_rfi_link", None)
    if not callable(fn):
        return
    fn(link_id)

def db_add_rfi_attachments(backend: StorageBackend, rfi_id: int, files: list[dict]):
    fn = getattr(backend, "add_rfi_attachments", None)
    if not callable(fn):
        return
    fn(rfi_id, files)

def db_list_rfi_attachments(backend: StorageBackend, rfi_id: int) -> list[dict]:
    fn = getattr(backend, "list_rfi_attachments", None)
    if not callable(fn):
        return []
    return fn(rfi_id)

def db_get_rfi_attachment_data(backend: StorageBackend, attachment_id: int) -> tuple[str, str, bytes] | None:
    fn = getattr(backend, "get_rfi_attachment_data", None)
    if not callable(fn):
        return None
    return fn(attachment_id)

def db_delete_rfi_attachment(backend: StorageBackend, attachment_id: int):
    fn = getattr(backend, "delete_rfi_attachment", None)
    if not callable(fn):
        return
    fn(attachment_id)

def settings_load(backend: StorageBackend) -> dict:
    try:
        presets = db_load_presets(backend)
        return presets.get(_APP_SETTINGS_KEY, {})
    except Exception:
        return {}

def settings_save(backend: StorageBackend, s: dict) -> None:
    try:
        db_save_preset(backend, _APP_SETTINGS_KEY, s)
    except Exception as e:
        st.warning(f"Could not persist settings: {e}")

# -------------------------
# Email senders: SendGrid (preferred) or SMTP
# Configure one of:
#   st.secrets["sendgrid"]["api_key"]
#   st.secrets["smtp"] = {"host": "...", "port": 587, "user": "...", "password": "...", "from": "App <app@domain>"}
# -------------------------
def send_email(recipients: list[str], subject: str, html_body: str, attachments: list[tuple[str, bytes]] | None = None) -> tuple[bool,str]:
    # Try SendGrid
    sg = st.secrets.get("sendgrid")
    if sg and sg.get("api_key"):
        try:
            import base64, requests
            data = {
                "personalizations": [{"to": [{"email": r} for r in recipients]}],
                "from": {"email": sg.get("from_email", "no-reply@example.com"), "name": sg.get("from_name", "FieldFlow")},
                "subject": subject,
                "content": [{"type": "text/html", "value": html_body}],
            }
            if attachments:
                data["attachments"] = [
                    {"content": base64.b64encode(b).decode("ascii"), "type": "text/csv", "filename": fn}
                    for (fn, b) in attachments
                ]
            r = requests.post(
                "https://api.sendgrid.com/v3/mail/send",
                headers={"Authorization": f"Bearer {sg['api_key']}", "Content-Type": "application/json"},
                data=json.dumps(data),
                timeout=20,
            )
            if r.status_code in (200, 202):
                return True, "sent via SendGrid"
            return False, f"SendGrid HTTP {r.status_code}: {r.text[:200]}"
        except Exception as e:
            return False, f"SendGrid error: {e}"

    # Fallback SMTP
    smtp = st.secrets.get("smtp")
    if smtp:
        try:
            import smtplib
            from email.mime.multipart import MIMEMultipart
            from email.mime.text import MIMEText
            from email.mime.base import MIMEBase
            from email import encoders

            msg = MIMEMultipart()
            msg["From"] = smtp.get("from", smtp.get("user", "no-reply@example.com"))
            msg["To"] = ", ".join(recipients)
            msg["Subject"] = subject
            msg.attach(MIMEText(html_body, "html"))

            for (filename, bytes_) in attachments or []:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(bytes_)
                encoders.encode_base64(part)
                part.add_header("Content-Disposition", f'attachment; filename="{filename}"')
                msg.attach(part)

            server = smtplib.SMTP(smtp["host"], int(smtp.get("port", 587)))
            server.starttls()
            server.login(smtp["user"], smtp["password"])
            server.sendmail(smtp.get("from", smtp["user"]), recipients, msg.as_string())
            server.quit()
            return True, "sent via SMTP"
        except Exception as e:
            return False, f"SMTP error: {e}"

    return False, "No email provider configured (set sendgrid.api_key or smtp.* in st.secrets)"


# -------------------------
# RFI helpers (PDF + email)
# -------------------------

def parse_emails(s: str) -> list[str]:
    parts = re.split(r"[;,\s]+", (s or "").strip())
    out = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if "@" in p and "." in p:
            out.append(p)
    # de-dup, preserve order
    seen = set(); dedup = []
    for e in out:
        if e.lower() in seen:
            continue
        seen.add(e.lower()); dedup.append(e)
    return dedup


def generate_rfi_pdf(rfi: dict) -> bytes:
    """Create a simple one-page PDF for an RFI."""
    try:
        from reportlab.lib.pagesizes import LETTER
        from reportlab.pdfgen import canvas
    except Exception as e:
        raise RuntimeError("Missing reportlab. Add it to requirements.txt") from e

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=LETTER)
    w, h = LETTER

    def draw_label_value(y, label, value):
        c.setFont("Helvetica-Bold", 10)
        c.drawString(40, y, label)
        c.setFont("Helvetica", 10)
        c.drawString(150, y, (value or "—")[:120])
        return y - 14

    c.setFont("Helvetica-Bold", 16)
    c.drawString(40, h - 50, "Request for Information (RFI)")

    y = h - 80
    y = draw_label_value(y, "RFI ID:", str(rfi.get("id") or "—"))
    y = draw_label_value(y, "Project:", rfi.get("project"))
    y = draw_label_value(y, "Subject:", rfi.get("subject"))
    y = draw_label_value(y, "Discipline:", rfi.get("discipline"))
    y = draw_label_value(y, "Spec Section:", rfi.get("spec_section"))
    y = draw_label_value(y, "Priority:", rfi.get("priority"))
    y = draw_label_value(y, "Status:", rfi.get("status"))
    y = draw_label_value(y, "Due Date:", rfi.get("due_date"))
    y = draw_label_value(y, "Assignee:", rfi.get("assignee_email"))

    y -= 8
    c.setFont("Helvetica-Bold", 11)
    c.drawString(40, y, "Question")
    y -= 14
    c.setFont("Helvetica", 10)

    question = (rfi.get("question") or "").strip() or "—"
    # Simple wrapping
    max_chars = 95
    lines = []
    for para in question.splitlines():
        para = para.strip()
        if not para:
            lines.append("")
            continue
        while len(para) > max_chars:
            cut = para.rfind(" ", 0, max_chars)
            if cut < 20:
                cut = max_chars
            lines.append(para[:cut].strip())
            para = para[cut:].strip()
        lines.append(para)

    for ln in lines[:35]:
        c.drawString(50, y, ln)
        y -= 12
        if y < 80:
            break

    c.setFont("Helvetica-Oblique", 8)
    c.drawString(40, 40, "Generated by FieldFlow")
    c.showPage()
    c.save()

    return buf.getvalue()


def rfi_email_html(rfi: dict) -> str:
    esc = lambda s: (s or "").replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
    q_html = esc(rfi.get("question") or "").replace("\n", "<br/>")
    return f"""
    <h3>RFI #{esc(str(rfi.get('id') or ''))} — {esc(rfi.get('subject') or '')}</h3>
    <p><b>Project:</b> {esc(rfi.get('project') or '')}<br/>
       <b>Discipline:</b> {esc(rfi.get('discipline') or '')}<br/>
       <b>Spec Section:</b> {esc(rfi.get('spec_section') or '')}<br/>
       <b>Priority:</b> {esc(rfi.get('priority') or '')}<br/>
       <b>Due:</b> {esc(rfi.get('due_date') or '')}</p>
    <p><b>Question</b><br/>{q_html}</p>
    <p style='color:#666'>Sent from FieldFlow</p>
    """

def apply_rfi_impacts(tasks_df: pd.DataFrame, rfis_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply schedule impact days from open RFIs to a tasks table.

    Expects rfis_df to have columns: related_tasks, schedule_impact_days, status, id.
    Returns: (modified tasks_df, impacts_df)
    """
    if tasks_df is None or tasks_df.empty:
        return tasks_df, pd.DataFrame()
    if rfis_df is None or rfis_df.empty:
        return tasks_df, pd.DataFrame()

    tasks = tasks_df.copy()
    if "RFI_Impact_Days" not in tasks.columns:
        tasks["RFI_Impact_Days"] = 0

    impacts = []
    for _, r in rfis_df.iterrows():
        try:
            days = int(float(r.get("schedule_impact_days") or 0))
        except Exception:
            days = 0
        if days <= 0:
            continue
        rel = str(r.get("related_tasks") or "").strip()
        if not rel:
            continue
        task_names = [t.strip() for t in rel.split(",") if t.strip()]
        if not task_names:
            continue

        # Distribute evenly (integer days) across listed tasks.
        per = max(1, days // len(task_names))
        remainder = days - per * len(task_names)

        for i, tname in enumerate(task_names):
            add = per + (1 if i < remainder else 0)
            mask = tasks["Task"].astype(str) == tname
            if mask.any():
                tasks.loc[mask, "RFI_Impact_Days"] = tasks.loc[mask, "RFI_Impact_Days"] + add
                impacts.append({
                    "rfi_id": r.get("id"),
                    "task": tname,
                    "added_days": add,
                    "status": r.get("status"),
                    "subject": r.get("subject"),
                })

    if impacts:
        # Preserve original duration if needed
        if "Duration_Base" not in tasks.columns:
            tasks["Duration_Base"] = tasks["Duration"]
        tasks["Duration"] = (pd.to_numeric(tasks["Duration"], errors="coerce").fillna(0).astype(int) +
                              pd.to_numeric(tasks["RFI_Impact_Days"], errors="coerce").fillna(0).astype(int))

    return tasks, pd.DataFrame(impacts)

# -------------------------
# Feedback digest (pull, filter, summarize)
# -------------------------
def make_feedback_digest_df(backend: StorageBackend, period: str = "last_30_days") -> pd.DataFrame:
    # Grab a lot, then filter locally (keeps backend API simple)
    df = backend.list_feedback(limit=10000) if hasattr(backend, "list_feedback") else pd.DataFrame()
    if df.empty:
        return df
    # Normalize
    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    else:
        df["created_at"] = pd.Timestamp.utcnow()

    now = pd.Timestamp.utcnow()
    if period == "this_month":
        start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    elif period == "last_month":
        start = (now.replace(day=1) - pd.offsets.MonthBegin(1))
        start = start.replace(hour=0, minute=0, second=0, microsecond=0)
        end = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        return df[(df["created_at"] >= start) & (df["created_at"] < end)].copy()
    else:  # last_30_days
        start = now - pd.Timedelta(days=30)
    return df[df["created_at"] >= start].copy()

def summarize_feedback(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"count": 0, "avg_rating": None, "by_page": {}, "top_words": []}
    count = len(df)
    avg = float(pd.to_numeric(df.get("rating", 0), errors="coerce").fillna(0).mean())
    by_page = df.groupby("page")["rating"].mean().round(2).to_dict()

    # naive “top words” from message
    all_text = " ".join(df.get("message", "").astype(str).tolist()).lower()
    tokens = re.findall(r"[a-z]{3,}", all_text)
    stop = set("the and for with this that you your are was were have has but not into from out too very more some much they them can like just get make".split())
    freq = {}
    for t in tokens:
        if t in stop: continue
        freq[t] = freq.get(t, 0) + 1
    top_words = sorted(freq.items(), key=lambda kv: kv[1], reverse=True)[:15]
    return {"count": count, "avg_rating": avg, "by_page": by_page, "top_words": top_words}




def _current_user_label() -> str:
    if "__google_user__" in st.session_state:
        return st.session_state["__google_user__"]
    if "__ms_user__" in st.session_state:
        return st.session_state["__ms_user__"]
    return "anonymous"


# =========================
# Modular sidebar helper (used by multipage app)
# =========================

def render_sidebar(active_page: str = "") -> None:
    """Render the left sidebar (storage selection + OAuth sign-in).

    This is safe to call from Streamlit multipage apps (no st.set_page_config here).
    """
    from pathlib import Path

    # Logo (case-sensitive on Streamlit Cloud)
    logo_path = Path(__file__).parent / "assets" / "FieldFlow_logo.png"
    if logo_path.exists():
        st.sidebar.image(str(logo_path), use_container_width=True)
    else:
        st.sidebar.title("FieldFlow")

    st.sidebar.subheader("Storage")

    # Storage backend choice
    backend_choice = st.sidebar.selectbox(
        "Save presets & memory bank to",
        BACKEND_CHOICES,
        index=BACKEND_CHOICES.index(_ensure_ss("__backend_choice__", BACKEND_SQLITE)),
        key="__backend_choice__",
    )

    # OAuth callback handling (runs when provider redirects back with ?code=...&state=...)
    try:
        qp = st.query_params  # Streamlit >= 1.30
        state = qp.get("state")
    except Exception:
        try:
            qp = st.experimental_get_query_params()
            state = qp.get("state", [None])[0]
        except Exception:
            state = None

    if state:
        handled = False
        # Prefix state with g_ / m_ to avoid mixed-provider callbacks
        if isinstance(state, str) and state.startswith("g_"):
            try:
                handled = bool(google_oauth_callback())
            except Exception as e:
                st.sidebar.warning(f"Google OAuth callback failed: {e}")
        elif isinstance(state, str) and state.startswith("m_"):
            try:
                handled = bool(ms_oauth_callback())
            except Exception as e:
                st.sidebar.warning(f"Microsoft OAuth callback failed: {e}")
        else:
            # Back-compat: attempt both but don't crash
            try:
                handled = bool(google_oauth_callback()) or handled
            except Exception as e:
                st.sidebar.warning(f"Google OAuth callback failed: {e}")
            try:
                handled = bool(ms_oauth_callback()) or handled
            except Exception as e:
                st.sidebar.warning(f"Microsoft OAuth callback failed: {e}")

        if handled:
            st.rerun()

    # Status badges
    def _badge(ok: bool) -> str:
        return "🟢 signed in" if ok else "⚪ not signed in"

    g_ok = google_credentials() is not None
    m_ok = bool(ms_access_token())

    st.sidebar.write(f"Google: {_badge(g_ok)}")
    cols = st.sidebar.columns(2)
    if cols[0].button("Google Sign in", use_container_width=True, disabled=g_ok):
        google_oauth_start()
    if cols[1].button("Google Sign out", use_container_width=True, disabled=not g_ok):
        st.session_state.pop("__google_creds__", None)
        st.session_state.pop("__google_user__", None)
        st.rerun()

    st.sidebar.write(f"Microsoft: {_badge(m_ok)}")
    cols = st.sidebar.columns(2)
    if cols[0].button("Microsoft Sign in", use_container_width=True, disabled=m_ok):
        ms_oauth_start()
    if cols[1].button("Microsoft Sign out", use_container_width=True, disabled=not m_ok):
        st.session_state.pop("__ms_token__", None)
        st.session_state.pop("__ms_user__", None)
        st.session_state.pop("__ms_workbook_id__", None)
        st.rerun()

    # OneDrive workbook initialization (prevents Graph quota spam on reruns)
    if backend_choice == BACKEND_MS_OAUTH and m_ok:
        with st.sidebar.expander("Microsoft workbook", expanded=False):
            wid = st.session_state.get("__ms_workbook_id__")
            if wid:
                st.success("Workbook initialized.")
            else:
                st.info("Click once to create/find your OneDrive workbook used for storage.")
                if st.button("Initialize OneDrive workbook", use_container_width=True):
                    try:
                        b = get_backend(BACKEND_MS_OAUTH)
                        if hasattr(b, "init_workbook"):
                            b.init_workbook()
                        else:
                            # Fallback: old behavior
                            _ = b._ensure_workbook()  # type: ignore
                        st.success("Initialized workbook.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Init failed: {e}")

    st.sidebar.divider()
    if active_page:
        st.sidebar.caption(f"Page: {active_page}")


