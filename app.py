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
st.set_page_config(page_title="FieldFlow", layout="wide")

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
def google_oauth_start():
    from google_auth_oauthlib.flow import Flow
    cfg = get_secret("google_oauth", {})
    if not cfg:
        st.error("Google OAuth not configured in secrets."); return
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
            "openid","email","profile",
        ],
    )
    flow.redirect_uri = cfg["redirect_uri"]
    auth_url, state = flow.authorization_url(
        access_type="offline", include_granted_scopes="true", prompt="consent"
    )
    st.session_state["__google_state__"] = state
    st.session_state["__google_flow__"] = flow
    st.markdown(f"[Continue to Google]({auth_url})")

def google_oauth_callback():
    from google_auth_oauthlib.flow import Flow
    cfg = get_secret("google_oauth", {})
    params = st.query_params
    if "code" not in params or "state" not in params: return False
    if params.get("state") != st.session_state.get("__google_state__"): return False
    flow: Flow = st.session_state.get("__google_flow__")
    if not flow: return False
    flow.redirect_uri = cfg["redirect_uri"]
    flow.fetch_token(code=params["code"])
    creds = flow.credentials
    st.session_state["__google_token__"] = {
        "token": creds.token,
        "refresh_token": getattr(creds, "refresh_token", None),
        "token_uri": creds.token_uri,
        "client_id": creds.client_id,
        "client_secret": creds.client_secret,
        "scopes": creds.scopes,
    }
    ui = requests.get("https://www.googleapis.com/oauth2/v3/userinfo",
                      headers={"Authorization": f"Bearer {creds.token}"}).json()
    st.session_state["__google_user__"] = ui.get("email","google-user")
    st.query_params.clear()
    return True

def google_credentials():
    from google.oauth2.credentials import Credentials
    tok = st.session_state.get("__google_token__")
    if not tok: return None
    return Credentials(**tok)

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
    cfg = get_secret("microsoft_oauth", {})
    if not cfg:
        st.error("Microsoft OAuth not configured in secrets."); return
    params = {
        "client_id": cfg["client_id"],
        "response_type": "code",
        "redirect_uri": cfg["redirect_uri"],
        "response_mode": "query",
        "scope": " ".join(["openid","profile","email","offline_access","Files.ReadWrite","User.Read"]),
        "state": "ms_state_" + str(int(time.time()))
    }
    st.session_state["__ms_state__"] = params["state"]
    auth_url = "https://login.microsoftonline.com/{tenant}/oauth2/v2.0/authorize?{q}".format(
        tenant=cfg.get("tenant_id","common"), q=urllib.parse.urlencode(params)
    )
    st.markdown(f"[Continue to Microsoft]({auth_url})")

def ms_oauth_callback():
    cfg = get_secret("microsoft_oauth", {})
    params = st.query_params
    if "code" not in params or "state" not in params: return False
    if params.get("state") != st.session_state.get("__ms_state__"): return False
    token_url = f"https://login.microsoftonline.com/{cfg.get('tenant_id','common')}/oauth2/v2.0/token"
    data = {
        "client_id": cfg["client_id"],
        "client_secret": cfg["client_secret"],
        "code": params["code"],
        "redirect_uri": cfg["redirect_uri"],
        "grant_type": "authorization_code",
    }
    tok = requests.post(token_url, data=data).json()
    if "access_token" not in tok:
        return False
    tok["_obtained"] = int(time.time())
    st.session_state["__ms_token__"] = tok
    me = requests.get("https://graph.microsoft.com/v1.0/me",
                      headers={"Authorization": f"Bearer {tok['access_token']}"}).json()
    st.session_state["__ms_user__"] = me.get("userPrincipalName","microsoft-user")
    st.query_params.clear()
    return True

def ms_access_token():
    tok = st.session_state.get("__ms_token__")
    if not tok: return None
    return tok["access_token"]

class MSExcelOAuth(StorageBackend):
    def __init__(self, title: str):
        self.base = "https://graph.microsoft.com/v1.0"
        self.token = ms_access_token()
        if not self.token:
            raise RuntimeError("Microsoft not authenticated.")
        self.title = title
        self._workbook_id = self._ensure_workbook()

    def _hed(self): return {"Authorization": f"Bearer {self.token}", "Content-Type":"application/json"}

    def _ensure_workbook(self):
        r = requests.get(f"{self.base}/me/drive/root/children", headers=self._hed()).json()
        wid = None
        for item in r.get("value", []):
            if item.get("name")==f"{self.title}.xlsx":
                wid = item["id"]; break
        if not wid:
            payload = {"name": f"{self.title}.xlsx","file":{}}
            created = requests.post(f"{self.base}/me/drive/root/children", headers=self._hed(), data=json.dumps(payload)).json()
            wid = created["id"]
        return wid

    def _ensure_worksheet(self, name: str, headers: list[str]):
        url_ws = f"{self.base}/me/drive/items/{self._workbook_id}/workbook/worksheets"
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

        url = f"{self.base}/me/drive/items/{self._workbook_id}/workbook/worksheets/{urllib.parse.quote(ws_name)}/range(address='{address}')"
        requests.patch(url, headers=self._hed(), data=json.dumps({"values": values_2d}))

    def _append_rows(self, ws_name: str, values_2d: list[list]):
        url = f"{self.base}/me/drive/items/{self._workbook_id}/workbook/worksheets/{urllib.parse.quote(ws_name)}/usedRange(valuesOnly=true)"
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
        url = f"{self.base}/me/drive/items/{self._workbook_id}/workbook/worksheets/{urllib.parse.quote('feedback')}/usedRange(valuesOnly=true)"
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
        url = f"{self.base}/me/drive/items/{self._workbook_id}/workbook/worksheets/{urllib.parse.quote(ws)}/usedRange(valuesOnly=true)"
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
        url = f"{self.base}/me/drive/items/{self._workbook_id}/workbook/worksheets/{urllib.parse.quote(ws)}/usedRange(valuesOnly=true)"
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
        url = f"{self.base}/me/drive/items/{self._workbook_id}/workbook/worksheets/{urllib.parse.quote(ws)}/usedRange(valuesOnly=true)"
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
        url = f"{self.base}/me/drive/items/{self._workbook_id}/workbook/worksheets/{urllib.parse.quote(ws)}/usedRange(valuesOnly=true)"
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
        url = f"{self.base}/me/drive/items/{self._workbook_id}/workbook/worksheets/{urllib.parse.quote(ws)}/usedRange(valuesOnly=true)"
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

@st.cache_resource
def get_backend(kind: str) -> StorageBackend:
    title = get_secret("sheets.title", "SubmittalCheckerData")
    if kind == BACKEND_GS_SERVICE:
        return GoogleSheetsSA(title)
    if kind == BACKEND_GS_OAUTH:
        return GoogleSheetsOAuth(title)
    if kind == BACKEND_MS_OAUTH:
        return MSExcelOAuth(title)
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(data_dir, exist_ok=True)
    return SQLiteBackend(os.path.join(data_dir, "checker.db"))

# Thin wrappers

def get_backend_choice():
    """Convenience wrapper for pages that just want the currently selected backend."""
    backend_key = st.session_state.get("__backend_choice__", BACKEND_SQLITE)
    return get_backend(backend_key)

def db_save_preset(b: StorageBackend, name: str, payload: dict): return b.save_preset(name, payload)
def db_load_presets(b: StorageBackend) -> dict: return b.load_presets()
def db_delete_preset(b: StorageBackend, name: str): return b.delete_preset(name)
def db_save_submittal(b: StorageBackend, meta: dict, csv_bytes: bytes, spec_excerpt: str, sub_excerpt: str) -> int:
    return b.save_submittal(meta, csv_bytes, spec_excerpt, sub_excerpt)
def db_list_submittals(b: StorageBackend) -> pd.DataFrame: return b.list_submittals()
def db_get_submittal(b: StorageBackend, id_: int) -> dict: return b.get_submittal(id_)
def db_delete_submittal(b: StorageBackend, id_: int): return b.delete_submittal(id_)
def db_open_url_hint(b: StorageBackend, rec: dict) -> str | None: return b.open_url_hint(rec)


# RFI wrappers

def db_upsert_rfi(b: StorageBackend, row: dict) -> int:
    return b.upsert_rfi(row)

def db_list_rfis(b: StorageBackend, limit: int = 5000) -> pd.DataFrame:
    return b.list_rfis(limit=limit)

def db_get_rfi(b: StorageBackend, id_: int) -> dict:
    return b.get_rfi(id_)

def db_delete_rfi(b: StorageBackend, id_: int) -> None:
    return b.delete_rfi(id_)

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

from pathlib import Path
import streamlit as st

LOGO_PATH = Path(__file__).parent / "assets" / "fieldflow_logo.png"

if LOGO_PATH.exists():
    st.sidebar.image(str(LOGO_PATH), use_container_width=True)
else:
    st.sidebar.write("FieldFlow")  # fallback

st.set_page_config(
    page_title="FieldFlow",
    page_icon="🛠️",  # easiest
    layout="wide"
)

# =========================
# Sidebar: storage + auth
# =========================
st.sidebar.title("FieldFlow")
st.sidebar.subheader("Storage")

backend_choice = st.sidebar.selectbox(
    "Save presets & memory bank to",
    BACKEND_CHOICES,
    index=BACKEND_CHOICES.index(_ensure_ss("__backend_choice__", BACKEND_SQLITE)),
)
st.session_state["__backend_choice__"] = backend_choice

def _badge(ok): return "🟢 signed in" if ok else "⚪ not signed in"

if backend_choice == BACKEND_GS_OAUTH:
    st.sidebar.write("Google: " + _badge("__google_user__" in st.session_state))
    colA, colB = st.sidebar.columns(2)
    if colA.button("Sign in"):
        st.session_state["__do_google_oauth__"] = True
        st.rerun()
    if colB.button("Sign out"):
        for k in ["__google_token__","__google_user__","__google_flow__","__google_state__"]:
            st.session_state.pop(k, None)
        st.success("Signed out of Google.")
elif backend_choice == BACKEND_MS_OAUTH:
    st.sidebar.write("Microsoft: " + _badge("__ms_user__" in st.session_state))
    colA, colB = st.sidebar.columns(2)
    if colA.button("Sign in"):
        st.session_state["__do_ms_oauth__"] = True
        st.rerun()
    if colB.button("Sign out"):
        for k in ["__ms_token__","__ms_user__","__ms_state__"]:
            st.session_state.pop(k, None)
        st.success("Signed out of Microsoft.")

# Handle OAuth flows
if st.session_state.get("__do_google_oauth__"):
    st.session_state.pop("__do_google_oauth__", None)
    google_oauth_start()
elif st.session_state.get("__do_ms_oauth__"):
    st.session_state.pop("__do_ms_oauth__", None)
    ms_oauth_start()
else:
    if "code" in st.query_params and "state" in st.query_params:
        # try both; only one will succeed
        google_oauth_callback()
        ms_oauth_callback()

# Instantiate backend (fallback to SQLite on error)
try:
    backend = get_backend(backend_choice)
except Exception as e:
    st.sidebar.error(f"Backend error: {e}")
    backend = get_backend(BACKEND_SQLITE)


# =========================

# Sidebar: Pages
PAGES = ["Submittal Checker", "Schedule What-Ifs", "RFI Manager", "Aging Dashboard"]
page = st.sidebar.radio("Pages", PAGES, index=0, key="__page__")
st.sidebar.divider()

# Sidebar: About + Feedback
# =========================

def _current_user_label() -> str:
    if "__google_user__" in st.session_state:
        return st.session_state["__google_user__"]
    if "__ms_user__" in st.session_state:
        return st.session_state["__ms_user__"]
    return "anonymous"

with st.sidebar.expander("About", expanded=False):
    st.markdown(
        """
**FieldFlow** helps speed up common civil/CM workflows:

- **Submittal Checker** — hybrid scoring of submittals vs. specs (lexical + semantic + reviewer keyword coverage), with a memory bank for runs.
- **Schedule What-Ifs** — CPM with FS/SS/FF lags, optional fast-track overlap, floats breakdown, calendar mode, crash-to-target.

Built by a civil engineering student for practical use in the field.
"""
    )
    st.markdown("Made by Emiliano A. Aguiar — [LinkedIn](https://www.linkedin.com/in/emiliano-aguiar)")

with st.sidebar.expander("Leave feedback", expanded=False):
    _cur_page = st.session_state.get("__page__", "Submittal Checker")
    pages_for_fb = ["Submittal Checker", "Schedule What-Ifs", "RFI Manager", "Aging Dashboard"]
    fb_page = st.selectbox("Which page?", pages_for_fb, index=(pages_for_fb.index(_cur_page) if _cur_page in pages_for_fb else 0))
    fb_rating = st.slider("Overall rating", 1, 5, 5, help="5 = great; 1 = needs work")
    fb_cats = st.multiselect(
        "What applies?",
        ["UI/UX", "Speed", "Accuracy", "Docs/Help", "Integrations", "Other"]
    )
    fb_msg = st.text_area("Share details (what you liked, what to improve)", height=120)
    fb_email = st.text_input("Your email (optional)")
    if st.button("Send feedback"):
        if not fb_msg.strip():
            st.warning("Please add a short note so I know what to improve.")
        else:
            try:
                backend.save_feedback({
                    "created_at": datetime.utcnow().isoformat(),
                    "user_id": _current_user_label(),
                    "page": fb_page,
                    "rating": fb_rating,
                    "categories": fb_cats,
                    "email": fb_email.strip() or None,
                    "message": fb_msg.strip(),
                })
                st.success("Thanks! Your feedback was saved.")
            except Exception as e:
                st.error(f"Could not save feedback: {e}")

# (Optional) quick glance for you — last 5 feedback items
with st.sidebar.expander("Recent feedback (owner view)", expanded=False):
    try:
        _fb = backend.list_feedback(limit=5)
        if _fb.empty:
            st.caption("No feedback yet.")
        else:
            st.dataframe(_fb[["created_at","user_id","page","rating","message"]], width='stretch', hide_index=True)
    except Exception as e:
        st.caption(f"(Feedback listing unavailable: {e})")

# =========================
# Sidebar: Notifications & Digest
# =========================
with st.sidebar.expander("Notifications & Monthly Digest", expanded=False):
    s = settings_load(backend)
    default_emails = s.get("notify_emails", "")
    notify_emails = st.text_input("Notification recipients (comma-separated emails)", value=default_emails)
    default_period = s.get("digest_period", "last_30_days")
    digest_period = st.selectbox("Digest period", ["last_30_days", "this_month", "last_month"], index=["last_30_days","this_month","last_month"].index(default_period))
    st.caption("The digest pulls recent feedback from the selected storage backend.")

    c1, c2 = st.columns(2)
    if c1.button("Save notification settings"):
        s["notify_emails"] = notify_emails.strip()
        s["digest_period"] = digest_period
        settings_save(backend, s)
        st.success("Notification settings saved.")

    if c2.button("Send test email"):
        recips = [e.strip() for e in (notify_emails or "").split(",") if e.strip()]
        if not recips:
            st.warning("Add at least one recipient first.")
        else:
            ok, msg = send_email(recips, "FieldFlow — test email", "<p>This is a test from your app.</p>")
            st.success(msg) if ok else st.error(msg)

    st.markdown("---")
    if st.button("Generate digest now"):
        df = make_feedback_digest_df(backend, period=digest_period)
        summary = summarize_feedback(df)
        if df.empty:
            st.info("No feedback in the selected period.")
        else:
            # CSV for download
            csv_bytes = df.to_csv(index=False).encode()
            st.download_button("Download digest CSV", csv_bytes, file_name=f"feedback_digest_{digest_period}.csv", mime="text/csv")

            # Push to workspace tabs if available
            pushed = False
            try:
                if hasattr(backend, "_ensure_ws"):  # Google Sheets backends
                    title = f"digest_{pd.Timestamp.utcnow():%Y_%m_%d_%H%M}"
                    wsname = "feedback_" + title
                    backend._ensure_ws(wsname, list(df.columns))
                    # Write header + body
                    values = [list(df.columns)] + df.astype(object).where(pd.notna(df), "").values.tolist()
                    backend.ss.worksheet(wsname).update("A1", values)
                    st.success(f"Pushed digest to Google Sheet tab '{wsname}'.")
                    pushed = True
                elif isinstance(backend, MSExcelOAuth):
                    wsname = f"feedback_{pd.Timestamp.utcnow():%Y_%m_%d_%H%M}"
                    backend._ensure_worksheet(wsname, list(df.columns))
                    values = [list(df.columns)] + df.astype(object).where(pd.notna(df), "").values.tolist()
                    backend._update_range(wsname, "A1", values)
                    st.success(f"Pushed digest to Excel workbook tab '{wsname}'.")
                    pushed = True
            except Exception as e:
                st.warning(f"Could not push to workspace: {e}")

            # Email digest
            recips = [e.strip() for e in (notify_emails or "").split(",") if e.strip()]
            if recips:
                html = f"""
                <h3>FieldFlow — Feedback Digest ({digest_period.replace('_',' ')})</h3>
                <p><b>Count:</b> {summary['count']} &nbsp; <b>Avg rating:</b> {summary['avg_rating'] if summary['avg_rating'] is not None else '—'}</p>
                <p><b>By page:</b> {summary['by_page']}</p>
                <p><b>Top words:</b> {', '.join([w for w,_ in summary['top_words']])}</p>
                <p>This message was generated by your app. Attachments: CSV digest.</p>
                """
                ok, msg = send_email(recips, "FieldFlow — feedback digest", html_body=html, attachments=[(f"feedback_digest_{digest_period}.csv", csv_bytes)])
                st.success(f"Emailed digest: {msg}") if ok else st.error(msg)
            else:
                st.info("Add recipients to email the digest.")

# =========================
# Submittal Checker (page)
# =========================
def submittal_checker_page():
    st.header("Submittal Checker")

    # Readers
    try:
        from rapidfuzz import fuzz
    except Exception:
        st.error("Missing dependency `rapidfuzz`. pip install rapidfuzz")
        return
    try:
        from pypdf import PdfReader
    except Exception:
        PdfReader = None
    try:
        import docx
    except Exception:
        docx = None

    def _read_pdf(file: io.BytesIO) -> str:
        if not PdfReader: return ""
        reader = PdfReader(file)
        parts = []
        for p in reader.pages:
            try: parts.append(p.extract_text() or "")
            except Exception: pass
        return "\n".join(parts)

    def _read_docx(file: io.BytesIO) -> str:
        if not docx: return ""
        d = docx.Document(file)
        return "\n".join(p.text for p in d.paragraphs)

    def _read_txt(file: io.BytesIO) -> str:
        return file.read().decode(errors="ignore")

    def _read_csv(file: io.BytesIO) -> str:
        df = pd.read_csv(file)
        strings = []
        for col in df.columns:
            try: strings.extend(df[col].astype(str).tolist())
            except: pass
        return "\n".join(strings)

    def read_any(uploaded) -> str:
        name = (getattr(uploaded, "name","") or "").lower()
        raw_bytes = uploaded.read()
        data = io.BytesIO(raw_bytes)

        # primary text extraction
        if name.endswith(".pdf"):
            base = _read_pdf(data)
            # OCR fallback
            if use_ocr and (not base or len(base.strip()) < 40):
                base = ocr_pdf_to_text(io.BytesIO(raw_bytes)) or base
            # tables
            tabtxt = tabula_tables_to_text(io.BytesIO(raw_bytes)) if use_table else ""
            return (base + "\n\n" + tabtxt).strip()
        if name.endswith(".docx"): return _read_docx(data)
        if name.endswith(".txt"):  return _read_txt(data)
        if name.endswith(".csv"):  return _read_csv(data)
        raise ValueError(f"Unsupported file: {name}")

    BULLET_RE = re.compile(r"^\s*(?:[-•*]|\d+[.)]|[A-Z]\.|[A-Z]\))\s+")
    HEADER_RE = re.compile(r"^(?:PART|SECTION|DIVISION|SUBMITTALS?|WARRANTY|SHOP DRAWINGS)\b", re.I)
    def clean_text(t: str) -> str:
        t = re.sub(r"\r","\n", t)
        t = re.sub(r"\n{2,}","\n", t)
        return t.strip()
    def split_into_chunks(t: str) -> List[str]:
        lines = [ln.strip() for ln in t.split("\n")]
        chunks, acc = [], []
        def flush():
            if acc:
                s = " ".join(acc).strip()
                if s: chunks.append(s)
                acc.clear()
        for ln in lines:
            if not ln:
                flush(); continue
            if BULLET_RE.search(ln) or HEADER_RE.search(ln):
                flush(); chunks.append(ln); continue
            acc.append(ln)
        flush()
        seen, out = set(), []
        for c in chunks:
            k = c.lower()
            if k not in seen:
                out.append(c); seen.add(k)
        return out

    # Presets + defaults
    DEFAULTS = {
        "must": "data sheet,warranty,certificate",
        "nice": "shop drawing,test report,O&M manual",
        "forbid": "by others,not provided,N/A",
        "weights": {"alpha":0.45,"beta":0.35,"gamma":0.20,"delta":0.20,"epsilon":0.05},
        "threshold": 85,
    }
    def _apply_preset(p):
        st.session_state["kw_must"] = p.get("must", DEFAULTS["must"])
        st.session_state["kw_nice"] = p.get("nice", DEFAULTS["nice"])
        st.session_state["kw_forbid"] = p.get("forbid", DEFAULTS["forbid"])
        w = p.get("weights", DEFAULTS["weights"])
        st.session_state["w_alpha"] = float(w.get("alpha", DEFAULTS["weights"]["alpha"]))
        st.session_state["w_beta"] = float(w.get("beta", DEFAULTS["weights"]["beta"]))
        st.session_state["w_gamma"] = float(w.get("gamma", DEFAULTS["weights"]["gamma"]))
        st.session_state["w_delta"] = float(w.get("delta", DEFAULTS["weights"]["delta"]))
        st.session_state["w_epsilon"] = float(w.get("epsilon", DEFAULTS["weights"]["epsilon"]))
        st.session_state["hybrid_threshold"] = int(p.get("threshold", DEFAULTS["threshold"]))
        st.rerun()
    if "kw_must" not in st.session_state:
        _apply_preset(DEFAULTS)

    # Sources
    cL, cR = st.columns(2)
    with cL:
        st.subheader("Spec Source")
        spec_file = st.file_uploader("Upload spec (PDF/DOCX/TXT/CSV)", type=["pdf","docx","txt","csv"], key="spec_file")
        spec_text_area = st.text_area("Or paste spec text", height=220, placeholder="Paste specification clauses…", key="spec_text")
    with cR:
        st.subheader("Submittal Source")
        sub_file = st.file_uploader("Upload submittal (PDF/DOCX/TXT/CSV)", type=["pdf","docx","txt","csv"], key="sub_file")
        sub_text_area = st.text_area("Or paste submittal text", height=220, placeholder="Paste submittal content…", key="sub_text")

    st.markdown("---")
    with st.expander("Reviewer policy, keywords & weights", expanded=False):
        # Presets
        row1 = st.columns([0.35,0.2,0.2,0.25])
        presets = db_load_presets(backend)
        with row1[0]:
            sel = st.selectbox("Load preset", ["—"] + sorted(presets.keys()))
        with row1[1]:
            if st.button("Load"):
                if sel != "—":
                    _apply_preset(presets[sel])
        with row1[2]:
            if st.button("Reset"):
                _apply_preset(DEFAULTS)
        with row1[3]:
            if sel != "—" and st.button("Delete preset"):
                db_delete_preset(backend, sel); st.success(f"Deleted {sel}"); st.rerun()

        cA, cB = st.columns(2)
        with cA:
            must_text = st.text_area("Must-have terms (comma-separated)",
                                     st.session_state["kw_must"], key="kw_must",
                                     help="All must appear in the best-matching submittal excerpt; otherwise PASS is blocked.")
            nice_text = st.text_area("Nice-to-have terms",
                                     st.session_state["kw_nice"], key="kw_nice",
                                     help="Improves the coverage portion of the hybrid score.")
        with cB:
            forbid_text = st.text_area("Forbidden phrases",
                                       st.session_state["kw_forbid"], key="kw_forbid",
                                       help="Any hit blocks PASS and applies a penalty.")
            st.caption("Forbidden always blocks PASS regardless of score.")

        st.markdown("**Scoring weights**")
        α = st.slider("Lexical weight", 0.0, 1.0, st.session_state["w_alpha"], 0.05, key="w_alpha",
                      help="RapidFuzz token-set similarity of spec vs. submittal text (literal match strength).")
        β = st.slider("Semantic weight", 0.0, 1.0, st.session_state["w_beta"], 0.05, key="w_beta",
                      help="Embedding cosine similarity (meaning match). Uses sentence-transformers if available.")
        γ = st.slider("Coverage weight", 0.0, 1.0, st.session_state["w_gamma"], 0.05, key="w_gamma",
                      help="Coverage: 70% must + 30% nice. Encourages presence of reviewer keywords.")
        δ = st.slider("Forbidden penalty per hit", 0.0, 1.0, st.session_state["w_delta"], 0.05, key="w_delta",
                      help="Subtract this per forbidden phrase hit in the best match.")
        ε = st.slider("Section boost (match)", 0.0, 0.5, st.session_state["w_epsilon"], 0.01, key="w_epsilon",
                      help="Small bonus when headings align (e.g., Warranty vs. Warranty).")

        row2 = st.columns([0.8,0.2])
        with row2[0]:
            pname = st.text_input("Preset name", placeholder="e.g., Div03_Concrete_ReviewerA")
        with row2[1]:
            if st.button("Save preset"):
                payload = {
                    "must": st.session_state["kw_must"],
                    "nice": st.session_state["kw_nice"],
                    "forbid": st.session_state["kw_forbid"],
                    "weights": {"alpha":α, "beta":β, "gamma":γ, "delta":δ, "epsilon":ε},
                    "threshold": st.session_state.get("hybrid_threshold", 85),
                }
                if not pname.strip():
                    st.warning("Enter a preset name.")
                else:
                    db_save_preset(backend, pname.strip(), payload)
                    st.success(f"Saved preset: {pname.strip()}"); st.rerun()

    threshold = st.slider("Hybrid PASS threshold", 0, 100,
                          st.session_state.get("hybrid_threshold", 85),
                          help="Final decision threshold on the 0–100 hybrid score.")
    st.markdown("---")
    t1, t2, t3, t4 = st.columns(4)
    use_ocr   = t1.checkbox("OCR if PDF text is empty", value=True,
                            help="Try Tesseract when PDFs are scans.")
    use_table = t2.checkbox("Extract tables", value=True,
                            help="Use tabula to read embedded tables and index them.")
    use_bm25  = t3.checkbox("BM25 boost", value=True,
                            help="Lexical retriever for long clauses.")
    use_sem   = t4.checkbox("Semantic embeddings", value=True,
                            help="Use sentence-transformer embeddings if available")

    run_btn = st.button("Analyze", type="primary")

    if run_btn:
        try:
            spec_text = ""
            if spec_file: spec_text = read_any(spec_file)
            if spec_text_area: spec_text = (spec_text + "\n" + spec_text_area).strip()

            sub_text = ""
            if sub_file: sub_text = read_any(sub_file)
            if sub_text_area: sub_text = (sub_text + "\n" + sub_text_area).strip()

            if not spec_text: st.warning("Provide spec text or upload a spec file."); st.stop()
            if not sub_text: st.warning("Provide submittal text or upload a submittal file."); st.stop()

            spec_text = clean_text(spec_text); sub_text = clean_text(sub_text)
            spec_chunks = split_into_chunks(spec_text)
            sub_chunks = split_into_chunks(sub_text)

            # lexical + semantic(optional)
            try:
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer("all-MiniLM-L6-v2")
                emb_spec = model.encode(spec_chunks, show_progress_bar=False, normalize_embeddings=True)
                emb_sub  = model.encode(sub_chunks, show_progress_bar=False, normalize_embeddings=True)
                use_sem = True
            except Exception:
                use_sem = False

            def section_of(s: str) -> str:
                m = re.search(r"(WARRANTY|SUBMITTALS?|SHOP DRAWINGS|PRODUCT DATA|TESTS?)", s, re.I)
                return (m.group(1).upper() if m else "")

            must = [w.strip() for w in st.session_state["kw_must"].split(",") if w.strip()]
            nice = [w.strip() for w in st.session_state["kw_nice"].split(",") if w.strip()]
            forbid = [w.strip() for w in st.session_state["kw_forbid"].split(",") if w.strip()]

            rows = []
            for i, s in enumerate(spec_chunks):
                s_norm = s.lower()
                sec_s = section_of(s)
                best_rf = -1.0; best_sem = 0.0; best_sec = 0.0
                best_chunk = ""
                for j, t in enumerate(sub_chunks):
                    t_norm = t.lower()
                    rf = float(fuzz.token_set_ratio(s_norm, t_norm)) / 100.0
                    sem = float(np.dot(emb_spec[i], emb_sub[j])) if use_sem else 0.0
                    sec_bonus = ε if (sec_s and sec_s == section_of(t)) else 0.0
                    score = α*rf + β*sem + sec_bonus
                    ref = α*best_rf + β*best_sem + best_sec
                    if score > ref:
                        best_rf, best_sem, best_sec, best_chunk = rf, sem, sec_bonus, t

                # coverage / forbid
                t_use = best_chunk.lower()
                must_hits = sum(1 for w in must if w.lower() in t_use)
                nice_hits = sum(1 for w in nice if w.lower() in t_use)
                forb_hits = sum(1 for w in forbid if w.lower() in t_use)
                must_cov = (must_hits / max(1, len(must))) if must else 0.0
                nice_cov = (nice_hits / max(1, len(nice))) if nice else 0.0
                coverage = 0.7*must_cov + 0.3*nice_cov

                hybrid = α*best_rf + β*(best_sem if use_sem else 0.0) + γ*coverage + best_sec - δ*forb_hits
                hybrid = max(0.0, min(1.0, hybrid))
                exact_like = (s_norm in t_use) or (best_rf >= 0.97)
                pass_logic = (exact_like or (hybrid*100 >= threshold)) and (must_hits == len(must)) and (forb_hits == 0)
                decision = "PASS" if pass_logic else "REVIEW"

                def hl(text: str, words: List[str], color: str) -> str:
                    out = text
                    for w in sorted(words, key=len, reverse=True):
                        if not w: continue
                        out = re.sub(rf"({re.escape(w)})", rf"<mark style='background:{color};padding:0 2px'>\1</mark>", out, flags=re.I)
                    return out

                best_chunk_html = hl(best_chunk, must, "#d1ffd1")
                best_chunk_html = hl(best_chunk_html, nice, "#d1e8ff")
                best_chunk_html = hl(best_chunk_html, forbid, "#ffd6d6")

                rows.append({
                    "Concept": " ".join(re.findall(r"[A-Za-z0-9-/]+", s)[:10]),
                    "Spec Item": s,
                    "Best Match": best_chunk,
                    "Hybrid_Score": round(hybrid, 3),
                    "Lexical": round(best_rf, 3),
                    "Semantic": round(best_sem if use_sem else 0.0, 3),
                    "Coverage": round(coverage, 3),
                    "Must_hits": must_hits,
                    "Nice_hits": nice_hits,
                    "Forbidden_hits": forb_hits,
                    "Section_bonus": round(best_sec, 3),
                    "Exact_like": exact_like,
                    "Decision": decision,
                    "_best_chunk_html": best_chunk_html,
                })

            df = pd.DataFrame(rows)
            pass_rate = (df["Decision"]=="PASS").mean() if not df.empty else 0.0

            m1,m2,m3 = st.columns(3)
            m1.metric("Coverage (PASS)", f"{pass_rate*100:.0f}%")
            m2.metric("Threshold", threshold)
            m3.metric("Items", len(df))

            st.markdown("### Results")
            df_show = df.drop(columns=["_best_chunk_html"])
            df_fullwidth(df_show, hide_index=True, height=rows_to_height(len(df_show)+5))

            st.markdown("### Best-match highlights")
            for _, r in df.iterrows():
                with st.expander(r["Concept"][:80] or r["Spec Item"][:80]):
                    st.write("**Spec:** ", r["Spec Item"])
                    st.write("**Decision:** ", r["Decision"], " — Hybrid: ", r["Hybrid_Score"])
                    st.markdown(r["_best_chunk_html"], unsafe_allow_html=True)

            csv_bytes = df_show.to_csv(index=False).encode()
            st.download_button("Download results CSV", csv_bytes, "submittal_checker_results.csv", "text/csv")

            # Memory Bank save
            st.markdown("---")
            st.subheader("Save to Memory Bank")
            pass_count = int((df["Decision"]=="PASS").sum())
            review_count = int((df["Decision"]=="REVIEW").sum())
            meta_weights = {"alpha":α,"beta":β,"gamma":γ,"delta":δ,"epsilon":ε}

            with st.form("save_run"):
                c1,c2,c3 = st.columns(3)
                company = c1.text_input("Company", "")
                client  = c2.text_input("Client", "")
                project = c3.text_input("Project", "")
                c4,c5,c6 = st.columns(3)
                date_submitted = c4.date_input("Date submitted", value=date.today())
                quote = c5.number_input("Quote (if any)", min_value=0.0, value=0.0, step=1000.0)
                notes = c6.text_input("Notes", "")
                submit = st.form_submit_button("Save run")
                if submit:
                    if not company or not project:
                        st.warning("Company and Project required.")
                    else:
                        run_id = db_save_submittal(
                            backend,
                            {
                                "company":company,"client":client,"project":project,
                                "date_submitted":str(date_submitted),"quote":float(quote),"notes":notes,
                                "threshold":int(threshold),"weights":meta_weights,
                                "must":",".join(must),"nice":",".join(nice),"forbid":",".join(forbid),
                                "pass_count":pass_count,"review_count":review_count,"pass_rate":float(pass_rate),
                            },
                            csv_bytes,
                            spec_text[:2000],
                            sub_text[:2000],
                        )
                        st.success(f"Saved run #{run_id} ({company} – {project})")

            st.subheader("Memory Bank")
            bank = db_list_submittals(backend)
            if bank.empty:
                st.info("No saved runs yet.")
            else:
                fc1,fc2,fc3,fc4 = st.columns(4)
                f_company = fc1.selectbox("Company", ["All"] + sorted(bank["company"].dropna().unique().tolist()))
                f_project = fc2.selectbox("Project", ["All"] + sorted(bank["project"].dropna().unique().tolist()))
                f_client  = fc3.selectbox("Client",  ["All"] + sorted(bank["client"].dropna().unique().tolist()))
                sort_by   = fc4.selectbox("Sort by", ["created_at (new→old)","pass_rate (high→low)","date_submitted (new→old)"])
                view = bank.copy()
                if f_company!="All": view = view[view["company"]==f_company]
                if f_project!="All": view = view[view["project"]==f_project]
                if f_client!="All":  view = view[view]["client"]==f_client
                if sort_by.startswith("pass_rate"): view = view.sort_values("pass_rate", ascending=False)
                elif sort_by.startswith("date_submitted"): view = view.sort_values("date_submitted", ascending=False)
                else: view = view.sort_values("created_at", ascending=False)
                df_fullwidth(view.assign(PassPct=(view["pass_rate"]*100).round(1)).rename(columns={"PassPct":"Pass %"}),
                             hide_index=True, height=rows_to_height(len(view)+5))
                st.markdown("##### Inspect / Download a saved run")
                sid = st.number_input("Run ID", min_value=0, step=1)
                a,b,c = st.columns(3)
                if a.button("View details"):
                    rec = db_get_submittal(backend, int(sid))
                    if not rec: st.warning("No such run.")
                    else:
                        st.json({k: rec.get(k) for k in ["id","company","client","project","date_submitted","quote","pass_count","review_count","pass_rate","threshold","created_at"]})
                        st.code(rec.get("spec_excerpt") or "—", language="text")
                        st.code(rec.get("submittal_excerpt") or "—", language="text")
                        if "result_csv" in rec:  # SQLite path
                            st.download_button("Download results CSV", rec["result_csv"], f"submittal_run_{rec['id']}.csv", "text/csv")
                        else:
                            st.caption("For Google/Microsoft backends, open the workspace to fetch results.")
                        url = db_open_url_hint(backend, rec)
                        if url: st.markdown(f"[Open in workspace]({url})")
                if b.button("Delete"):
                    db_delete_submittal(backend, int(sid)); st.success(f"Deleted run #{int(sid)}"); st.rerun()

        except Exception as e:
            st.exception(e)

# =========================
# Schedule What-Ifs (page)
# =========================
def schedule_whatifs_page():
    st.header("Schedule What-Ifs — Floats + Calendar")

    REQUIRED = ["Task","Duration","Predecessors","Normal_Cost_per_day","Crash_Cost_per_day"]
    ALIASES: Dict[str,str] = {
        "task":"Task","activity":"Task","name":"Task",
        "duration":"Duration","duration_days":"Duration","duration_(days)":"Duration","dur":"Duration",
        "predecessors":"Predecessors","pred":"Predecessors","predecessor":"Predecessors",
        "normal_cost_per_day":"Normal_Cost_per_day","normal/day":"Normal_Cost_per_day","normal_cost/day":"Normal_Cost_per_day",
        "normal_cost":"Normal_Cost_per_day","normal_cost_usd":"Normal_Cost_per_day",
        "crash_cost_per_day":"Crash_Cost_per_day","crash/day":"Crash_Cost_per_day","crash_cost/day":"Crash_Cost_per_day",
        "crash_cost":"Crash_Cost_per_day","crash_cost_usd":"Crash_Cost_per_day",
        "min_duration":"Min_Duration","min_dur":"Min_Duration","min":"Min_Duration",
        "crash_duration":"Min_Duration","crash_duration_days":"Min_Duration","crash_dur":"Min_Duration",
        "overlap_ok":"Overlap_OK","overlap?":"Overlap_OK","allow_overlap":"Overlap_OK",
    }

    def norm_col(s: str) -> str:
        s = str(s).strip().lower()
        s = re.sub(r"[^a-z0-9]+","_", s)
        return re.sub(r"_+","_", s).strip("_")

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
            df["Min_Duration"] = df["Duration"]; warnings.append("Min_Duration not provided; defaulting to Duration.")
        if "Overlap_OK" not in df.columns:
            df["Overlap_OK"] = False; warnings.append("Overlap_OK not provided; defaulting to False.")
        for col in ["Duration","Min_Duration","Normal_Cost_per_day","Crash_Cost_per_day"]:
            if col in df.columns: df[col] = pd.to_numeric(df[col], errors="coerce")
        if "Overlap_OK" in df.columns:
            df["Overlap_OK"] = df["Overlap_OK"].astype(str).str.strip().str.lower().map(
                {"1":True,"0":False,"true":True,"false":False,"yes":True,"no":False,"y":True,"n":False}
            ).fillna(False)
        if "Task" in df.columns: df["Task"] = df["Task"].astype(str)
        if "Predecessors" in df.columns: df["Predecessors"] = df["Predecessors"].fillna("").astype(str)
        if {"Duration","Min_Duration"} <= set(df.columns):
            mask = df["Min_Duration"] > df["Duration"]
            if mask.any():
                warnings.append("Some rows have Min_Duration > Duration; clamping to Duration.")
                df.loc[mask,"Min_Duration"] = df.loc[mask,"Duration"]
        return df, warnings

    # CPM with lags: FS/SS/FF + integer lag (±)
    @dataclass
    class CPMNode:
        task: str
        dur: int
        preds: List[Tuple[str,str,int]]  # (pred, rel, lag)
        es: int=0; ef: int=0; ls: int=0; lf: int=0; tf: int=0

    def parse_predecessors(s: str) -> List[Tuple[str,str,int]]:
        """
        Accepts: "A", "A FS+0", "B SS+2; C FF-1"
        Returns list of (pred_task, REL, lag_days)
        """
        if not s: return []
        out = []
        parts = [p.strip() for p in re.split(r"[;,]", s) if p.strip()]
        for p in parts:
            m = re.match(r"^(.+?)\s*(FS|SS|FF)?\s*([+\-]\d+)?$", p, re.I)
            if m:
                pred = m.group(1).strip()
                rel  = (m.group(2) or "FS").upper()
                lag  = int(m.group(3)) if m.group(3) else 0
                out.append((pred, rel, lag))
            else:
                out.append((p, "FS", 0))
        return out

    def topological_order(nodes: Dict[str, CPMNode]) -> List[str]:
        indeg = {k: 0 for k in nodes}
        for n in nodes.values():
            for p,_,_ in n.preds:
                if p in indeg:
                    indeg[n.task] += 1
        Q = [k for k,d in indeg.items() if d==0]
        order = []
        while Q:
            v = Q.pop(0); order.append(v)
            for w in nodes:
                if any(v==pp for pp,_,_ in nodes[w].preds):
                    indeg[w]-=1
                    if indeg[w]==0: Q.append(w)
        if len(order)!=len(nodes):
            raise ValueError("Dependency cycle detected. Check Predecessors.")
        return order

    def cpm_schedule(df: pd.DataFrame, overlap_frac: float=0.0, clamp_free_float: bool=True) -> Tuple[pd.DataFrame, int]:
        nodes: Dict[str, CPMNode] = {}
        for _, r in df.iterrows():
            preds = parse_predecessors(r.get("Predecessors",""))
            nodes[r["Task"]] = CPMNode(task=r["Task"], dur=int(max(0, r["Duration"])), preds=preds)

        order = topological_order(nodes)

        # Forward pass
        for name in order:
            node = nodes[name]
            if not node.preds:
                node.es = 0
            else:
                starts = []
                for (p, rel, lag) in node.preds:
                    if p not in nodes: raise ValueError(f"Unknown predecessor '{p}' for task '{name}'.")
                    pred = nodes[p]
                    if rel == "FS":
                        base = pred.ef
                        allow = bool(df.loc[df["Task"]==name, "Overlap_OK"].iloc[0]) if "Overlap_OK" in df.columns else False
                        if allow and overlap_frac>0:
                            base = max(0, pred.ef - int(round(overlap_frac*pred.dur)))
                        starts.append(base + lag)
                    elif rel == "SS":
                        starts.append(pred.es + lag)
                    elif rel == "FF":
                        starts.append((pred.ef + lag) - node.dur)
                    else:
                        starts.append(pred.ef + lag)
                node.es = max(starts)
            node.ef = node.es + node.dur

        project_duration = max((n.ef for n in nodes.values()), default=0)

        # Successors
        succs: Dict[str,List[str]] = {k:[] for k in nodes}
        for t, n in nodes.items():
            for (p,_,_) in n.preds:
                if p in succs: succs[p].append(t)

        # Backward pass
        for name in reversed(order):
            node = nodes[name]
            succ_ls = [nodes[s].ls for s in succs[name]]
            node.lf = min(succ_ls) if succ_ls else project_duration
            node.ls = node.lf - node.dur
            node.tf = node.ls - node.es

        rows = []
        for name in order:
            n = nodes[name]
            succ_es_min = min([nodes[s].es for s in succs[name]], default=project_duration)
            free_raw = succ_es_min - n.ef
            free_float = max(0, free_raw) if clamp_free_float else free_raw
            max_pred_ef = 0
            if n.preds:
                max_pred_ef = max([
                    nodes[p].ef if rel!="SS" else nodes[p].es
                    for (p,rel,_) in n.preds if p in nodes
                ], default=0)
            indep = max(0, succ_es_min - max_pred_ef - n.dur)
            interfering = n.tf - free_float
            rows.append({
                "Task": n.task, "Duration": n.dur,
                "ES": n.es, "EF": n.ef, "LS": n.ls, "LF": n.lf,
                "Total_Float": n.tf, "Slack": n.tf,
                "Free_Float": free_float, "Free_Float_Raw": free_raw,
                "Independent_Float": indep, "Interfering_Float": interfering,
                "Critical": n.tf==0,
            })
        out = pd.DataFrame(rows).sort_values("ES", kind="stable")
        return out, project_duration

    # Calendar helpers
    from pandas.tseries.offsets import CustomBusinessDay
    def make_cbd(workdays: List[str], holidays: List[str]) -> CustomBusinessDay:
        weekmask = " ".join(workdays)
        hols = [pd.to_datetime(h).date() for h in holidays if h.strip()]
        return CustomBusinessDay(weekmask=weekmask, holidays=hols)
    def calendarize(schedule_df: pd.DataFrame, project_start: date, cbd: CustomBusinessDay) -> pd.DataFrame:
        if schedule_df is None or schedule_df.empty: return schedule_df
        out = schedule_df.copy()
        start_ts = pd.to_datetime(project_start)
        out["ES_date"] = start_ts + out["ES"].astype(int) * cbd
        out["EF_date_excl"] = start_ts + out["EF"].astype(int) * cbd
        out["Start_Date"] = out["ES_date"]
        out["Finish_Date"] = out["EF_date_excl"] - 1 * cbd
        return out
    def gantt_chart(schedule_df: pd.DataFrame) -> alt.Chart:
        if schedule_df is None or schedule_df.empty:
            return alt.Chart(pd.DataFrame({"ES":[0],"EF":[0],"Task":["No tasks"]})).mark_bar()
        data = schedule_df.copy()
        data["Task"] = data["Task"].astype(str)
        height = max(160, min(40*len(data), 1200))
        if "ES_date" in data.columns and "EF_date_excl" in data.columns:
            x = alt.X("ES_date:T", title="Start"); x2 = "EF_date_excl:T"
        else:
            x = alt.X("ES:Q", title="Day (project time)"); x2 = "EF:Q"
        return (alt.Chart(data).mark_bar().encode(
            x=x, x2=x2,
            y=alt.Y("Task:N", sort=alt.SortField("ES", order="ascending")),
            color=alt.condition("datum.Critical", alt.value("#d62728"), alt.value("#1f77b4")),
            tooltip=["Task","Duration","ES","EF","LS","LF","Total_Float","Free_Float","Independent_Float","Interfering_Float",
                     alt.Tooltip("Start_Date:T", title="Start Date", format="%Y-%m-%d"),
                     alt.Tooltip("Finish_Date:T", title="Finish Date", format="%Y-%m-%d")],
        ).properties(height=height))

    # UI
    st.subheader("Task Table")
    uploaded = st.file_uploader("Upload tasks CSV", type=["csv"], accept_multiple_files=False)
    with st.expander("CSV columns & example", expanded=False):
        st.code(
            "Task,Duration,Predecessors,Normal_Cost_per_day,Crash_Cost_per_day,Min_Duration,Overlap_OK\n"
            "A - Site Prep,5,,1200,1800,3,TRUE\n"
            "B - Foundations,10,\"A - Site Prep FS+0\",1600,2600,7,TRUE\n"
            "C - Structure,12,\"B - Foundations SS+2\",1900,3000,9,TRUE\n"
            "D - Enclosure,9,\"C - Structure FF+0\",1400,2300,7,FALSE\n",
            language="csv"
        )
        st.caption("Predecessors accept lags like FS+2, SS-1, FF+0. Separate multiple predecessors by comma/semicolon.")

    base_df = pd.DataFrame({
        "Task":["A - Site Prep","B - Foundations","C - Structure","D - MEP Rough-In","E - Enclosure","F - Finishes"],
        "Duration":[5,10,12,8,9,10],
        "Predecessors":["","A - Site Prep FS+0","B - Foundations SS+2","C - Structure","C - Structure FF+0","D - MEP Rough-In, E - Enclosure"],
        "Normal_Cost_per_day":[1200,1600,1900,1500,1400,1550],
        "Crash_Cost_per_day":[1800,2600,3000,2400,2300,2550],
        "Min_Duration":[3,7,9,6,7,8],
        "Overlap_OK":[True, True, True, True, False, False],
    })
    if uploaded:
        df, warns = load_csv(uploaded)
    else:
        df, warns = base_df.copy(), ["Using example table — upload your CSV to replace it."]

    for w in warns: st.warning(w)
    editor_height = rows_to_height(len(df)+5)
    edited_df = editor_fullwidth(
        df, hide_index=True, num_rows="dynamic",
        column_config={
            "Task": st.column_config.TextColumn("Task", required=True),
            "Duration": st.column_config.NumberColumn("Duration", min_value=0, step=1),
            "Predecessors": st.column_config.TextColumn(
                "Predecessors",
                help="Use FS/SS/FF with optional ±lag. e.g., A FS+0; B SS+2; C FF-1"
            ),
            "Normal_Cost_per_day": st.column_config.NumberColumn("Normal Cost / day", min_value=0),
            "Crash_Cost_per_day": st.column_config.NumberColumn("Crash Cost / day", min_value=0),
            "Min_Duration": st.column_config.NumberColumn("Min Duration", min_value=0, step=1),
            "Overlap_OK": st.column_config.CheckboxColumn(
                "Overlap OK", help="If enabled, FS successors may start earlier by a fraction of predecessor duration (see overlap control)."
            ),
        },
        key="task_table", height=editor_height
    )
    # Coercions
    edited_df["Task"] = edited_df["Task"].astype(str)
    edited_df["Predecessors"] = edited_df["Predecessors"].fillna("").astype(str)
    for c in ["Duration","Min_Duration"]: edited_df[c] = pd.to_numeric(edited_df[c], errors="coerce").fillna(0).astype(int).clip(lower=0)
    for c in ["Normal_Cost_per_day","Crash_Cost_per_day"]: edited_df[c] = pd.to_numeric(edited_df[c], errors="coerce").fillna(0.0)
    if "Overlap_OK" in edited_df.columns: edited_df["Overlap_OK"] = edited_df["Overlap_OK"].fillna(False).astype(bool)
    else: edited_df["Overlap_OK"] = False

    # Optional: apply schedule impacts from open RFIs
    apply_rfi = st.checkbox("Apply open RFI schedule impacts", value=False, help="Adds Schedule_Impact_Days from open RFIs to linked tasks (Related_Tasks).")
    edited_df_use = edited_df
    if apply_rfi:
        try:
            rfis = db_list_rfis(backend, limit=5000)
            if not rfis.empty:
                impacts_src = rfis.copy()
                impacts_src["schedule_impact_days"] = pd.to_numeric(impacts_src.get("schedule_impact_days", 0), errors="coerce").fillna(0).astype(int)
                impacts_src["status"] = impacts_src.get("status", "").astype(str)
                open_mask = ~impacts_src["status"].str.lower().isin(["closed", "cancelled"])
                impacts_src = impacts_src[open_mask & (impacts_src["schedule_impact_days"] > 0)]
                edited_df_use, impacts_applied = apply_rfi_impacts(edited_df, impacts_src)
                if impacts_applied.empty:
                    st.info("No open RFIs with schedule impact days to apply.")
                else:
                    st.success(f"Applied RFI impacts to {impacts_applied['Task'].nunique()} task(s).")
                    df_fullwidth(impacts_applied, hide_index=True, height=rows_to_height(len(impacts_applied)+2))
            else:
                st.info("No RFIs found in the current storage backend.")
        except Exception as e:
            st.warning(f"Could not apply RFI impacts: {e}")

    st.markdown("---")
    left, right = st.columns([1,1])
    with left:
        target_days = st.number_input("Target project duration (days)", min_value=1, value=30, step=1,
                                      help="Desired total project duration for crash analysis.")
    with right:
        overlap_frac = st.number_input("Fast-track overlap fraction", min_value=0.0, max_value=0.9, value=0.0, step=0.05,
            help="When Overlap OK is true, FS successors may start earlier by this fraction of predecessor duration.")

    c1, c2 = st.columns(2)
    clamp_ff = c1.checkbox("Clamp Free Float at ≥ 0", value=True,
                           help="Classic CPM reports Free Float as zero when negative. Uncheck to reveal negative FF caused by overlaps.")
    cal_mode = c2.checkbox("Calendar mode (map to dates)", value=True,
                           help="Convert ES/EF to working dates using your workweek and holidays.")
    if cal_mode:
        cw1, cw2 = st.columns([1,1])
        with cw1:
            proj_start = st.date_input("Project start date", value=date.today())
        with cw2:
            workdays = st.multiselect("Workdays", options=["Mon","Tue","Wed","Thu","Fri","Sat","Sun"],
                                      default=["Mon","Tue","Wed","Thu","Fri"],
                                      help="Working days for date conversion.")
        holidays_text = st.text_area("Holidays (YYYY-MM-DD, one per line)", height=80, placeholder="2025-11-27\n2025-12-25",
                                     help="Non-working dates to exclude.")
        holidays = [ln.strip() for ln in holidays_text.splitlines() if ln.strip()]
        cbd = make_cbd(workdays, holidays)

    cA, cB = st.columns([1,1])
    compute = cA.button("Compute CPM", type="secondary")
    run_crash = cB.button("Crash to Target", type="primary")

    if compute or run_crash:
        try:
            missing = [c for c in REQUIRED if c not in edited_df.columns]
            if missing:
                st.error(f"Missing required columns: {missing}"); st.stop()

            base_schedule, base_days = cpm_schedule(edited_df_use, overlap_frac, clamp_ff)
            if cal_mode: base_schedule = calendarize(base_schedule, proj_start, cbd)

            st.success(f"Baseline duration: {base_days} days")
            df_fullwidth(base_schedule, hide_index=True, height=rows_to_height(len(base_schedule)+5))
            st.subheader("Gantt (Baseline)")
            chart_fullwidth(gantt_chart(base_schedule))

            # Simple greedy crash loop
            def crash_once(df_cfg: pd.DataFrame, schedule: pd.DataFrame) -> Optional[str]:
                crit = schedule[schedule["Critical"]]
                if crit.empty: return None
                merged = crit.merge(
                    df_cfg[["Task","Duration","Min_Duration","Normal_Cost_per_day","Crash_Cost_per_day"]],
                    on="Task", how="left", suffixes=("_sched","_cfg")
                )
                merged["slope"] = (merged["Crash_Cost_per_day"] - merged["Normal_Cost_per_day"]).astype(float)
                # After merge, pandas suffixes duplicate column names (e.g., Duration -> Duration_sched/Duration_cfg).
                dur_col = "Duration_cfg" if "Duration_cfg" in merged.columns else "Duration"
                merged[dur_col] = pd.to_numeric(merged[dur_col], errors="coerce")
                merged["Min_Duration"] = pd.to_numeric(merged["Min_Duration"], errors="coerce")
                can = merged[merged[dur_col] > merged["Min_Duration"]]
                if can.empty: return None
                can = can.sort_values(["slope","ES"], kind="stable")
                return str(can.iloc[0]["Task"])

            def apply_crash(df_cfg: pd.DataFrame, task: str) -> pd.DataFrame:
                new = df_cfg.copy()
                new.loc[new["Task"]==task, "Duration"] = new.loc[new["Task"]==task, "Duration"] - 1
                return new

            def total_cost(df_cfg: pd.DataFrame) -> float:
                baseline = (df_cfg["Normal_Cost_per_day"] * df_cfg["Duration"].round(0)).sum()
                crashed_days = (df_cfg.get("_baseline_duration", df_cfg["Duration"]) - df_cfg["Duration"]).clip(lower=0)
                slope = (df_cfg["Crash_Cost_per_day"] - df_cfg["Normal_Cost_per_day"]).clip(lower=0)
                return float(baseline + (crashed_days * slope).sum())

            if run_crash:
                if target_days >= base_days:
                    st.info("Target ≥ baseline; nothing to crash.")
                else:
                    df_cfg = edited_df_use.copy()
                    df_cfg["_baseline_duration"] = df_cfg["Duration"]
                    log = []
                    schedule, cur = cpm_schedule(df_cfg, overlap_frac, clamp_ff)
                    while cur > target_days:
                        pick = crash_once(df_cfg, schedule)
                        if pick is None: break
                        df_cfg = apply_crash(df_cfg, pick)
                        log.append(f"Shortened '{pick}' by 1 day.")
                        schedule, cur = cpm_schedule(df_cfg, overlap_frac, clamp_ff)

                    crashed_schedule, final_days = schedule, cur
                    if cal_mode: crashed_schedule = calendarize(crashed_schedule, proj_start, cbd)
                    edited_df_use["_baseline_duration"] = edited_df_use["Duration"]; base_cost = total_cost(edited_df_use)
                    df_cfg["_baseline_duration"] = edited_df_use["Duration"]; crash_cost = total_cost(df_cfg)

                    st.markdown("### Crashed Scenario")
                    st.success(f"New duration: {final_days} days (target: {target_days})")
                    st.metric("Added cost (approx)", f"${crash_cost - base_cost:,.0f}")

                    st.subheader("Crashed Schedule")
                    df_fullwidth(crashed_schedule, hide_index=True, height=rows_to_height(len(crashed_schedule)+5))
                    st.subheader("Gantt (Crashed)")
                    chart_fullwidth(gantt_chart(crashed_schedule))
                    st.subheader("Crash Log")
                    if log:
                        for line in log: st.write("•", line)
                    else:
                        st.write("No feasible crashes — target may be below theoretical minimum.")

        except Exception as e:
            st.exception(e)



# =========================
# RFI Manager (page)
# =========================

def rfi_manager_page():
    st.header("RFI Manager")

    backend = get_backend_choice()

    # --- Create / Draft form ---
    with st.expander("Create new RFI", expanded=True):
        st.caption("Fields marked required: **Project**, **Subject**, **Question / Clarification needed**.")
        with st.form("new_rfi_form"):
            c1, c2, c3 = st.columns(3)
            project = c1.text_input(
                "Project *",
                value="",
                help="Your project/job identifier (e.g., 'LA River Rehab - Phase 2').",
            )
            discipline = c2.selectbox(
                "Discipline",
                ["General", "Civil", "Structural", "MEP", "Geotech", "Traffic", "Utilities", "Architectural"],
                index=0,
                help="Used for filtering and reporting.",
            )
            priority = c3.selectbox(
                "Priority",
                ["Low", "Normal", "High", "Urgent"],
                index=1,
                help="Used for triage and the aging dashboard.",
            )

            subject = st.text_input(
                "Subject *",
                help="Short title you would put in an email subject line.",
            )
            spec_section = st.text_input(
                "Spec / Drawing Ref (optional)",
                placeholder="e.g., 03 30 00 / S-201",
                help="Reference the spec section, sheet, detail, or addendum item.",
            )
            question = st.text_area(
                "Question / Clarification needed *",
                height=160,
                help="Write the actual question. Include assumptions and what decision you need.",
            )

            c4, c5, c6 = st.columns(3)
            due_date = c4.date_input(
                "Due date (optional)",
                value=None,
                help="When you need a response by (drives 'Overdue' logic).",
            )
            assignee_email = c5.text_input(
                "Assignee (optional)",
                placeholder="pm@company.com",
                help="Internal owner responsible for follow-up.",
            )
            to_emails = c6.text_input(
                "To (emails)",
                placeholder="architect@firm.com; engineer@firm.com",
                help="External recipients. Separate with commas or semicolons.",
            )

            cc_emails = st.text_input(
                "CC (emails)",
                placeholder="super@company.com",
                help="Optional CC list. Separate with commas or semicolons.",
            )

            st.markdown("**Schedule impact (optional)**")
            c7, c8, c9 = st.columns(3)
            related_tasks = c7.text_input(
                "Related schedule task(s)",
                placeholder="B - Foundations; C - Structure",
                help="Helps tie the RFI back to the schedule what-ifs.",
            )
            schedule_impact_days = c8.number_input(
                "Potential delay (days)",
                min_value=0,
                step=1,
                value=0,
                help="Your best estimate if this is not resolved quickly.",
            )
            cost_impact = c9.number_input(
                "Potential cost impact ($)",
                min_value=0.0,
                step=1000.0,
                value=0.0,
                help="Rough order-of-magnitude cost impact.",
            )

            thread_notes = st.text_area(
                "Notes / Thread",
                height=100,
                help="Paste email snippets, meeting notes, and decisions as they happen.",
            )

            links_raw = st.text_area(
                "Links (optional)",
                height=90,
                help="One link per line (plan room, BIM 360, Procore, Drive, etc.).",
            )

            attachments_files = st.file_uploader(
                "Attachments (optional)",
                type=["pdf", "csv", "docx", "doc", "txt"],
                accept_multiple_files=True,
                help=(
                    "Upload supporting docs. For Google Docs, download as .docx or PDF first. "
                    "Supported: PDF, CSV, Word (.doc/.docx), TXT."
                ),
            )

            cbtn1, cbtn2 = st.columns(2)

            save_draft = cbtn1.form_submit_button("Save draft", type="secondary", use_container_width=True)

            submit_rfi = cbtn2.form_submit_button("Submit", type="primary", use_container_width=True)
        if save_draft or submit_rfi:
            if not project.strip() or not subject.strip() or not question.strip():
                st.warning("Project, Subject, and Question are required.")
            else:
                new_status = "Draft" if save_draft else "Sent"
                sent_at = datetime.utcnow().isoformat() if submit_rfi else None
                rfi_id = db_upsert_rfi(
                    backend,
                    {
                        "id": None,
                        "created_at": datetime.utcnow().isoformat(),
                        "updated_at": datetime.utcnow().isoformat(),
                        "user_id": _current_user_label(),
                        "project": project.strip(),
                        "subject": subject.strip(),
                        "question": question.strip(),
                        "discipline": discipline,
                        "spec_section": spec_section.strip() or None,
                        "priority": priority,
                        "status": new_status,
                        "due_date": str(due_date) if due_date else None,
                        "assignee_email": assignee_email.strip() or None,
                        "to_emails": ";".join(parse_emails(to_emails)),
                        "cc_emails": ";".join(parse_emails(cc_emails)),
                        "related_tasks": related_tasks.strip() or None,
                        "schedule_impact_days": int(schedule_impact_days or 0),
                        "cost_impact": float(cost_impact or 0.0),
                        "last_sent_at": sent_at,
                        "last_reminded_at": None,
                        "last_response_at": None,
                        "thread_notes": thread_notes.strip() or None,
                    },
                )

                links = [u.strip() for u in (links_raw or "").splitlines() if u.strip()]
                if links:
                    db_add_rfi_links(backend, rfi_id, links)

                if attachments_files:
                    db_add_rfi_attachments(backend, rfi_id, attachments_files)

                st.success(f"{'Submitted' if submit_rfi else 'Saved draft'} RFI #{rfi_id}.")
                st.rerun()

    st.markdown("---")
    st.subheader("RFI List")

    rfis = db_list_rfis(backend)
    if rfis.empty:
        st.info("No RFIs yet. Create one above to start.")
        return

    # Filters
    f1, f2, f3, f4 = st.columns([0.28, 0.24, 0.24, 0.24])
    proj_opts = ["All"] + sorted([p for p in rfis.get("project", pd.Series(dtype=str)).dropna().unique().tolist() if str(p).strip()])
    status_opts = ["All", "Draft", "Sent", "Answered", "Closed"]
    prio_opts = ["All", "Low", "Normal", "High", "Urgent"]

    f_project = f1.selectbox("Project", proj_opts, index=0)
    f_status = f2.selectbox("Status", status_opts, index=0)
    f_priority = f3.selectbox("Priority", prio_opts, index=0)
    search = f4.text_input("Search", placeholder="subject / spec / keyword")

    view = rfis.copy()
    if f_project != "All":
        view = view[view["project"] == f_project]
    if f_status != "All":
        view = view[view["status"] == f_status]
    if f_priority != "All":
        view = view[view["priority"] == f_priority]
    if search.strip():
        s = search.strip().lower()
        for col in ["subject", "question", "spec_section", "thread_notes"]:
            if col not in view.columns:
                view[col] = ""
        mask = (
            view["subject"].fillna("").str.lower().str.contains(s)
            | view["question"].fillna("").str.lower().str.contains(s)
            | view["spec_section"].fillna("").str.lower().str.contains(s)
            | view["thread_notes"].fillna("").str.lower().str.contains(s)
        )
        view = view[mask]

    # Quick counts
    today = date.today()
    overdue = 0
    for _, r in view.iterrows():
        try:
            if r.get("status") not in ("Answered", "Closed") and r.get("due_date"):
                dd = date.fromisoformat(str(r.get("due_date")))
                if dd < today:
                    overdue += 1
        except Exception:
            pass

    cA, cB, cC = st.columns(3)
    cA.metric("Open RFIs", int((view["status"].fillna("") != "Closed").sum()))
    cB.metric("Overdue", int(overdue))
    cC.metric("Total", int(len(view)))

    st.dataframe(
        view[
            [
                "id",
                "project",
                "subject",
                "status",
                "priority",
                "due_date",
                "last_sent_at",
                "last_response_at",
                "schedule_impact_days",
                "related_tasks",
            ]
        ].sort_values(by=["id"], ascending=False),
        width='stretch',
        hide_index=True,
    )

    st.markdown("---")
    st.subheader("Open / Edit an RFI")

    rfi_ids = view["id"].dropna().astype(int).tolist()
    pick = st.selectbox("RFI ID", [0] + rfi_ids, index=0, help="Select an RFI to view/edit details.")
    if not pick:
        return

    rfi = rfis[rfis["id"] == pick].iloc[0].to_dict()

    with st.form("edit_rfi_form"):
        c1, c2, c3 = st.columns(3)
        status = c1.selectbox("Status", ["Draft", "Sent", "Answered", "Closed"], index=["Draft","Sent","Answered","Closed"].index(rfi.get("status","Draft")))
        priority = c2.selectbox("Priority", ["Low", "Normal", "High", "Urgent"], index=["Low","Normal","High","Urgent"].index(rfi.get("priority","Normal")))
        due = c3.date_input("Due date (optional)", value=date.fromisoformat(rfi["due_date"]) if rfi.get("due_date") else None)

        subject = st.text_input("Subject", value=rfi.get("subject",""), help="Short title.")
        spec_section = st.text_input("Spec / Drawing Ref (optional)", value=rfi.get("spec_section") or "")
        question = st.text_area("Question / Clarification needed", value=rfi.get("question",""), height=160)

        c4, c5, c6 = st.columns(3)
        assignee_email = c4.text_input("Assignee (optional)", value=rfi.get("assignee_email") or "")
        to_emails = c5.text_input("To (emails)", value=rfi.get("to_emails") or "")
        cc_emails = c6.text_input("CC (emails)", value=rfi.get("cc_emails") or "")

        st.markdown("**Schedule impact (optional)**")
        c7, c8, c9 = st.columns(3)
        related_tasks = c7.text_input("Related schedule task(s)", value=rfi.get("related_tasks") or "")
        schedule_impact_days = c8.number_input("Potential delay (days)", min_value=0, step=1, value=int(rfi.get("schedule_impact_days") or 0))
        cost_impact = c9.number_input("Potential cost impact ($)", min_value=0.0, step=1000.0, value=float(rfi.get("cost_impact") or 0.0))

        thread_notes = st.text_area("Notes / Thread", value=rfi.get("thread_notes") or "", height=140)

        save = st.form_submit_button("Save changes")

    if save:
        db_upsert_rfi(
            backend,
            {
                **rfi,
                "updated_at": datetime.utcnow().isoformat(),
                "status": status,
                "priority": priority,
                "due_date": str(due) if due else None,
                "subject": subject.strip(),
                "spec_section": spec_section.strip() or None,
                "question": question.strip(),
                "assignee_email": assignee_email.strip() or None,
                "to_emails": ";".join(parse_emails(to_emails)),
                "cc_emails": ";".join(parse_emails(cc_emails)),
                "related_tasks": related_tasks.strip() or None,
                "schedule_impact_days": int(schedule_impact_days or 0),
                "cost_impact": float(cost_impact or 0.0),
                "thread_notes": thread_notes.strip() or None,
            },
        )
        st.success("Saved.")
        st.rerun()

def aging_dashboard_page():
    st.header("Aging Dashboard")
    st.caption("Spot lagging submittals and RFIs. Use this as a lightweight 'PM pulse' panel.")

    # Tunables
    c1, c2, c3 = st.columns(3)
    submittal_lag_days = c1.number_input("Flag submittals older than (days)", min_value=1, value=14, step=1)
    rfi_remind_after = c2.number_input("Remind on RFIs after (days since last sent)", min_value=1, value=7, step=1)
    rfi_overdue_grace = c3.number_input("Overdue grace (days)", min_value=0, value=0, step=1)

    st.markdown("---")
    st.subheader("Submittal Runs (Memory Bank)")

    bank = db_list_submittals(backend)
    if bank.empty:
        st.info("No saved submittal runs yet.")
    else:
        # age from date_submitted if present, otherwise created_at
        dt = pd.to_datetime(bank.get("date_submitted"), errors="coerce")
        created = pd.to_datetime(bank.get("created_at"), errors="coerce")
        base = dt.fillna(created)
        bank = bank.copy()
        bank["Age_days"] = (pd.Timestamp.utcnow() - base).dt.days
        lag = bank[bank["Age_days"] >= int(submittal_lag_days)].copy()
        st.metric("Lagging runs", len(lag))
        df_fullwidth(lag.sort_values(["Age_days","created_at"], ascending=[False,False], kind="stable"), hide_index=True, height=rows_to_height(min(len(lag)+6, 50)))

    st.markdown("---")
    st.subheader("RFIs")
    rfis = db_list_rfis(backend, limit=5000)
    if rfis.empty:
        st.info("No RFIs yet.")
        return

    rfis = rfis.copy()
    rfis["created_at_dt"] = pd.to_datetime(rfis.get("created_at"), errors="coerce")
    rfis["last_sent_at_dt"] = pd.to_datetime(rfis.get("last_sent_at"), errors="coerce")
    rfis["due_dt"] = pd.to_datetime(rfis.get("due_date"), errors="coerce")

    open_mask = rfis.get("status","").isin(["Draft","Sent","Answered"])  # treat Answered as open until Closed
    open_rfis = rfis[open_mask].copy()

    # derived ages
    open_rfis["Age_days"] = (pd.Timestamp.utcnow() - open_rfis["created_at_dt"]).dt.days
    open_rfis["Days_since_sent"] = (pd.Timestamp.utcnow() - open_rfis["last_sent_at_dt"]).dt.days

    overdue_mask = open_rfis["due_dt"].notna() & ((open_rfis["due_dt"].dt.date + pd.to_timedelta(rfi_overdue_grace, unit='D')).dt.date < datetime.utcnow().date())
    remind_mask = open_rfis["last_sent_at_dt"].notna() & (open_rfis["Days_since_sent"] >= int(rfi_remind_after))

    m1, m2, m3 = st.columns(3)
    m1.metric("Open RFIs", len(open_rfis))
    m2.metric("Overdue", int(overdue_mask.sum()))
    m3.metric("Need reminder", int(remind_mask.sum()))

    show = open_rfis.copy()
    show["Overdue"] = overdue_mask
    show["Needs_Reminder"] = remind_mask

    show_cols = ["id","project","subject","status","priority","due_date","Age_days","Days_since_sent","Overdue","Needs_Reminder","assignee_email","to_emails"]
    for c in show_cols:
        if c not in show.columns:
            show[c] = None
    df_fullwidth(show[show_cols].sort_values(["Overdue","Needs_Reminder","Age_days"], ascending=[False,False,False], kind="stable"), hide_index=True, height=rows_to_height(min(len(show)+6, 70)))

    st.markdown("### Reminder sending")
    st.caption("Reminders send via your configured email provider (SendGrid/SMTP).")

    attach_pdf = st.checkbox("Attach RFI PDF to reminders", value=False)
    reminder_cc_owner = st.text_input("CC me (optional)", value="")

    if st.button("Send reminders to flagged RFIs"):
        if not remind_mask.any():
            st.info("No RFIs currently meet the reminder rule.")
        else:
            sent = 0
            failed = 0
            for _, r in open_rfis[remind_mask].iterrows():
                rfi = db_get_rfi(backend, int(r["id"])) or r.to_dict()
                recips = parse_emails(rfi.get("to_emails") or "")
                recips += parse_emails(rfi.get("cc_emails") or "")
                recips += parse_emails(reminder_cc_owner)
                recips = list(dict.fromkeys(recips))
                if not recips:
                    continue

                subj = f"Reminder: RFI #{rfi.get('id')}: {rfi.get('subject') or ''}"
                html = rfi_email_html(rfi) + "<p><i>Reminder:</i> Please respond when you can. Thank you.</p>"
                atts = None
                if attach_pdf:
                    try:
                        atts = [(f"RFI_{rfi.get('id')}.pdf", generate_rfi_pdf(rfi))]
                    except Exception:
                        atts = None

                ok, msg = send_email(recips, subj, html, attachments=atts)
                if ok:
                    sent += 1
                    rfi["last_reminded_at"] = datetime.utcnow().isoformat()
                    rfi["updated_at"] = datetime.utcnow().isoformat()
                    db_upsert_rfi(backend, rfi)
                else:
                    failed += 1
            st.success(f"Reminders sent: {sent}. Failed: {failed}.")

# =========================
# App Navigation
# =========================
page = st.session_state.get("__page__", "Submittal Checker")
if page == "Submittal Checker":
    submittal_checker_page()
elif page == "Schedule What-Ifs":
    schedule_whatifs_page()
elif page == "RFI Manager":
    rfi_manager_page()
else:
    aging_dashboard_page()

# =========================
# README (quick)
# =========================
# For local dev:
# 1) Put a .streamlit/secrets.toml like:
# [sheets]
# title = "SubmittalCheckerData"
# [google_oauth]
# client_id = "YOUR_ID"
# client_secret = "YOUR_SECRET"
# redirect_uri = "http://localhost:8501"
# [gcp_service_account]  # only if you use service-account mode
# type="service_account"
# project_id="..."
# private_key_id="..."
# private_key="-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
# client_email="svc@proj.iam.gserviceaccount.com"
# token_uri="https://oauth2.googleapis.com/token"
# [microsoft_oauth]
# client_id = "YOUR_APP_ID"
# client_secret = "YOUR_APP_SECRET"
# tenant_id = "common"
# redirect_uri = "http://localhost:8501"
#
# 2) pip install -r requirements.txt
# 3) streamlit run app.py
