# app.py
# Civil Mini-Apps: Submittal Checker + Schedule What-Ifs
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

# =========================
# App bootstrap
# =========================
st.set_page_config(page_title="Civil Mini-Apps", layout="wide")

# ---------- Width shims (handles Streamlit deprecations gracefully)
def df_fullwidth(df, **kwargs):
    try:
        return st.dataframe(df, use_container_width=True, **kwargs)
    except TypeError:
        return st.dataframe(df, **kwargs)

def editor_fullwidth(df, **kwargs):
    try:
        return st.data_editor(df, use_container_width=True, **kwargs)
    except TypeError:
        return st.data_editor(df, **kwargs)

def chart_fullwidth(chart, **kwargs):
    try:
        return st.altair_chart(chart, use_container_width=True, **kwargs)
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

    def _update_range(self, ws_name: str, start_cell: str, values_2d: list[list]):
        url = f"{self.base}/me/drive/items/{self._workbook_id}/workbook/worksheets/{urllib.parse.quote(ws_name)}/range(address='{start_cell}')"
        requests.patch(url, headers=self._hed(), data=json.dumps({"values": values_2d}))

    def _append_rows(self, ws_name: str, values_2d: list[list]):
        url = f"{self.base}/me/drive/items/{self._workbook_id}/workbook/worksheets/{urllib.parse.quote(ws_name)}/usedRange(valuesOnly=true)"
        r = requests.get(url, headers=self._hed()).json()
        row_count = r.get("rowCount", 0) or 0
        start_cell = f"A{row_count+1}" if row_count>0 else "A1"
        self._update_range(ws_name, start_cell, values_2d)

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
def db_save_preset(b: StorageBackend, name: str, payload: dict): return b.save_preset(name, payload)
def db_load_presets(b: StorageBackend) -> dict: return b.load_presets()
def db_delete_preset(b: StorageBackend, name: str): return b.delete_preset(name)
def db_save_submittal(b: StorageBackend, meta: dict, csv_bytes: bytes, spec_excerpt: str, sub_excerpt: str) -> int:
    return b.save_submittal(meta, csv_bytes, spec_excerpt, sub_excerpt)
def db_list_submittals(b: StorageBackend) -> pd.DataFrame: return b.list_submittals()
def db_get_submittal(b: StorageBackend, id_: int) -> dict: return b.get_submittal(id_)
def db_delete_submittal(b: StorageBackend, id_: int): return b.delete_submittal(id_)
def db_open_url_hint(b: StorageBackend, rec: dict) -> str | None: return b.open_url_hint(rec)

# =========================
# Sidebar: storage + auth
# =========================
st.sidebar.title("Civil Mini-Apps")
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
        data = io.BytesIO(uploaded.read())
        if name.endswith(".pdf"):  return _read_pdf(data)
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
                if f_client!="All":  view = view[view["client"]==f_client]
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

            base_schedule, base_days = cpm_schedule(edited_df, overlap_frac, clamp_ff)
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
                    on="Task", how="left"
                )
                merged["slope"] = (merged["Crash_Cost_per_day"] - merged["Normal_Cost_per_day"]).astype(float)
                can = merged[merged["Duration"] > merged["Min_Duration"]]
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
                    df_cfg = edited_df.copy()
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
                    edited_df["_baseline_duration"] = edited_df["Duration"]; base_cost = total_cost(edited_df)
                    df_cfg["_baseline_duration"] = edited_df["Duration"]; crash_cost = total_cost(df_cfg)

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
# App Navigation
# =========================
page = st.sidebar.radio("Pages", ["Submittal Checker", "Schedule What-Ifs"], index=0)
if page == "Submittal Checker":
    submittal_checker_page()
else:
    schedule_whatifs_page()

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
# client_id="YOUR_APP_ID"
# client_secret="YOUR_APP_SECRET"
# tenant_id="common"
# redirect_uri="http://localhost:8501"
#
# 2) pip install -r requirements.txt
# 3) streamlit run app.py
