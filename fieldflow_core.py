
from __future__ import annotations

import base64
import datetime as _dt
import io
import json
import os
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st

# ----------------------------
# Constants / UI
# ----------------------------
BACKEND_SQLITE = "Local (SQLite)"
BACKEND_GDRIVE = "Google Drive (OAuth) — files"
BACKEND_ONEDRIVE = "Microsoft OneDrive (OAuth) — files"

BACKEND_CHOICES = [BACKEND_SQLITE, BACKEND_GDRIVE, BACKEND_ONEDRIVE]

APP_NAME = "FieldFlow"
FOLDER_NAME = "FieldFlow"

LOGO_CANDIDATES = [
    Path(__file__).parent / "assets" / "FieldFlow_logo.png",
    Path(__file__).parent / "assets" / "fieldflow_logo.png",
]

# Google scopes
GOOGLE_SCOPES = [
    "https://www.googleapis.com/auth/drive.file",
    "https://www.googleapis.com/auth/drive.appdata",
]

# Microsoft scopes
MS_SCOPES = [
    "offline_access",
    "User.Read",
    "Files.ReadWrite",
]

# ----------------------------
# Small session helpers
# ----------------------------
def _ss_get(k: str, default=None):
    return st.session_state.get(k, default)

def _ss_set(k: str, v):
    st.session_state[k] = v

def _ss_del(k: str):
    if k in st.session_state:
        del st.session_state[k]

def _now_iso() -> str:
    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def _clear_query_params():
    # streamlit>=1.31: st.query_params exists
    try:
        st.query_params.clear()
    except Exception:
        try:
            st.experimental_set_query_params()
        except Exception:
            pass

def _rerun():
    try:
        st.rerun()
    except Exception:
        try:
            st.experimental_rerun()
        except Exception:
            pass

# ----------------------------
# Secrets helpers
# ----------------------------
def get_secret(path: str, default: str = "") -> str:
    """Fetch from st.secrets using dotted path, fallback to env."""
    # dotted: "google_oauth.client_id"
    parts = path.split(".")
    cur: Any = st.secrets
    try:
        for p in parts:
            cur = cur[p]
        if isinstance(cur, str):
            return cur
        return str(cur)
    except Exception:
        env_key = path.upper().replace(".", "_")
        return os.getenv(env_key, default)

# ----------------------------
# Storage backend interface
# ----------------------------
class StorageBackend:
    name: str = "Base"

    # presets (optional, can be no-op for file backends)
    def load_presets(self) -> Dict[str, Any]:
        return {}

    def save_preset(self, name: str, payload: Dict[str, Any]) -> None:
        return None

    def delete_preset(self, name: str) -> None:
        return None

    # submittal checks
    def save_submittal_check(self, payload: Dict[str, Any]) -> str:
        raise NotImplementedError

    def list_submittal_checks(self) -> pd.DataFrame:
        raise NotImplementedError

    def load_submittal_check(self, check_id: str) -> Optional[Dict[str, Any]]:
        raise NotImplementedError

    def delete_submittal_check(self, check_id: str) -> None:
        raise NotImplementedError

    # schedule runs
    def save_schedule_run(self, meta: Dict[str, Any], baseline_df: pd.DataFrame, crashed_df: pd.DataFrame) -> str:
        raise NotImplementedError

    def list_schedule_runs(self) -> pd.DataFrame:
        raise NotImplementedError

    def get_schedule_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        raise NotImplementedError

    def delete_schedule_run(self, run_id: str) -> None:
        raise NotImplementedError

# ----------------------------
# SQLite backend (reliable fallback)
# ----------------------------
class SQLiteBackend(StorageBackend):
    name = "SQLite"

    def __init__(self, path: str = "fieldflow.db"):
        self.path = path
        self._init_db()

    def _conn(self):
        conn = sqlite3.connect(self.path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._conn() as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS submittal_checks (
                id TEXT PRIMARY KEY,
                created_at TEXT,
                payload_json TEXT
            )""")
            conn.execute("""CREATE TABLE IF NOT EXISTS schedule_runs (
                id TEXT PRIMARY KEY,
                created_at TEXT,
                meta_json TEXT,
                baseline_csv TEXT,
                crashed_csv TEXT
            )""")
            conn.commit()

    def save_submittal_check(self, payload: Dict[str, Any]) -> str:
        cid = uuid.uuid4().hex
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO submittal_checks (id, created_at, payload_json) VALUES (?,?,?)",
                (cid, _now_iso(), json.dumps(payload)),
            )
            conn.commit()
        return cid

    def list_submittal_checks(self) -> pd.DataFrame:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT id, created_at, payload_json FROM submittal_checks ORDER BY created_at DESC"
            ).fetchall()
        out = []
        for r in rows:
            try:
                payload = json.loads(r["payload_json"])
            except Exception:
                payload = {}
            out.append({"id": r["id"], "created_at": r["created_at"], **payload})
        return pd.DataFrame(out)

    def load_submittal_check(self, check_id: str) -> Optional[Dict[str, Any]]:
        with self._conn() as conn:
            r = conn.execute("SELECT payload_json FROM submittal_checks WHERE id=?", (check_id,)).fetchone()
        if not r:
            return None
        try:
            return json.loads(r["payload_json"])
        except Exception:
            return None

    def delete_submittal_check(self, check_id: str) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM submittal_checks WHERE id=?", (check_id,))
            conn.commit()

    def save_schedule_run(self, meta: Dict[str, Any], baseline_df: pd.DataFrame, crashed_df: pd.DataFrame) -> str:
        run_id = uuid.uuid4().hex
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO schedule_runs (id, created_at, meta_json, baseline_csv, crashed_csv) VALUES (?,?,?,?,?)",
                (
                    run_id,
                    _now_iso(),
                    json.dumps(meta),
                    baseline_df.to_csv(index=False),
                    crashed_df.to_csv(index=False),
                ),
            )
            conn.commit()
        return run_id

    def list_schedule_runs(self) -> pd.DataFrame:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT id, created_at, meta_json FROM schedule_runs ORDER BY created_at DESC"
            ).fetchall()
        out = []
        for r in rows:
            try:
                meta = json.loads(r["meta_json"])
            except Exception:
                meta = {}
            out.append({"run_id": r["id"], "created_at": r["created_at"], **meta})
        return pd.DataFrame(out)

    def get_schedule_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        with self._conn() as conn:
            r = conn.execute(
                "SELECT created_at, meta_json, baseline_csv, crashed_csv FROM schedule_runs WHERE id=?",
                (run_id,),
            ).fetchone()
        if not r:
            return None
        try:
            meta = json.loads(r["meta_json"])
        except Exception:
            meta = {}
        baseline_df = pd.read_csv(io.StringIO(r["baseline_csv"]))
        crashed_df = pd.read_csv(io.StringIO(r["crashed_csv"]))
        return {"run_id": run_id, "created_at": r["created_at"], "meta": meta, "baseline_df": baseline_df, "crashed_df": crashed_df}

    def delete_schedule_run(self, run_id: str) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM schedule_runs WHERE id=?", (run_id,))
            conn.commit()

# ----------------------------
# Google Drive backend (files)
# ----------------------------
def _google_credentials_from_session():
    data = _ss_get("__google_tokens__")
    if not data:
        return None
    try:
        from google.oauth2.credentials import Credentials
        creds = Credentials(
            token=data.get("token"),
            refresh_token=data.get("refresh_token"),
            token_uri="https://oauth2.googleapis.com/token",
            client_id=get_secret("google_oauth.client_id"),
            client_secret=get_secret("google_oauth.client_secret"),
            scopes=GOOGLE_SCOPES,
        )
        # expiry handling
        exp = data.get("expiry")
        if exp:
            try:
                creds.expiry = _dt.datetime.fromisoformat(exp.replace("Z",""))
            except Exception:
                pass
        if creds and creds.expired and creds.refresh_token:
            from google.auth.transport.requests import Request
            creds.refresh(Request())
            _ss_set("__google_tokens__", {
                "token": creds.token,
                "refresh_token": creds.refresh_token,
                "expiry": (creds.expiry.isoformat() if creds.expiry else ""),
            })
        return creds
    except Exception:
        return None

class GoogleDriveBackend(StorageBackend):
    name = "Google Drive"

    def __init__(self):
        creds = _google_credentials_from_session()
        if creds is None:
            raise RuntimeError("Google not authenticated.")
        from googleapiclient.discovery import build
        self.drive = build("drive", "v3", credentials=creds, cache_discovery=False)
        self.folder_id = self._ensure_folder()

    def _ensure_folder(self) -> str:
        q = "mimeType='application/vnd.google-apps.folder' and name='%s' and trashed=false" % FOLDER_NAME
        res = self.drive.files().list(q=q, spaces="drive", fields="files(id,name)").execute()
        files = res.get("files", [])
        if files:
            return files[0]["id"]
        body = {"name": FOLDER_NAME, "mimeType": "application/vnd.google-apps.folder"}
        created = self.drive.files().create(body=body, fields="id").execute()
        return created["id"]

    def _upload_bytes(self, name: str, data: bytes, mime: str) -> str:
        from googleapiclient.http import MediaIoBaseUpload
        media = MediaIoBaseUpload(io.BytesIO(data), mimetype=mime, resumable=False)
        body = {"name": name, "parents": [self.folder_id]}
        created = self.drive.files().create(body=body, media_body=media, fields="id,name,createdTime").execute()
        return created["id"]

    def _list_in_folder(self) -> List[Dict[str, Any]]:
        q = f"'{self.folder_id}' in parents and trashed=false"
        res = self.drive.files().list(q=q, fields="files(id,name,createdTime,mimeType)", pageSize=200).execute()
        return res.get("files", [])

    def _download(self, file_id: str) -> bytes:
        from googleapiclient.http import MediaIoBaseDownload
        request = self.drive.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        return fh.getvalue()

    def _delete(self, file_id: str):
        self.drive.files().delete(fileId=file_id).execute()

    # submittal checks as JSON files
    def save_submittal_check(self, payload: Dict[str, Any]) -> str:
        cid = uuid.uuid4().hex
        payload = dict(payload)
        payload.setdefault("created_at", _now_iso())
        name = f"submittal_{cid}.json"
        self._upload_bytes(name, json.dumps(payload, indent=2).encode("utf-8"), "application/json")
        return cid

    def list_submittal_checks(self) -> pd.DataFrame:
        files = [f for f in self._list_in_folder() if f["name"].startswith("submittal_") and f["name"].endswith(".json")]
        out = []
        for f in sorted(files, key=lambda x: x.get("createdTime",""), reverse=True)[:50]:
            cid = f["name"].replace("submittal_","").replace(".json","")
            out.append({"id": cid, "created_at": f.get("createdTime",""), "file_id": f["id"], "name": f["name"]})
        return pd.DataFrame(out)

    def load_submittal_check(self, check_id: str) -> Optional[Dict[str, Any]]:
        target = f"submittal_{check_id}.json"
        files = [f for f in self._list_in_folder() if f["name"] == target]
        if not files:
            return None
        data = self._download(files[0]["id"])
        try:
            return json.loads(data.decode("utf-8"))
        except Exception:
            return None

    def delete_submittal_check(self, check_id: str) -> None:
        target = f"submittal_{check_id}.json"
        files = [f for f in self._list_in_folder() if f["name"] == target]
        if files:
            self._delete(files[0]["id"])

    # schedule runs: meta json + baseline/crashed csv
    def save_schedule_run(self, meta: Dict[str, Any], baseline_df: pd.DataFrame, crashed_df: pd.DataFrame) -> str:
        run_id = uuid.uuid4().hex
        created_at = _now_iso()
        meta_obj = dict(meta)
        meta_obj.setdefault("created_at", created_at)
        meta_name = f"schedule_{run_id}_meta.json"
        base_name = f"schedule_{run_id}_baseline.csv"
        crash_name = f"schedule_{run_id}_crashed.csv"
        self._upload_bytes(meta_name, json.dumps(meta_obj, indent=2).encode("utf-8"), "application/json")
        self._upload_bytes(base_name, baseline_df.to_csv(index=False).encode("utf-8"), "text/csv")
        self._upload_bytes(crash_name, crashed_df.to_csv(index=False).encode("utf-8"), "text/csv")
        return run_id

    def list_schedule_runs(self) -> pd.DataFrame:
        files = [f for f in self._list_in_folder() if f["name"].startswith("schedule_") and f["name"].endswith("_meta.json")]
        out = []
        for f in sorted(files, key=lambda x: x.get("createdTime",""), reverse=True)[:50]:
            run_id = f["name"].replace("schedule_","").replace("_meta.json","")
            # try parse some meta fields quickly by downloading small json
            meta = {}
            try:
                meta = json.loads(self._download(f["id"]).decode("utf-8"))
            except Exception:
                meta = {}
            out.append({
                "run_id": run_id,
                "created_at": meta.get("created_at", f.get("createdTime","")),
                **{k: meta.get(k) for k in ["company","client","project","scenario","kind","baseline_days","crashed_days"]},
            })
        return pd.DataFrame(out)

    def get_schedule_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        meta_name = f"schedule_{run_id}_meta.json"
        base_name = f"schedule_{run_id}_baseline.csv"
        crash_name = f"schedule_{run_id}_crashed.csv"
        files = {f["name"]: f["id"] for f in self._list_in_folder()}
        if meta_name not in files or base_name not in files or crash_name not in files:
            return None
        meta = json.loads(self._download(files[meta_name]).decode("utf-8"))
        baseline_df = pd.read_csv(io.StringIO(self._download(files[base_name]).decode("utf-8")))
        crashed_df = pd.read_csv(io.StringIO(self._download(files[crash_name]).decode("utf-8")))
        return {"run_id": run_id, "created_at": meta.get("created_at",""), "meta": meta, "baseline_df": baseline_df, "crashed_df": crashed_df}

    def delete_schedule_run(self, run_id: str) -> None:
        names = [
            f"schedule_{run_id}_meta.json",
            f"schedule_{run_id}_baseline.csv",
            f"schedule_{run_id}_crashed.csv",
        ]
        files = {f["name"]: f["id"] for f in self._list_in_folder()}
        for n in names:
            if n in files:
                self._delete(files[n])

# ----------------------------
# OneDrive backend (files)
# ----------------------------
def _ms_token_valid(tokens: Dict[str, Any]) -> bool:
    if not tokens or "access_token" not in tokens:
        return False
    exp = tokens.get("expires_at", 0)
    return time.time() < exp - 60

def _ms_refresh(tokens: Dict[str, Any]) -> Dict[str, Any]:
    refresh_token = tokens.get("refresh_token")
    if not refresh_token:
        return tokens
    tenant = get_secret("microsoft_oauth.tenant_id", "common")
    token_url = f"https://login.microsoftonline.com/{tenant}/oauth2/v2.0/token"
    data = {
        "client_id": get_secret("microsoft_oauth.client_id"),
        "client_secret": get_secret("microsoft_oauth.client_secret"),
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "redirect_uri": get_secret("microsoft_oauth.redirect_uri"),
        "scope": " ".join(MS_SCOPES),
    }
    r = requests.post(token_url, data=data, timeout=30)
    r.raise_for_status()
    js = r.json()
    tokens = {
        "access_token": js["access_token"],
        "refresh_token": js.get("refresh_token", refresh_token),
        "expires_at": time.time() + int(js.get("expires_in", 3599)),
    }
    _ss_set("__ms_tokens__", tokens)
    return tokens

class OneDriveBackend(StorageBackend):
    name = "OneDrive"

    def __init__(self):
        tokens = _ss_get("__ms_tokens__")
        if not tokens:
            raise RuntimeError("Microsoft not authenticated.")
        if not _ms_token_valid(tokens):
            try:
                tokens = _ms_refresh(tokens)
            except Exception as e:
                raise RuntimeError(f"Microsoft token refresh failed: {e}")
        self.tokens = tokens
        self.base_url = "https://graph.microsoft.com/v1.0"
        self.folder_item = self._ensure_folder()

    def _headers(self):
        return {"Authorization": f"Bearer {self.tokens['access_token']}"}

    def _request(self, method: str, url: str, **kwargs):
        # throttling-aware request
        for attempt in range(6):
            r = requests.request(method, url, headers=self._headers(), timeout=30, **kwargs)
            if r.status_code in (429, 503, 504):
                ra = r.headers.get("Retry-After")
                sleep_s = int(ra) if ra and ra.isdigit() else min(2 ** attempt, 30)
                time.sleep(sleep_s)
                continue
            if r.status_code >= 400:
                try:
                    msg = r.json()
                except Exception:
                    msg = r.text
                raise RuntimeError(f"Graph error {r.status_code}: {msg}")
            return r
        raise RuntimeError("Graph throttling: too many retries.")

    def _ensure_folder(self) -> Dict[str, Any]:
        # list approot children and find/create FieldFlow folder
        url = f"{self.base_url}/me/drive/special/approot/children?$select=id,name,folder"
        r = self._request("GET", url)
        items = r.json().get("value", [])
        for it in items:
            if it.get("name") == FOLDER_NAME and "folder" in it:
                return it
        # create
        url = f"{self.base_url}/me/drive/special/approot/children"
        body = {"name": FOLDER_NAME, "folder": {}, "@microsoft.graph.conflictBehavior": "fail"}
        r = self._request("POST", url, json=body)
        return r.json()

    def _upload_bytes(self, filename: str, data: bytes) -> Dict[str, Any]:
        # PUT content into approot/FieldFlow
        url = f"{self.base_url}/me/drive/special/approot:/{FOLDER_NAME}/{filename}:/content"
        r = self._request("PUT", url, data=data)
        return r.json()

    def _list(self) -> List[Dict[str, Any]]:
        url = f"{self.base_url}/me/drive/items/{self.folder_item['id']}/children?$select=id,name,createdDateTime"
        r = self._request("GET", url)
        return r.json().get("value", [])

    def _download(self, item_id: str) -> bytes:
        url = f"{self.base_url}/me/drive/items/{item_id}/content"
        r = requests.get(url, headers=self._headers(), timeout=30, allow_redirects=True)
        if r.status_code >= 400:
            raise RuntimeError(f"Download failed {r.status_code}: {r.text}")
        return r.content

    def _delete(self, item_id: str):
        url = f"{self.base_url}/me/drive/items/{item_id}"
        self._request("DELETE", url)

    def save_submittal_check(self, payload: Dict[str, Any]) -> str:
        cid = uuid.uuid4().hex
        payload = dict(payload)
        payload.setdefault("created_at", _now_iso())
        name = f"submittal_{cid}.json"
        self._upload_bytes(name, json.dumps(payload, indent=2).encode("utf-8"))
        return cid

    def list_submittal_checks(self) -> pd.DataFrame:
        items = [it for it in self._list() if it["name"].startswith("submittal_") and it["name"].endswith(".json")]
        out = []
        for it in sorted(items, key=lambda x: x.get("createdDateTime",""), reverse=True)[:50]:
            cid = it["name"].replace("submittal_","").replace(".json","")
            out.append({"id": cid, "created_at": it.get("createdDateTime",""), "item_id": it["id"], "name": it["name"]})
        return pd.DataFrame(out)

    def load_submittal_check(self, check_id: str) -> Optional[Dict[str, Any]]:
        target = f"submittal_{check_id}.json"
        items = [it for it in self._list() if it["name"] == target]
        if not items:
            return None
        data = self._download(items[0]["id"])
        try:
            return json.loads(data.decode("utf-8"))
        except Exception:
            return None

    def delete_submittal_check(self, check_id: str) -> None:
        target = f"submittal_{check_id}.json"
        items = [it for it in self._list() if it["name"] == target]
        if items:
            self._delete(items[0]["id"])

    def save_schedule_run(self, meta: Dict[str, Any], baseline_df: pd.DataFrame, crashed_df: pd.DataFrame) -> str:
        run_id = uuid.uuid4().hex
        meta_obj = dict(meta)
        meta_obj.setdefault("created_at", _now_iso())
        self._upload_bytes(f"schedule_{run_id}_meta.json", json.dumps(meta_obj, indent=2).encode("utf-8"))
        self._upload_bytes(f"schedule_{run_id}_baseline.csv", baseline_df.to_csv(index=False).encode("utf-8"))
        self._upload_bytes(f"schedule_{run_id}_crashed.csv", crashed_df.to_csv(index=False).encode("utf-8"))
        return run_id

    def list_schedule_runs(self) -> pd.DataFrame:
        items = [it for it in self._list() if it["name"].startswith("schedule_") and it["name"].endswith("_meta.json")]
        out = []
        for it in sorted(items, key=lambda x: x.get("createdDateTime",""), reverse=True)[:50]:
            run_id = it["name"].replace("schedule_","").replace("_meta.json","")
            meta = {}
            try:
                meta = json.loads(self._download(it["id"]).decode("utf-8"))
            except Exception:
                meta = {}
            out.append({
                "run_id": run_id,
                "created_at": meta.get("created_at", it.get("createdDateTime","")),
                **{k: meta.get(k) for k in ["company","client","project","scenario","kind","baseline_days","crashed_days"]},
            })
        return pd.DataFrame(out)

    def get_schedule_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        names = {
            "meta": f"schedule_{run_id}_meta.json",
            "baseline": f"schedule_{run_id}_baseline.csv",
            "crashed": f"schedule_{run_id}_crashed.csv",
        }
        items = {it["name"]: it for it in self._list()}
        if names["meta"] not in items or names["baseline"] not in items or names["crashed"] not in items:
            return None
        meta = json.loads(self._download(items[names["meta"]]["id"]).decode("utf-8"))
        baseline_df = pd.read_csv(io.StringIO(self._download(items[names["baseline"]]["id"]).decode("utf-8")))
        crashed_df = pd.read_csv(io.StringIO(self._download(items[names["crashed"]]["id"]).decode("utf-8")))
        return {"run_id": run_id, "created_at": meta.get("created_at",""), "meta": meta, "baseline_df": baseline_df, "crashed_df": crashed_df}

    def delete_schedule_run(self, run_id: str) -> None:
        items = {it["name"]: it for it in self._list()}
        for n in [f"schedule_{run_id}_meta.json", f"schedule_{run_id}_baseline.csv", f"schedule_{run_id}_crashed.csv"]:
            if n in items:
                self._delete(items[n]["id"])

# ----------------------------
# OAuth flows
# ----------------------------
def _google_auth_url() -> Optional[str]:
    cid = get_secret("google_oauth.client_id")
    cs = get_secret("google_oauth.client_secret")
    ru = get_secret("google_oauth.redirect_uri")
    if not (cid and cs and ru):
        return None
    try:
        from google_auth_oauthlib.flow import Flow
        client_config = {
            "web": {
                "client_id": cid,
                "client_secret": cs,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
            }
        }
        flow = Flow.from_client_config(client_config, scopes=GOOGLE_SCOPES, redirect_uri=ru)
        import secrets as _secrets
        state = "g_" + _secrets.token_urlsafe(16)
        _ss_set("__google_state__", state)
        auth_url, _ = flow.authorization_url(
            access_type="offline",
            include_granted_scopes="true",
            prompt="consent",
            state=state,
        )
        return auth_url
    except Exception:
        return None

def _google_handle_callback(code: str, state: str) -> bool:
    if not state.startswith("g_"):
        return False
    expected = _ss_get("__google_state__")
    if expected and state != expected:
        return False
    cid = get_secret("google_oauth.client_id")
    cs = get_secret("google_oauth.client_secret")
    ru = get_secret("google_oauth.redirect_uri")
    from google_auth_oauthlib.flow import Flow
    client_config = {
        "web": {
            "client_id": cid,
            "client_secret": cs,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
        }
    }
    flow = Flow.from_client_config(client_config, scopes=GOOGLE_SCOPES, redirect_uri=ru)
    flow.fetch_token(code=code)
    creds = flow.credentials
    _ss_set("__google_tokens__", {
        "token": creds.token,
        "refresh_token": creds.refresh_token,
        "expiry": (creds.expiry.isoformat() if creds.expiry else ""),
    })
    return True

def _ms_auth_url() -> Optional[str]:
    cid = get_secret("microsoft_oauth.client_id")
    cs = get_secret("microsoft_oauth.client_secret")
    ru = get_secret("microsoft_oauth.redirect_uri")
    tenant = get_secret("microsoft_oauth.tenant_id", "common")
    if not (cid and cs and ru and tenant):
        return None
    import secrets as _secrets
    state = "m_" + _secrets.token_urlsafe(16)
    _ss_set("__ms_state__", state)
    params = {
        "client_id": cid,
        "response_type": "code",
        "redirect_uri": ru,
        "response_mode": "query",
        "scope": " ".join(MS_SCOPES),
        "state": state,
    }
    from urllib.parse import urlencode
    return f"https://login.microsoftonline.com/{tenant}/oauth2/v2.0/authorize?{urlencode(params)}"

def _ms_handle_callback(code: str, state: str) -> bool:
    if not state.startswith("m_"):
        return False
    expected = _ss_get("__ms_state__")
    if expected and state != expected:
        return False
    cid = get_secret("microsoft_oauth.client_id")
    cs = get_secret("microsoft_oauth.client_secret")
    ru = get_secret("microsoft_oauth.redirect_uri")
    tenant = get_secret("microsoft_oauth.tenant_id", "common")
    token_url = f"https://login.microsoftonline.com/{tenant}/oauth2/v2.0/token"
    data = {
        "client_id": cid,
        "client_secret": cs,
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": ru,
        "scope": " ".join(MS_SCOPES),
    }
    r = requests.post(token_url, data=data, timeout=30)
    r.raise_for_status()
    js = r.json()
    tokens = {
        "access_token": js["access_token"],
        "refresh_token": js.get("refresh_token", ""),
        "expires_at": time.time() + int(js.get("expires_in", 3599)),
    }
    _ss_set("__ms_tokens__", tokens)
    return True

def _handle_oauth_callback_if_present():
    # Use query params to detect code/state
    try:
        qp = dict(st.query_params)
    except Exception:
        qp = st.experimental_get_query_params()
    code = qp.get("code")
    state = qp.get("state")
    if isinstance(code, list): code = code[0] if code else None
    if isinstance(state, list): state = state[0] if state else None
    if not code or not state:
        return
    ok = False
    err = None
    try:
        if state.startswith("g_"):
            ok = _google_handle_callback(code, state)
        elif state.startswith("m_"):
            ok = _ms_handle_callback(code, state)
    except Exception as e:
        err = str(e)
    _clear_query_params()
    if ok:
        _rerun()
    if err:
        st.sidebar.error(f"OAuth callback failed: {err}")

# ----------------------------
# Backend selection (safe)
# ----------------------------
def get_backend() -> StorageBackend:
    choice = _ss_get("__backend_choice__", BACKEND_SQLITE)
    if choice == BACKEND_GDRIVE:
        try:
            return GoogleDriveBackend()
        except Exception as e:
            st.sidebar.warning(f"Google storage unavailable: {e}. Using Local instead.")
            return SQLiteBackend()
    if choice == BACKEND_ONEDRIVE:
        try:
            return OneDriveBackend()
        except Exception as e:
            st.sidebar.warning(f"Microsoft storage unavailable: {e}. Using Local instead.")
            return SQLiteBackend()
    return SQLiteBackend()

# ----------------------------
# Sidebar
# ----------------------------
def render_sidebar(active_page: str = "") -> StorageBackend:
    _handle_oauth_callback_if_present()

    # Logo at very top
    for p in LOGO_CANDIDATES:
        if p.exists():
            st.sidebar.image(str(p), use_container_width=True)
            break

    st.sidebar.markdown("### Storage")
    choice = st.sidebar.selectbox("Save location", BACKEND_CHOICES, index=BACKEND_CHOICES.index(_ss_get("__backend_choice__", BACKEND_SQLITE)))
    _ss_set("__backend_choice__", choice)

    st.sidebar.markdown("### Sign in")
    g_tokens = _ss_get("__google_tokens__")
    m_tokens = _ss_get("__ms_tokens__")

    c1, c2 = st.sidebar.columns(2)
    with c1:
        if g_tokens:
            if st.button("Sign out Google"):
                _ss_del("__google_tokens__")
                _ss_del("__google_state__")
                _rerun()
        else:
            url = _google_auth_url()
            if url:
                st.link_button("Sign in Google", url, use_container_width=True)
            else:
                st.caption("Google OAuth not configured.")
    with c2:
        if m_tokens:
            if st.button("Sign out Microsoft"):
                _ss_del("__ms_tokens__")
                _ss_del("__ms_state__")
                _rerun()
        else:
            url = _ms_auth_url()
            if url:
                st.link_button("Sign in Microsoft", url, use_container_width=True)
            else:
                st.caption("Microsoft OAuth not configured.")

    st.sidebar.markdown("---")
    st.sidebar.caption(f"Active page: {active_page}")
    return get_backend()

# ----------------------------
# Pages
# ----------------------------
def submittal_checker_page() -> None:
    backend = render_sidebar("Submittal Checker")
    st.title("Submittal Checker (simple)")
    st.caption("Paste spec + submittal text, run analysis, and save the run. Saved runs appear under Saved Results.")

    spec = st.text_area("Paste spec text", height=200, placeholder="Paste specification clauses…")
    sub = st.text_area("Paste submittal text", height=200, placeholder="Paste submittal content…")

    if st.button("Analyze", type="primary"):
        payload = {
            "created_at": _now_iso(),
            "spec_len": len(spec or ""),
            "submittal_len": len(sub or ""),
        }
        cid = backend.save_submittal_check(payload)
        st.success(f"Saved check: {cid}")
        st.json(payload)

def _load_schedule_csv(uploaded) -> pd.DataFrame:
    return pd.read_csv(uploaded)

def _compute_cpm(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    # Minimal CPM: assumes columns task_id, duration, predecessors (comma-separated ids)
    df = df.copy()
    df["duration"] = pd.to_numeric(df["duration"], errors="coerce").fillna(0).astype(int)
    df["predecessors"] = df.get("predecessors", "").fillna("")
    preds = {row["task_id"]: [p.strip() for p in str(row["predecessors"]).split(",") if p.strip()] for _, row in df.iterrows()}
    # forward pass
    es, ef = {}, {}
    for _ in range(len(df)*2):
        changed=False
        for _, row in df.iterrows():
            tid=row["task_id"]
            pre=preds.get(tid,[])
            if all(p in ef for p in pre):
                new_es = max([ef[p] for p in pre], default=0)
                new_ef = new_es + int(row["duration"])
                if es.get(tid)!=new_es or ef.get(tid)!=new_ef:
                    es[tid]=new_es; ef[tid]=new_ef; changed=True
        if not changed:
            break
    project_duration = max(ef.values()) if ef else 0
    # backward pass
    ls, lf = {}, {}
    succs = {t: [] for t in df["task_id"]}
    for t, pre in preds.items():
        for p in pre:
            succs.setdefault(p, []).append(t)
    for _ in range(len(df)*2):
        changed=False
        for _, row in df.iterrows():
            tid=row["task_id"]
            su=succs.get(tid,[])
            if not su:
                new_lf = project_duration
            elif all(s in ls for s in su):
                new_lf = min([ls[s] for s in su])
            else:
                continue
            new_ls = new_lf - int(row["duration"])
            if lf.get(tid)!=new_lf or ls.get(tid)!=new_ls:
                lf[tid]=new_lf; ls[tid]=new_ls; changed=True
        if not changed:
            break
    df["ES"] = df["task_id"].map(es).fillna(0).astype(int)
    df["EF"] = df["task_id"].map(ef).fillna(df["ES"] + df["duration"]).astype(int)
    df["LS"] = df["task_id"].map(ls).fillna(df["ES"]).astype(int)
    df["LF"] = df["task_id"].map(lf).fillna(df["EF"]).astype(int)
    df["float"] = (df["LS"] - df["ES"]).astype(int)
    return df, project_duration

def _crash_to_target(df: pd.DataFrame, target_days: int) -> Tuple[pd.DataFrame, int]:
    # Simple crash: reduce durations on zero-float tasks first (by 1 day each loop)
    df2=df.copy()
    df2["duration"]=pd.to_numeric(df2["duration"], errors="coerce").fillna(0).astype(int)
    for _ in range(1000):
        cpm, dur = _compute_cpm(df2)
        if dur <= target_days:
            return cpm, dur
        crit = cpm.sort_values(["float","duration"], ascending=[True, False])
        # reduce first critical task with duration>0
        idx = crit.index[0]
        if df2.loc[idx,"duration"]<=0:
            break
        df2.loc[idx,"duration"] -= 1
    cpm, dur = _compute_cpm(df2)
    return cpm, dur

def schedule_whatifs_page() -> None:
    backend = render_sidebar("Schedule What-Ifs")
    st.title("Schedule What-Ifs")
    st.caption("Upload a simple task CSV, compute CPM, optionally crash to a target duration, then save/browse/download results.")

    up = st.file_uploader("Upload tasks CSV", type=["csv"])
    if up is not None:
        df = _load_schedule_csv(up)
        st.dataframe(df, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Compute CPM", type="primary"):
                cpm, dur = _compute_cpm(df)
                _ss_set("__sched_baseline__", cpm.to_json(orient="records"))
                _ss_set("__sched_baseline_days__", int(dur))
        with col2:
            target = st.number_input("Target duration (days)", min_value=0, value=20)
            if st.button("Crash to target"):
                cpm, dur = _crash_to_target(df, int(target))
                _ss_set("__sched_crashed__", cpm.to_json(orient="records"))
                _ss_set("__sched_crashed_days__", int(dur))

    # Display computed
    base_df = None
    crash_df = None
    if _ss_get("__sched_baseline__"):
        base_df = pd.read_json(io.StringIO(_ss_get("__sched_baseline__")), orient="records")
        st.markdown("### Baseline CPM")
        st.write(f"Duration: **{_ss_get('__sched_baseline_days__', 0)} days**")
        st.dataframe(base_df, use_container_width=True)
        st.download_button("Download baseline CSV", data=base_df.to_csv(index=False), file_name="schedule_baseline.csv", mime="text/csv", use_container_width=True)
    if _ss_get("__sched_crashed__"):
        crash_df = pd.read_json(io.StringIO(_ss_get("__sched_crashed__")), orient="records")
        st.markdown("### Crashed CPM")
        st.write(f"Duration: **{_ss_get('__sched_crashed_days__', 0)} days**")
        st.dataframe(crash_df, use_container_width=True)
        st.download_button("Download crashed CSV", data=crash_df.to_csv(index=False), file_name="schedule_crashed.csv", mime="text/csv", use_container_width=True)

    st.markdown("---")
    st.markdown("## Save this schedule run")
    company = st.text_input("Company", value="")
    client = st.text_input("Client", value="")
    project = st.text_input("Project", value="")
    scenario = st.text_input("Scenario name", value="")

    kind = st.selectbox("Save which version?", ["baseline", "crashed", "both"], index=2)
    can_save = (base_df is not None) and (crash_df is not None or kind != "crashed")

    if st.button("Save schedule run"):
        if not can_save:
            st.error("Compute CPM first (and Crash if you chose crashed/both).")
        else:
            meta = {
                "company": company,
                "client": client,
                "project": project,
                "scenario": scenario,
                "kind": kind,
                "baseline_days": int(_ss_get("__sched_baseline_days__", 0)),
                "crashed_days": int(_ss_get("__sched_crashed_days__", 0)) if crash_df is not None else None,
            }
            if kind == "baseline":
                rid = backend.save_schedule_run(meta, base_df, base_df)
            elif kind == "crashed":
                rid = backend.save_schedule_run(meta, crash_df, crash_df)
            else:
                rid = backend.save_schedule_run(meta, base_df, crash_df if crash_df is not None else base_df)
            st.success(f"Saved schedule run: {rid}")

    st.markdown("---")
    st.markdown("## Browse saved schedule runs")
    try:
        runs = backend.list_schedule_runs()
    except Exception as e:
        st.warning(f"Could not load saved runs from this backend: {e}")
        runs = pd.DataFrame()

    if not runs.empty:
        st.dataframe(runs, use_container_width=True)
        sel = st.text_input("Enter a run_id to open", value="")
        if st.button("Open run"):
            run = backend.get_schedule_run(sel.strip())
            if not run:
                st.error("Run not found.")
            else:
                st.write(run.get("meta", {}))
                st.download_button("Download baseline CSV (saved)", run["baseline_df"].to_csv(index=False), file_name=f"schedule_{sel}_baseline.csv", mime="text/csv", use_container_width=True)
                st.download_button("Download crashed CSV (saved)", run["crashed_df"].to_csv(index=False), file_name=f"schedule_{sel}_crashed.csv", mime="text/csv", use_container_width=True)

def saved_results_page() -> None:
    backend = render_sidebar("Saved Results")
    st.title("Saved Results")
    st.caption("Browse saved submittal checks and schedule runs. Downloads work for all backends.")

    st.markdown("## Submittal checks")
    try:
        checks = backend.list_submittal_checks()
    except Exception as e:
        st.warning(f"Could not list submittal checks: {e}")
        checks = pd.DataFrame()

    if checks is None or checks.empty:
        st.info("No submittal checks found.")
    else:
        st.dataframe(checks, use_container_width=True)
        cid = st.text_input("Load submittal check by id", value="")
        if st.button("Load submittal check"):
            payload = backend.load_submittal_check(cid.strip())
            if payload is None:
                st.error("Not found.")
            else:
                st.json(payload)
                st.download_button("Download JSON", data=json.dumps(payload, indent=2), file_name=f"submittal_{cid}.json", mime="application/json", use_container_width=True)

    st.markdown("---")
    st.markdown("## Schedule runs")
    try:
        runs = backend.list_schedule_runs()
    except Exception as e:
        st.warning(f"Could not list schedule runs: {e}")
        runs = pd.DataFrame()

    if runs is None or runs.empty:
        st.info("No schedule runs found.")
    else:
        st.dataframe(runs, use_container_width=True)
        rid = st.text_input("Load schedule run by run_id", value="")
        if st.button("Load schedule run"):
            run = backend.get_schedule_run(rid.strip())
            if not run:
                st.error("Not found.")
            else:
                st.write(run.get("meta", {}))
                st.download_button("Download baseline CSV", run["baseline_df"].to_csv(index=False), file_name=f"schedule_{rid}_baseline.csv", mime="text/csv", use_container_width=True)
                st.download_button("Download crashed CSV", run["crashed_df"].to_csv(index=False), file_name=f"schedule_{rid}_crashed.csv", mime="text/csv", use_container_width=True)



# =========================
# Missing pages (stubs)
# =========================

def rfi_manager_page() -> None:
    """Simple placeholder RFI Manager page.

    The storage backends already support RFIs in SQLite; cloud file backends can be extended later.
    """
    import streamlit as st
    st.title("RFI Manager")
    st.info("RFI Manager is not wired up in this build yet. Use it as a placeholder page for now.")

def aging_dashboard_page() -> None:
    """Simple placeholder Aging Dashboard page."""
    import streamlit as st
    st.title("Aging Dashboard")
    st.info("Aging Dashboard is not wired up in this build yet. Use it as a placeholder page for now.")

def settings_examples_page() -> None:
    """Settings & Examples page: provides sample downloads and quick notes."""
    import streamlit as st
    from pathlib import Path

    st.title("Settings & Examples")

    st.markdown("### Sample files (download)")
    base = Path(__file__).resolve().parent / "sample_data"
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
    st.markdown("### Notes")
    st.markdown(
        "- **Google Drive / OneDrive backends** store FieldFlow data as files in an app folder.\n"
        "- **Local (SQLite)** stores data in a local database file on the server."
    )
