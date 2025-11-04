# 1_Submittal_Checker.py
# Rewritten with file preview panes for uploaded Spec/Submittal files.
# Requirements: streamlit, rapidfuzz, pypdf, python-docx, pandas

from __future__ import annotations

import io
import re
import base64
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import streamlit as st
import streamlit.components.v1 as components

# Optional deps with clear errors
try:
    from rapidfuzz import fuzz
except Exception as e:  # pragma: no cover
    st.error("Missing dependency: rapidfuzz. Run `pip install rapidfuzz`.\n" f"Details: {e}")
    raise

try:
    from pypdf import PdfReader
except Exception as e:  # pragma: no cover
    st.error("Missing dependency: pypdf. Run `pip install pypdf`.\n" f"Details: {e}")
    raise

try:
    import docx  # python-docx
except Exception as e:  # pragma: no cover
    st.error("Missing dependency: python-docx. Run `pip install python-docx`.\n" f"Details: {e}")
    raise

import pandas as pd


# ---------------------------- Models ---------------------------- #
@dataclass
class MatchResult:
    spec_item: str
    best_sub_chunk: Optional[str]
    fuzzy_score: float
    exact_phrase: bool
    concept: Optional[str]
    decision: str  # "PASS" | "REVIEW"


# ---------------------------- File IO ---------------------------- #
SUPPORTED_EXT = (".pdf", ".docx", ".txt", ".csv")


def _read_pdf(raw: bytes) -> str:
    reader = PdfReader(io.BytesIO(raw))
    text_parts: List[str] = []
    for page in reader.pages:
        try:
            text_parts.append(page.extract_text() or "")
        except Exception:
            pass
    return "\n".join(text_parts)


def _read_docx(raw: bytes) -> str:
    d = docx.Document(io.BytesIO(raw))
    return "\n".join(p.text for p in d.paragraphs)


def _read_txt(raw: bytes) -> str:
    return raw.decode(errors="ignore")


def _read_csv(raw: bytes) -> str:
    df = pd.read_csv(io.BytesIO(raw))
    strings = []
    for col in df.columns:
        try:
            strings.extend(df[col].astype(str).tolist())
        except Exception:
            pass
    return "\n".join(strings)


def read_any(name: str, raw: bytes) -> str:
    """Read text from an uploaded file of supported type using raw bytes."""
    lname = name.lower()
    if lname.endswith(".pdf"):
        return _read_pdf(raw)
    if lname.endswith(".docx"):
        return _read_docx(raw)
    if lname.endswith(".txt"):
        return _read_txt(raw)
    if lname.endswith(".csv"):
        return _read_csv(raw)
    raise ValueError(
        f"Unsupported file type for '{name}'. Supported: {', '.join(SUPPORTED_EXT)}"
    )


# ---------------------------- Text Processing ---------------------------- #
BULLET_RE = re.compile(r"^\s*(?:[-•*]|\d+[.)]|[A-Z]\.|[A-Z]\))\s+")
HEADER_RE = re.compile(r"^(?:PART|SECTION|DIVISION)\b", re.I)


def clean_text(t: str) -> str:
    t = re.sub(r"\r", "\n", t)
    t = re.sub(r"\n{2,}", "\n", t)
    return t.strip()


def split_into_chunks(t: str, *, keep_headers: bool = True) -> List[str]:
    """Split text into logical lines / bullets. Keeps section headers optionally."""
    lines = [ln.strip() for ln in t.split("\n")]
    chunks: List[str] = []
    acc: List[str] = []

    def flush():
        if acc:
            joined = " ".join(acc).strip()
            if joined:
                chunks.append(joined)
            acc.clear()

    for ln in lines:
        if not ln:
            flush()
            continue
        if BULLET_RE.search(ln) or (keep_headers and HEADER_RE.search(ln)):
            flush()
            acc.append(ln)
            flush()
        else:
            acc.append(ln)
    flush()

    seen = set()
    deduped: List[str] = []
    for c in chunks:
        key = c.lower()
        if key not in seen:
            deduped.append(c)
            seen.add(key)
    return deduped


REQUIREMENT_HINTS = (
    "provide",
    "submit",
    "include",
    "certificates",
    "warranty",
    "shop drawings",
    "data sheets",
    "samples",
    "tests",
    "compliance",
    "standards",
)


def extract_spec_items(spec_chunks: Iterable[str]) -> List[str]:
    items: List[str] = []
    for ch in spec_chunks:
        low = ch.lower()
        if any(k in low for k in REQUIREMENT_HINTS):
            items.append(ch)
    return items or list(spec_chunks)


def derive_concept(s: str) -> str:
    s = re.sub(r"\([^)]*\)", "", s)
    words = re.findall(r"[A-Za-z0-9-/]+", s)
    return " ".join(words[:10])


# ---------------------------- Matching ---------------------------- #

def best_match(spec_item: str, sub_chunks: List[str]) -> Tuple[Optional[str], float, bool]:
    if not sub_chunks:
        return None, 0.0, False

    best_score = -1.0
    best_chunk: Optional[str] = None
    exact = False

    norm_item = spec_item.strip().lower()
    for ch in sub_chunks:
        ch_norm = ch.lower()
        if not exact and norm_item in ch_norm:
            exact = True
        score = float(fuzz.token_set_ratio(norm_item, ch_norm))
        if score > best_score:
            best_score = score
            best_chunk = ch
    return best_chunk, best_score, exact


def judge(fuzzy_score: float, exact: bool, threshold: int) -> str:
    return "PASS" if exact or fuzzy_score >= threshold else "REVIEW"


# ---------------------------- Preview Helpers ---------------------------- #

def _embed_pdf(raw: bytes, height: int = 500):
    b64 = base64.b64encode(raw).decode()
    src = f"data:application/pdf;base64,{b64}"
    components.html(
        f'<iframe src="{src}" style="width:100%;height:{height}px;border:none;" />',
        height=height,
        scrolling=True,
    )


def preview_file(name: str, raw: Optional[bytes], extracted_text: str):
    """Render a preview UI for the uploaded file + extracted text."""
    if raw is None and not extracted_text:
        st.info("No file or text to preview.")
        return

    lname = (name or "").lower()
    if raw is not None:
        st.caption(f"Uploaded file: **{name}** · {len(raw):,} bytes")

    # File-type specific preview
    if raw is not None and lname.endswith(".pdf"):
        try:
            _embed_pdf(raw, height=520)
        except Exception:
            st.warning("Inline PDF preview failed; showing extracted text instead.")
    elif raw is not None and lname.endswith(".csv"):
        try:
            df_prev = pd.read_csv(io.BytesIO(raw))
            st.dataframe(df_prev.head(200), use_container_width=True)
            st.caption(f"Rows: {len(df_prev):,} · Columns: {len(df_prev.columns)}")
        except Exception as e:
            st.warning(f"CSV preview failed: {e}")
    elif raw is not None and lname.endswith(".docx"):
        st.caption("DOCX preview shows extracted paragraphs (not layout-faithful).")

    # Extracted text preview (common fallback)
    if extracted_text:
        show_full = st.toggle("Show full extracted text", value=False)
        if show_full:
            st.text_area("Extracted text", extracted_text, height=300)
        else:
            snippet = extracted_text[:2000]
            st.text_area("Extracted text (first 2,000 chars)", snippet, height=220)


# ---------------------------- UI ---------------------------- #
st.set_page_config(page_title="Submittal Checker", layout="wide")
st.title("Submittal Checker")

left, right = st.columns(2)

with left:
    st.subheader("Spec Source")
    spec_file = st.file_uploader(
        "Upload spec (PDF/DOCX/TXT/CSV)", type=["pdf", "docx", "txt", "csv"], key="spec_file"
    )
    spec_text_area = st.text_area(
        "Or paste spec text:", height=220, placeholder="Paste specification clauses here…", key="spec_text"
    )

with right:
    st.subheader("Submittal Source")
    sub_file = st.file_uploader(
        "Upload submittal (PDF/DOCX/TXT/CSV)", type=["pdf", "docx", "txt", "csv"], key="sub_file"
    )
    sub_text_area = st.text_area(
        "Or paste submittal text:", height=220, placeholder="Paste submittal content or summary…", key="sub_text"
    )

threshold = st.slider("Match threshold (0–100)", min_value=0, max_value=100, value=78)

# -------- Build preview tabs -------- #
st.markdown("---")
prev_spec, prev_sub = st.tabs(["Preview: Spec", "Preview: Submittal"])

with prev_spec:
    spec_bytes = spec_file.getvalue() if spec_file is not None else None
    spec_text_from_file = ""
    if spec_bytes is not None and spec_file is not None:
        try:
            spec_text_from_file = read_any(spec_file.name, spec_bytes)
        except Exception as e:
            st.warning(f"Failed to read spec file: {e}")
    combined_spec_text = (spec_text_from_file + "\n" + spec_text_area).strip() if spec_text_area else spec_text_from_file
    preview_file(spec_file.name if spec_file else "(no file)", spec_bytes, combined_spec_text)

with prev_sub:
    sub_bytes = sub_file.getvalue() if sub_file is not None else None
    sub_text_from_file = ""
    if sub_bytes is not None and sub_file is not None:
        try:
            sub_text_from_file = read_any(sub_file.name, sub_bytes)
        except Exception as e:
            st.warning(f"Failed to read submittal file: {e}")
    combined_sub_text = (sub_text_from_file + "\n" + sub_text_area).strip() if sub_text_area else sub_text_from_file
    preview_file(sub_file.name if sub_file else "(no file)", sub_bytes, combined_sub_text)

# -------- Analysis button -------- #
if st.button("Analyze", type="primary"):
    try:
        # Use the combined preview texts for analysis
        spec_text = combined_spec_text
        sub_text = combined_sub_text

        if not spec_text:
            st.warning("Please provide spec text or upload a spec file.")
            st.stop()
        if not sub_text:
            st.warning("Please provide submittal text or upload a submittal file.")
            st.stop()

        spec_text = clean_text(spec_text)
        sub_text = clean_text(sub_text)

        spec_chunks = split_into_chunks(spec_text)
        sub_chunks = split_into_chunks(sub_text, keep_headers=False)

        spec_items = extract_spec_items(spec_chunks)

        results: List[MatchResult] = []
        for item in spec_items:
            best, score, exact = best_match(item, sub_chunks)
            decision = judge(score, exact, threshold)
            results.append(
                MatchResult(
                    spec_item=item,
                    best_sub_chunk=best,
                    fuzzy_score=score,
                    exact_phrase=exact,
                    concept=derive_concept(item),
                    decision=decision,
                )
            )

        df = pd.DataFrame(
            [
                {
                    "Concept": r.concept,
                    "Spec Item": r.spec_item,
                    "Best Submittal Match": r.best_sub_chunk or "",
                    "Fuzzy Score": round(r.fuzzy_score, 1),
                    "Exact Phrase?": "Yes" if r.exact_phrase else "No",
                    "Decision": r.decision,
                }
                for r in results
            ]
        )

        pass_rate = sum(r.decision == "PASS" for r in results) / max(len(results), 1)

        st.success(f"Analyzed {len(results)} requirement lines.")
        m1, m2, m3 = st.columns(3)
        m1.metric("Coverage (PASS)", f"{pass_rate:.0%}")
        m2.metric("Threshold", threshold)
        m3.metric("Exact phrase hits", sum(r.exact_phrase for r in results))

        st.markdown("---")
        st.subheader("Results")
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.subheader("Items Needing Review")
        needs_review = df[df["Decision"] == "REVIEW"]
        if len(needs_review) == 0:
            st.info("Nothing to review. All items cleared the threshold or matched exactly.")
        else:
            st.dataframe(needs_review, use_container_width=True, hide_index=True)

        csv = df.to_csv(index=False).encode()
        st.download_button(
            "Download CSV",
            data=csv,
            file_name="submittal_checker_results.csv",
            mime="text/csv",
        )

    except Exception as e:
        st.exception(e)
