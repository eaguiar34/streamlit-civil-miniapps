# 1_Submittal_Checker.py
# Rewritten, robust version that avoids the NameError and simplifies matching logic.
# Drop this file into your Streamlit app. Requires: streamlit, rapidfuzz, pypdf, python-docx, pandas
#
# pip install streamlit rapidfuzz pypdf python-docx pandas

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import streamlit as st

# Optional deps: we'll soft-import to give clearer errors if missing
try:
    from rapidfuzz import fuzz
except Exception as e:  # pragma: no cover
    st.error(
        "Missing dependency: rapidfuzz. Run `pip install rapidfuzz`.\n" f"Details: {e}"
    )
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


def _read_pdf(file: io.BytesIO) -> str:
    reader = PdfReader(file)
    text_parts: List[str] = []
    for page in reader.pages:
        try:
            text_parts.append(page.extract_text() or "")
        except Exception:
            # Fallback if a page fails; keep going
            pass
    return "\n".join(text_parts)


def _read_docx(file: io.BytesIO) -> str:
    d = docx.Document(file)
    return "\n".join(p.text for p in d.paragraphs)


def _read_txt(file: io.BytesIO) -> str:
    return file.read().decode(errors="ignore")


def _read_csv(file: io.BytesIO) -> str:
    df = pd.read_csv(file)
    # Join all textual cells into one blob
    strings = []
    for col in df.columns:
        try:
            strings.extend(df[col].astype(str).tolist())
        except Exception:
            pass
    return "\n".join(strings)


def read_any(file) -> str:
    """Read text from an uploaded file of supported type."""
    name = (getattr(file, "name", "").lower())
    data = io.BytesIO(file.read())
    if name.endswith(".pdf"):
        return _read_pdf(data)
    if name.endswith(".docx"):
        return _read_docx(data)
    if name.endswith(".txt"):
        return _read_txt(data)
    if name.endswith(".csv"):
        return _read_csv(data)
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

    # Dedup small noise
    seen = set()
    deduped: List[str] = []
    for c in chunks:
        key = c.lower()
        if key not in seen:
            deduped.append(c)
            seen.add(key)
    return deduped


# Lightweight heuristics to pull "requirements" from specs
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
    # Fallback: if no items identified, just return the chunks
    return items or list(spec_chunks)


def derive_concept(s: str) -> str:
    # Pull a short noun-phrase-ish concept for display
    s = re.sub(r"\([^)]*\)", "", s)
    # try to keep up to ~10 words
    words = re.findall(r"[A-Za-z0-9-/]+", s)
    return " ".join(words[:10])


# ---------------------------- Matching ---------------------------- #

def best_match(spec_item: str, sub_chunks: List[str]) -> Tuple[Optional[str], float, bool]:
    """Return (best_chunk, fuzzy_score, exact_phrase) for a spec item against submittal."""
    if not sub_chunks:
        return None, 0.0, False

    best_score = -1.0
    best_chunk: Optional[str] = None
    # Exact phrase check (case-insensitive)
    exact = False

    norm_item = spec_item.strip().lower()
    for ch in sub_chunks:
        ch_norm = ch.lower()
        if not exact and norm_item in ch_norm:
            exact = True
        # token_set_ratio is robust to ordering/duplication
        score = float(fuzz.token_set_ratio(norm_item, ch_norm))
        if score > best_score:
            best_score = score
            best_chunk = ch
    return best_chunk, best_score, exact


def judge(fuzzy_score: float, exact: bool, threshold: int) -> str:
    return "PASS" if exact or fuzzy_score >= threshold else "REVIEW"


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
        "Or paste spec text:",
        height=220,
        placeholder="Paste specification clauses here…",
        key="spec_text",
    )

with right:
    st.subheader("Submittal Source")
    sub_file = st.file_uploader(
        "Upload submittal (PDF/DOCX/TXT/CSV)", type=["pdf", "docx", "txt", "csv"], key="sub_file"
    )
    sub_text_area = st.text_area(
        "Or paste submittal text:",
        height=220,
        placeholder="Paste submittal content or summary…",
        key="sub_text",
    )

threshold = st.slider("Match threshold (0–100)", min_value=0, max_value=100, value=78)

if st.button("Analyze", type="primary"):
    # -------- Gather & clean text -------- #
    try:
        spec_text = ""
        if spec_file is not None:
            spec_text = read_any(spec_file)
        spec_text = (spec_text + "\n" + spec_text_area).strip() if spec_text_area else spec_text

        sub_text = ""
        if sub_file is not None:
            sub_text = read_any(sub_file)
        sub_text = (sub_text + "\n" + sub_text_area).strip() if sub_text_area else sub_text

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

        # -------- Summaries -------- #
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

        # Unmatched view
        st.markdown("---")
        st.subheader("Items Needing Review")
        needs_review = df[df["Decision"] == "REVIEW"]
        if len(needs_review) == 0:
            st.info("Nothing to review. All items cleared the threshold or matched exactly.")
        else:
            st.dataframe(needs_review, use_container_width=True, hide_index=True)

        # Download
        csv = df.to_csv(index=False).encode()
        st.download_button(
            "Download CSV",
            data=csv,
            file_name="submittal_checker_results.csv",
            mime="text/csv",
        )

    except Exception as e:
        st.exception(e)


# ----------------------------
# Optional smoke tests (run outside Streamlit)
# ----------------------------
# These tests don't run during normal Streamlit usage. To run them:
#   RUN_SELF_TESTS=1 python 1_Submittal_Checker.py

if __name__ == "__main__":
    import os as _os
    if _os.environ.get("RUN_SELF_TESTS") == "1":
        # Test clean_text + split_into_chunks
        raw = """
SECTION 01 33 00 Submittal Procedures
  - Provide product data
  - Submit samples

PART 2
  Materials shall comply with ASTM standards
"""
        ct = clean_text(raw)
        chunks = split_into_chunks(ct)
        assert any("Provide product data" in c for c in chunks), "split_into_chunks missed bullet"
        assert any("Submittal Procedures" in c for c in chunks), "split_into_chunks missed header"

        # Test requirement extraction
        items = extract_spec_items(chunks)
        assert len(items) >= 2, "extract_spec_items should find at least two hints"

        # Test matching
        sub_text = "Samples and product data are included per Section 01 33 00."
        best, score, exact = best_match(items[0], split_into_chunks(sub_text, keep_headers=False))
        assert best is not None and score > 50, "best_match score too low"

        print("Submittal Checker smoke tests passed.")
