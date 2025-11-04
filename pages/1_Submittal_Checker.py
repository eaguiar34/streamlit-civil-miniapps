import re
from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd
import unicodedata
import numpy as np
import streamlit as st
from rapidfuzz import fuzz
from security import read_text_any, require_passcode

st.set_page_config(page_title="Submittal Checker", page_icon="📝", layout="wide")
require_passcode()

st.title("Submittal Checker")
st.caption("Paste text or upload PDF/DOCX/CSV files for the spec and the submittal.")

REQ_CUES = [
    "shall","include","provide","submit","furnish","demonstrate","comply",
    "in accordance","per","minimum","at least","conform","certify","warranty",
]

# common filler words (we don't treat these as "essential" requirement tokens)
FILLER = set("""
a an the and or if of to for in on by with from as is are be been it its this that which such per
shall include provide submit furnish demonstrate comply accordance minimum least conform certify warranty
product products data documentation statement statements info information
""".split())

SYNONYM_MAP = {
    "datasheet": "data sheet",
    "datasheets": "data sheet",
    "shopdrawing": "shop drawing",
    "shopdrawings": "shop drawing",
    "certificate": "certification",
    "certificates": "certification",
    "guarantee": "warranty",
    "guarantees": "warranty",
}

def normalize_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    # Unicode fold, lowercase
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    # strip possessive 's and punctuation (keep letters/digits/spaces)
    s = s.lower()
    s = s.replace("'s", " ")
    s = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in s)
    # collapse spaces
    s = " ".join(s.split())
    # simple synonym normalization
    for k, v in SYNONYM_MAP.items():
        s = s.replace(k, v)
    return s

def tokens(s: str) -> list[str]:
    return [t for t in normalize_text(s).split() if t and t not in FILLER]

def essential_token_overlap(req: str, sub: str) -> float:
    rt = set(tokens(req))
    st = set(tokens(sub))
    if not rt:
        return 0.0
    return len(rt & st) / max(1, len(rt))

def best_match(requirement: str, submittal_chunks: list[str]) -> tuple[str, int, float]:
    """
    Returns (best_chunk, fuzzy_score_0to100, concept_overlap_0to1)
    fuzzy score is the max of several fuzzy measures on normalized text
    concept overlap measures essential-token coverage
    """
    if not submittal_chunks:
        return "", 0, 0.0
    req_n = normalize_text(requirement)
    scores = []
    for ch in submittal_chunks:
        ch_n = normalize_text(ch)
        # multiple views of similarity
        s1 = fuzz.token_set_ratio(req_n, ch_n)
        s2 = fuzz.partial_ratio(req_n, ch_n)
        s3 = fuzz.ratio(req_n, ch_n)
        scores.append(max(s1, s2, s3))
    idx = int(np.argmax(scores))
    best = submittal_chunks[idx]
    concept = essential_token_overlap(requirement, best)
    return best, int(scores[idx]), float(concept)

def split_lines(text: str) -> list[str]:
    return [ln.strip() for ln in text.splitlines() if ln.strip()]

def slice_requirement_candidates(lines: list[str]) -> list[str]:
    cands = []
    for ln in lines:
        ll = ln.lower()
        if any(cue in ll for cue in REQ_CUES):
            cands.append(ln); continue
        if re.match(r"^\s*(?:\d+\.|[a-zA-Z]\.|[-•*])\s+", ln):
            cands.append(ln)
    # merge short carry-over lines
    merged = []
    for ln in cands:
        if merged and (len(ln) < 40) and not ln.endswith((".", ";", ":")):
            merged[-1] = merged[-1] + " " + ln
        else:
            merged.append(ln)
    return merged

def best_match(requirement: str, submittal_chunks: List[str]) -> Tuple[str, int]:
    if not submittal_chunks:
        return "", 0
    scores = [fuzz.token_set_ratio(requirement, ch) for ch in submittal_chunks]
    idx = int(np.argmax(scores))
    return submittal_chunks[idx], int(scores[idx])

left, right = st.columns(2, gap="large")

with left:
    st.subheader("Spec Source")
    spec_upload = st.file_uploader("Upload spec (PDF/DOCX/TXT/CSV)", type=["pdf", "docx", "txt", "csv"], key="spec_up")
    spec_text_default = (
        "PART 1 - GENERAL\n"
        "1.02 SUBMITTALS\n"
        "A. Product Data: Provide manufacturer's data sheets.\n"
        "B. Shop Drawings: Submit coordinated drawings.\n"
        "C. Certificates: Provide compliance certifications.\n"
        "D. Warranty: Minimum one year warranty.\n"
    )
    spec_text_manual = st.text_area("Or paste spec text:", height=240, value=spec_text_default)
    spec_text = read_text_any(spec_upload) or spec_text_manual
    if spec_upload is not None:
        with st.expander("Preview extracted spec text", expanded=False):
            st.write(spec_text[:2000] + ("..." if len(spec_text) > 2000 else ""))

with right:
    st.subheader("Submittal Source")
    sub_upload = st.file_uploader("Upload submittal (PDF/DOCX/TXT/CSV)", type=["pdf", "docx", "txt", "csv"], key="sub_up")
    sub_text_default = (
        "We are submitting product data for review.\n"
        "Included: manufacturer data sheets and a warranty statement.\n"
    )
    sub_text_manual = st.text_area("Or paste submittal text:", height=240, value=sub_text_default)
    submittal_text = read_text_any(sub_upload) or sub_text_manual
    if sub_upload is not None:
        with st.expander("Preview extracted submittal text", expanded=False):
            st.write(submittal_text[:2000] + ("..." if len(submittal_text) > 2000 else ""))

threshold = st.slider("Match threshold (0-100)", min_value=50, max_value=100, value=78, help="Lower = more forgiving matches")
run = st.button("Analyze")

if run:
    if not spec_text.strip() or not submittal_text.strip():
        st.error("Need both spec and submittal text (upload or paste).")
        st.stop()

    spec_lines = split_lines(spec_text)
    reqs = slice_requirement_candidates(spec_lines)

    sub_lines = split_lines(submittal_text)
    sub_chunks: List[str] = sub_lines[:]
    for i in range(len(sub_lines) - 1):
        sub_chunks.append(sub_lines[i] + " " + sub_lines[i + 1])

    results: list[MatchResult] = []
for r in reqs:
    m, fuzzy_score, concept = best_match(r, sub_chunks)
    # Decision: accept if fuzzy high OR concept coverage high
    if fuzzy_score >= threshold or concept >= 0.75:
        status = "Found"
    elif fuzzy_score >= max(50, threshold - 20) or concept >= 0.40:
        status = "Weak"
    else:
        status = "Missing"
    results.append(MatchResult(r, m, fuzzy_score, status))

    df = pd.DataFrame([
        {"Requirement": r.requirement, "Best Match In Submittal": r.matched_text, "Score": r.score, "Status": r.status}
        for r in results
    ])

    found = int((df["Status"] == "Found").sum())
    weak = int((df["Status"] == "Weak").sum())
    missing = int((df["Status"] == "Missing").sum())
    coverage = int(round(100 * found / max(1, len(df))))

    st.success(f"Coverage: {coverage}%  |  Found: {found}  |  Weak: {weak}  |  Missing: {missing}")
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.download_button(
        label="Download results as CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="submittal_check_results.csv",
        mime="text/csv",
    )
else:
    st.markdown("Click Analyze to run the comparison. Adjust the threshold if matches feel too strict/loose.")
