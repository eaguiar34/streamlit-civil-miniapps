# 1_Submittal_Checker.py
# Submittal Checker with Hybrid Matching + Reviewer Policies + Highlights
# - Hybrid score: lexical (RapidFuzz + optional TF-IDF) + semantic (optional embeddings)
#   + keyword coverage − forbidden penalties + section boost
# - Reviewer keyword lists (must / nice-to-have / forbidden) with PASS gating
# - Highlights + reason panel for explainability
# - Robust file reading (PDF/DOCX/TXT/CSV), chunking, normalization

from __future__ import annotations

import io
import re
from typing import Iterable, List, Optional, Tuple

import streamlit as st
import pandas as pd

# ---- Optional deps (soft-imports) ----
try:
    from rapidfuzz import fuzz
except Exception as e:
    st.error("Missing dependency: rapidfuzz. Install with `pip install rapidfuzz`.")
    raise

try:
    from pypdf import PdfReader
    _HAS_PDF = True
except Exception:
    _HAS_PDF = False

try:
    import docx  # python-docx
    _HAS_DOCX = True
except Exception:
    _HAS_DOCX = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False

# sentence-transformers is optional; we cache the model if available
@st.cache_resource
def _get_embedder():
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception:
        return None


# ============================ File IO ============================

SUPPORTED = (".pdf", ".docx", ".txt", ".csv")

def _read_pdf(file: io.BytesIO) -> str:
    if not _HAS_PDF:
        return ""
    reader = PdfReader(file)
    out = []
    for page in reader.pages:
        try:
            out.append(page.extract_text() or "")
        except Exception:
            pass
    return "\n".join(out)

def _read_docx(file: io.BytesIO) -> str:
    if not _HAS_DOCX:
        return ""
    d = docx.Document(file)
    return "\n".join(p.text for p in d.paragraphs)

def _read_txt(file: io.BytesIO) -> str:
    return file.read().decode(errors="ignore")

def _read_csv(file: io.BytesIO) -> str:
    df = pd.read_csv(file)
    strings = []
    for col in df.columns:
        try:
            strings.extend(df[col].astype(str).tolist())
        except Exception:
            pass
    return "\n".join(strings)

def read_any(uploaded) -> str:
    name = getattr(uploaded, "name", "").lower()
    data = io.BytesIO(uploaded.read())
    if name.endswith(".pdf"):
        return _read_pdf(data)
    if name.endswith(".docx"):
        return _read_docx(data)
    if name.endswith(".txt"):
        return _read_txt(data)
    if name.endswith(".csv"):
        return _read_csv(data)
    raise ValueError(f"Unsupported file: {name} (supported: {', '.join(SUPPORTED)})")


# ============================ Text Processing ============================

def clean_text(t: str) -> str:
    t = re.sub(r"\r", "\n", t)
    t = re.sub(r"\n{2,}", "\n", t)
    return t.strip()

# Bullet/header detection
BULLET_RE = re.compile(r"^\s*(?:[-•*]|\d+[.)]|[A-Z]\s*[.)])\s+")
HEADER_RE = re.compile(r"^(?:PART|SECTION|DIVISION)\b", re.I)

def split_into_chunks(t: str, *, keep_headers: bool = True) -> List[str]:
    lines = [ln.strip() for ln in t.split("\n")]
    chunks, acc = [], []

    def flush():
        if acc:
            s = " ".join(acc).strip()
            if s:
                chunks.append(s)
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

    # Dedup tiny noise
    seen, out = set(), []
    for c in chunks:
        k = c.lower()
        if k not in seen:
            out.append(c)
            seen.add(k)
    return out

# Requirement hints to pull likely spec “requirements”
REQUIREMENT_HINTS = (
    "provide", "submit", "include", "certificates", "warranty",
    "shop drawings", "data sheets", "samples", "tests", "compliance",
)

def extract_spec_items(spec_chunks: Iterable[str]) -> List[str]:
    items = []
    for ch in spec_chunks:
        low = ch.lower()
        if any(k in low for k in REQUIREMENT_HINTS):
            items.append(ch)
    return items or list(spec_chunks)

# Normalization for “exact-like” and scoring
SECTION_HDR = re.compile(
    r"^(?P<label>(?:PART|SECTION|DIVISION|PRODUCT DATA|SHOP DRAWINGS|CERTIFICATES|WARRANTY|SAMPLES|TESTS?|CLOSEOUT|QUALITY).*)\\b",
    re.I,
)
BULLET_HDR  = re.compile(r"^\s*(?:[-•*]|\d+[.)]|[A-Z]\s*[.)])\s+")
PUNCT_FIXES = {"\u2018":"'","\u2019":"'","\u201C":"\"","\u201D":"\"","\u2013":"-","\u2014":"-","\u00A0":" "}

def simplify(s: str) -> str:
    for k, v in PUNCT_FIXES.items():
        s = s.replace(k, v)
    s = re.sub(r"(\w)-\n(\w)", r"\1\2", s)  # dehyphenate line wraps
    s = s.replace("\n", " ")
    s = BULLET_HDR.sub("", s)
    s = re.sub(r"\s+", " ", s)
    return s.lower().strip()

def label_sections(chunks: List[str]) -> List[str]:
    tags, cur = [], "General"
    for c in chunks:
        m = SECTION_HDR.match(c.strip())
        if m:
            cur = m.group("label").title()
        tags.append(cur)
    return tags


# ============================ Hybrid Scoring ============================

@st.cache_resource
def _build_tfidf(texts: List[str]):
    if not _HAS_SKLEARN:
        return None, None
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    X = vec.fit_transform(texts)
    return vec, X


def keyword_coverage(chunk_simpl: str, must: List[str], nice: List[str], forbid: List[str]):
    found_must = [k for k in must if k in chunk_simpl]
    miss_must  = [k for k in must if k not in chunk_simpl]
    found_nice = [k for k in nice if k in chunk_simpl]
    hit_forbid = [k for k in forbid if k in chunk_simpl]
    cov = 0.0
    if must or nice:
        cov_m = len(found_must)/max(1, len(must)) if must else 0.0
        cov_n = len(found_nice)/max(1, len(nice)) if nice else 0.0
        cov = 0.7*cov_m + 0.3*cov_n
    return cov, found_must, miss_must, found_nice, hit_forbid


def highlight_html(chunk: str, terms: List[str]) -> str:
    out = chunk
    for t in sorted(set([t for t in terms if t]), key=len, reverse=True):
        try:
            out = re.sub(f"(?i)({re.escape(t)})", r"<mark>\1</mark>", out)
        except re.error:
            pass
    return out


def guess_section_from_text(s_simpl: str) -> str:
    if "product data" in s_simpl:
        return "Product Data"
    if "shop drawing" in s_simpl or "shop drawings" in s_simpl:
        return "Shop Drawings"
    if "certificate" in s_simpl:
        return "Certificates"
    if "warranty" in s_simpl:
        return "Warranty"
    if "sample" in s_simpl:
        return "Samples"
    if "test" in s_simpl:
        return "Tests"
    if "o&m" in s_simpl or "operation and maintenance" in s_simpl:
        return "Closeout"
    return "General"


# ============================ UI ============================

st.set_page_config(page_title="Submittal Checker", layout="wide")
st.title("Submittal Checker")

left, right = st.columns(2)

with left:
    st.subheader("Spec Source")
    spec_file = st.file_uploader("Upload spec (PDF/DOCX/TXT/CSV)", type=["pdf", "docx", "txt", "csv"], key="spec_file")
    spec_text_area = st.text_area("Or paste spec text:", height=220, placeholder="Paste specification clauses here…", key="spec_text")

with right:
    st.subheader("Submittal Source")
    sub_file = st.file_uploader("Upload submittal (PDF/DOCX/TXT/CSV)", type=["pdf", "docx", "txt", "csv"], key="sub_file")
    sub_text_area = st.text_area("Or paste submittal text:", height=220, placeholder="Paste submittal content or summary…", key="sub_text")

threshold = st.slider("Hybrid PASS threshold (0–100)", 0, 100, 85)
with st.expander("Reviewer policy & keywords", expanded=False):
    colA, colB = st.columns(2)
    with colA:
        must_text = st.text_area("Must-have terms (comma-separated)", "data sheet,warranty,certificate")
        nice_text = st.text_area("Nice-to-have terms", "shop drawing,test report,O&M manual")
    with colB:
        forbid_text = st.text_area("Forbidden phrases", "by others,not provided,N/A")
        st.caption("If any forbidden phrase appears in the best match, the item cannot PASS.")

    st.markdown("**Scoring weights**")
    α = st.slider("Lexical weight", 0.0, 1.0, 0.45, 0.05)
    β = st.slider("Semantic weight", 0.0, 1.0, 0.35, 0.05)
    γ = st.slider("Coverage weight", 0.0, 1.0, 0.20, 0.05)
    δ = st.slider("Forbidden penalty per hit", 0.0, 1.0, 0.20, 0.05)
    ε = st.slider("Section boost (match)", 0.0, 0.5, 0.05, 0.01)

must = [t.strip().lower() for t in must_text.split(",") if t.strip()]
nice = [t.strip().lower() for t in nice_text.split(",") if t.strip()]
forbid = [t.strip().lower() for t in forbid_text.split(",") if t.strip()]

st.markdown("---")
tab1, tab2 = st.tabs(["Preview: Spec", "Preview: Submittal"]) 

# ============================ Analyze ============================

if st.button("Analyze", type="primary"):
    try:
        # Gather text
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

        # Previews
        with tab1:
            st.code((spec_text[:2000] + ("…" if len(spec_text) > 2000 else "")) or "—")
        with tab2:
            st.code((sub_text[:2000] + ("…" if len(sub_text) > 2000 else "")) or "—")

        # Chunking / sections
        spec_chunks = split_into_chunks(spec_text)
        sub_chunks = split_into_chunks(sub_text, keep_headers=True)
        sub_sections = label_sections(sub_chunks)

        # Which spec lines to evaluate
        spec_items = extract_spec_items(spec_chunks)

        # Normalized versions
        sub_simpl = [simplify(c) for c in sub_chunks]

        # Optional TF-IDF
        if _HAS_SKLEARN:
            tfidf, X_sub = _build_tfidf(sub_simpl)
        else:
            tfidf, X_sub = None, None

        # Optional embeddings (cached)
        embedder = _get_embedder()
        if embedder is not None:
            E_sub = embedder.encode(sub_simpl, normalize_embeddings=True)
        else:
            E_sub = None

        rows = []
        hybrid_thresh = threshold / 100.0

        for item in spec_items:
            s_raw = item.strip()
            s_simpl = simplify(s_raw)

            # Lexical RapidFuzz across all chunks (0..100 -> 0..1)
            rf_scores = [fuzz.token_set_ratio(s_simpl, t) / 100.0 for t in sub_simpl]

            # TF-IDF cosine (0..1)
            if tfidf is not None and X_sub is not None:
                q = tfidf.transform([s_simpl])
                tfidf_scores = cosine_similarity(q, X_sub).ravel()
            else:
                tfidf_scores = [0.0] * len(sub_simpl)

            # Combine lexical channels
            comb_lex = [
                (rf_scores[i] + tfidf_scores[i]) / (2.0 if tfidf is not None else 1.0)
                for i in range(len(sub_simpl))
            ]

            # Semantic: only compute for top-K lexical to save time
            sem_scores = [0.0] * len(sub_simpl)
            if embedder is not None and E_sub is not None:
                K = min(50, len(sub_simpl))
                top_idx = sorted(range(len(sub_simpl)), key=lambda i: comb_lex[i], reverse=True)[:K]
                E_q = embedder.encode([s_simpl], normalize_embeddings=True)[0]
                for i in top_idx:
                    sem_scores[i] = float(E_q @ E_sub[i])

            # Section bonus
            spec_section = guess_section_from_text(s_simpl)
            sec_bonus = [ε if sub_sections[i].lower().startswith(spec_section.split()[0].lower()) else 0.0
                         for i in range(len(sub_simpl))]

            # Coverage + forbidden
            cov_scores, forb_hits, cov_meta = [], [], []
            for i in range(len(sub_simpl)):
                cov, found_must, miss_must, found_nice, hit_forbid = keyword_coverage(sub_simpl[i], must, nice, forbid)
                cov_scores.append(cov)
                forb_hits.append(len(hit_forbid))
                cov_meta.append((found_must, miss_must, found_nice, hit_forbid))

            # Hybrid score
            hybrid = [
                α*comb_lex[i] + β*sem_scores[i] + γ*cov_scores[i] + sec_bonus[i] - δ*forb_hits[i]
                for i in range(len(sub_simpl))
            ]
            j = int(max(range(len(hybrid)), key=lambda i: hybrid[i]))
            best_chunk = sub_chunks[j]
            best_simpl = sub_simpl[j]

            # Decision gates
            exact_like = s_simpl in best_simpl  # normalized containment
            has_all_must = (len(cov_meta[j][1]) == 0) if must else True
            no_forbidden = (len(cov_meta[j][3]) == 0)
            decision = "PASS" if (exact_like or (hybrid[j] >= hybrid_thresh)) and has_all_must and no_forbidden else "REVIEW"

            # Row with explainability
            rows.append({
                "Concept": re.sub(r"\([^)]*\)", "", " ".join(re.findall(r"[A-Za-z0-9-/]+", s_raw))[:80]),
                "Spec Item": s_raw,
                "Best Submittal Match": best_chunk,
                "Hybrid Score": round(hybrid[j], 3),
                "Lexical": round(comb_lex[j], 3),
                "Semantic": round(sem_scores[j], 3),
                "Coverage": round(cov_scores[j], 3),
                "Must Missing": ", ".join(cov_meta[j][1]) or "—",
                "Forbidden Hits": ", ".join(cov_meta[j][3]) or "—",
                "Section (spec → submittal)": f"{spec_section} → {sub_sections[j]}",
                "Exact-like?": "Yes" if exact_like else "No",
                "Decision": decision,
                "_best_chunk_html": highlight_html(best_chunk, cov_meta[j][0] + cov_meta[j][2]),
            })

        df = pd.DataFrame(rows)

        # Summary metrics
        pass_rate = float((df["Decision"] == "PASS").mean()) if not df.empty else 0.0
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Coverage (PASS)", f"{pass_rate:.0%}")
        m2.metric("Threshold", threshold)
        m3.metric("Exact-like hits", int((df["Exact-like?"] == "Yes").sum()))
        m4.metric("Avg Hybrid Score", f"{df['Hybrid Score'].mean():.2f}" if not df.empty else "—")

        st.markdown("---")
        st.subheader("Results")
        st.dataframe(
            df.drop(columns=["_best_chunk_html"]),
            hide_index=True,
            use_container_width=True
        )

        st.markdown("---")
        st.subheader("Best match with highlights & reasons")
        for _, r in df.iterrows():
            st.markdown(
                f"**{r['Concept']}** — **Decision:** {r['Decision']}  \n"
                f"*Hybrid:* {r['Hybrid Score']:.2f} | *Lex:* {r['Lexical']:.2f} | *Sem:* {r['Semantic']:.2f} | "
                f"*Coverage:* {r['Coverage']:.2f} | *Missing must:* {r['Must Missing']} | *Forbidden:* {r['Forbidden Hits']}  \n"
                f"*Section:* {r['Section (spec → submittal)']} | *Exact-like:* {r['Exact-like?']}",
                help="Why this passed/failed, with score components."
            )
            st.markdown(r["_best_chunk_html"], unsafe_allow_html=True)
            st.markdown("---")

        # Download
        st.download_button(
            "Download CSV",
            data=df.drop(columns=["_best_chunk_html"]).to_csv(index=False).encode(),
            file_name="submittal_checker_results.csv",
            mime="text/csv",
        )

    except Exception as e:
        st.exception(e)
