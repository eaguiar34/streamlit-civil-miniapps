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
except Exception as e: # pragma: no cover
st.error(
"Missing dependency: rapidfuzz. Run `pip install rapidfuzz`.\n" f"Details: {e}"
)
raise


try:
from pypdf import PdfReader
except Exception as e: # pragma: no cover
st.error("Missing dependency: pypdf. Run `pip install pypdf`.\n" f"Details: {e}")
raise


try:
import docx # python-docx
except Exception as e: # pragma: no cover
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
decision: str # "PASS" | "REVIEW"




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
print("Submittal Checker smoke tests passed.")
