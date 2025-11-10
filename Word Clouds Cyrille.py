########################################
# CalWORKs CSA Policy Analysis Pipeline
# --------------------------------------
# Features:
# - Robust PDF extraction (pdfplumber → PyPDF2 fallback)
# - Preserves original stopwords & KEEP_TERMS
# - Defensive word statistics (no KeyError on empty input)
# - Avoids shadowing of pathlib.Path
# - Clear debug logging
# - WordCloud visualizations for TOP-20 terms
# - Sentence-level and word-level sentiment analysis
# - NEW: Computes sentence-level sentiment per Top-20 word per county
########################################

import os
import re
import zipfile
from pathlib import Path as _Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import nltk
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import unicodedata
from nltk.stem import WordNetLemmatizer

# -------------------------------
# Quiet noisy PDF warnings (pdfminer/ghostscript style)
# -------------------------------
import logging
for noisy in (
    "pdfminer",
    "pdfminer.pdfinterp",
    "pdfminer.cmapdb",
    "pdfminer.psparser",
    "pdfminer.pdfpage"
):
    logging.getLogger(noisy).setLevel(logging.ERROR)

# -------------------------------
# Optional: DOCX readers (kept for completeness)
# -------------------------------
import docx2txt
from docx import Document

# -------------------------------
# Optional: gensim (Word2Vec) support
# -------------------------------
W2V_OK = True
try:
    from gensim.models import Word2Vec, KeyedVectors
    from gensim.downloader import load as gensim_load
except Exception:
    W2V_OK = False

# -------------------------------
# Optional: PDF readers
# -------------------------------
_PDF_BACKENDS = {}
try:
    import pdfplumber
    _PDF_BACKENDS["pdfplumber"] = True
except Exception:
    _PDF_BACKENDS["pdfplumber"] = False

try:
    import PyPDF2
    _PDF_BACKENDS["pypdf2"] = True
except Exception:
    _PDF_BACKENDS["pypdf2"] = False

# -------------------------------
# 0. Ensure NLTK resources are available
# -------------------------------
def _safe_download(resource_path: str, download_name: str):
    """
    Ensure an NLTK resource exists; download if missing.
    
    Parameters
    ----------
    resource_path : str
        Path used internally by NLTK to check for the resource.
    download_name : str
        Name of the NLTK package to download if missing.
    """
    try:
        nltk.data.find(resource_path)
    except LookupError:
        nltk.download(download_name)

# Download essential NLTK packages
_safe_download("tokenizers/punkt", "punkt")
_safe_download("tokenizers/punkt_tab", "punkt_tab")
_safe_download("corpora/stopwords", "stopwords")
_safe_download("sentiment/vader_lexicon", "vader_lexicon")

# -------------------------------
# 1.1 USER CONFIGURATION
# -------------------------------

# Path to ZIP file containing CalWORKs CSAs
ZIP_PATH = r'/Users/cyrillefougere/Desktop/CalWORKs data.zip'  # Edit if needed
BASE_DIR = _Path(ZIP_PATH).parent

# Folder where ZIP will be extracted
EXTRACT_DIR = BASE_DIR / "CalWORKs data_extracted"
os.makedirs(EXTRACT_DIR, exist_ok=True)

# Output folder for WordClouds and results
OUTPUT_DIR = BASE_DIR / "Word Clouds Update"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------
# 1.2 Sanity check of paths
# -------------------------------
print("=== Path Verification ===")
print("ZIP exists:", _Path(ZIP_PATH).exists())
print("Extract dir:", EXTRACT_DIR)
print("Output dir:", OUTPUT_DIR)
print("=========================")

# -------------------------------
# 1.3 Extract ZIP once (if not already extracted)
# -------------------------------
if (not any(EXTRACT_DIR.rglob("*"))) and _Path(ZIP_PATH).exists():
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(EXTRACT_DIR)
print(f"Extracted ZIP to: {EXTRACT_DIR}")

# -------------------------------
# 1.4 List PDF files to analyze
# -------------------------------
pdf_files = list(EXTRACT_DIR.rglob("*.pdf"))
print("Files to analyze:")
for f in pdf_files:
    print(" -", f.relative_to(EXTRACT_DIR))

# -------------------------------
# 1.5 Confirm output directory exists
# -------------------------------
print(f"\nOutput folder ready: {OUTPUT_DIR}\n")

# ----------------------------------------------------
# 2. TEXT READER (PDF / DOCX)
# ----------------------------------------------------
# Features:
# - PDF reading fallback chain: pdfplumber → PyMuPDF (fitz) → PyPDF2 → OCR (first few pages)
# - DOCX reading fallback: python-docx → docx2txt
# - Automatic dispatch by file extension
# ----------------------------------------------------

# 2.1 Check for optional PyMuPDF (fitz) and OCR dependencies
_HAS_FITZ = False
try:
    import fitz  # PyMuPDF
    _HAS_FITZ = True
except Exception:
    pass

_HAS_OCR = False
try:
    import pytesseract
    from PIL import Image
    _HAS_OCR = True
except Exception:
    pass


# 2.2 Lightweight OCR for first `max_pages` of PDF
def _ocr_pdf_pages_to_text(path: _Path, max_pages: int = 5) -> str:
    """
    Rasterize the first `max_pages` of a PDF and extract text via OCR.
    
    Parameters
    ----------
    path : _Path
        Path to PDF file.
    max_pages : int
        Maximum number of pages to OCR.
    
    Returns
    -------
    str
        Extracted text, or empty string if OCR unavailable.
    """
    if not (_HAS_OCR and _HAS_FITZ):
        return ""

    text_parts = []
    try:
        doc = fitz.open(str(path))
        for i, page in enumerate(doc):
            if i >= max_pages:
                break
            pix = page.get_pixmap(dpi=200)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            try:
                t = pytesseract.image_to_string(img)
            except Exception:
                t = ""
            if t.strip():
                text_parts.append(t)
    except Exception:
        return ""

    return "\n".join(text_parts)


# 2.3 PDF Reader with multiple fallback methods
def read_pdf_text(path: _Path) -> str:
    """
    Read text from a PDF using the following fallback chain:
    pdfplumber → PyMuPDF (fitz) → PyPDF2 → OCR (first few pages).
    
    Parameters
    ----------
    path : _Path
        Path to PDF file.
    
    Returns
    -------
    str
        Extracted text or empty string if no text is found.
    """
    # --- 1) pdfplumber ---
    if _PDF_BACKENDS.get("pdfplumber", False):
        try:
            import pdfplumber
            with pdfplumber.open(path) as pdf:
                parts = [page.extract_text() or "" for page in pdf.pages if (page.extract_text() or "").strip()]
            if parts:
                text = "\n".join(parts)
                print(f"[PDF] {path.name} → pdfplumber OK ({len(text)} chars)")
                return text
            else:
                print(f"[PDF] {path.name} → pdfplumber empty.")
        except Exception as e:
            print(f"[PDF] pdfplumber failed on {path.name}: {e}")

    # --- 2) PyMuPDF (fitz) ---
    if _HAS_FITZ:
        try:
            doc = fitz.open(str(path))
            parts = [page.get_text("text") for page in doc if (page.get_text("text") or "").strip()]
            if parts:
                text = "\n".join(parts)
                print(f"[PDF] {path.name} → PyMuPDF OK ({len(text)} chars)")
                return text
            else:
                print(f"[PDF] {path.name} → PyMuPDF empty.")
        except Exception as e:
            print(f"[PDF] PyMuPDF failed on {path.name}: {e}")

    # --- 3) PyPDF2 ---
    if _PDF_BACKENDS.get("pypdf2", False):
        try:
            import PyPDF2
            reader = PyPDF2.PdfReader(str(path))
            parts = [page.extract_text() or "" for page in reader.pages if (page.extract_text() or "").strip()]
            if parts:
                text = "\n".join(parts)
                print(f"[PDF] {path.name} → PyPDF2 OK ({len(text)} chars)")
                return text
            else:
                print(f"[PDF] {path.name} → PyPDF2 empty.")
        except Exception as e:
            print(f"[PDF] PyPDF2 failed on {path.name}: {e}")

    # --- 4) OCR fallback ---
    ocr_text = _ocr_pdf_pages_to_text(path, max_pages=5)
    if ocr_text.strip():
        print(f"[PDF] {path.name} → OCR sample OK ({len(ocr_text)} chars; first 5 pages)")
        return ocr_text

    # --- No text found ---
    print(f"[PDF] {path.name} → NO TEXT (scanned/secured?)")
    return ""


# 2.4 DOCX Reader with python-docx → docx2txt fallback
def read_docx_text(path: _Path) -> str:
    """
    Extract text from a DOCX file, falling back from python-docx to docx2txt.
    
    Parameters
    ----------
    path : _Path
        Path to DOCX file.
    
    Returns
    -------
    str
        Extracted text (empty string if none found).
    """
    text1 = ""
    try:
        from docx import Document
        doc = Document(str(path))
        text1 = "\n".join([p.text for p in doc.paragraphs])
    except Exception:
        text1 = ""

    if len(text1.strip()) < 50:
        try:
            import docx2txt
            text2 = docx2txt.process(str(path))
            if len(text2.strip()) > len(text1.strip()):
                return text2
        except Exception:
            pass

    return text1


# 2.5 Dispatch function to read any text file
def read_any_text(path: _Path) -> str:
    """
    Dispatch text reading by file extension.
    
    Parameters
    ----------
    path : _Path
        Path to the file.
    
    Returns
    -------
    str
        Extracted text for PDF or DOCX, empty string for other types.
    """
    ext = path.suffix.lower()
    if ext == ".pdf":
        return read_pdf_text(path)
    elif ext == ".docx":
        return read_docx_text(path)
    else:
        return ""

# ----------------------------------------------------
# 3. STOPWORDS (policy-tuned) + GEO TERMS + LEMMATIZATION
# ----------------------------------------------------
# Features:
# - Custom stopwords combining NLTK base, policy/admin terms, organizational terms, geo terms, and filler words
# - Keeps important policy-relevant terms intact (KEEP_TERMS)
# - Word lemmatization for noun → verb → adjective
# - Token cleaning and filtering
# ----------------------------------------------------

from nltk.stem import WordNetLemmatizer

def build_stopwords() -> set:
    """
    Build a comprehensive, policy-tuned stopword set for CSA document analysis.

    Combines:
        - Standard NLTK English stopwords
        - Administrative / metrics terms
        - Organizational terms
        - Geographic terms
        - Common report filler words
    Excludes KEEP_TERMS to preserve policy-relevant vocabulary.
    """
    base_sw = set(stopwords.words('english'))

    # 3.1 Administrative / metrics terms
    admin_metrics = {
        "achieve", "analysis", "app", "applicable", "appraisal", "application", "applications", "apps",
        "assessment", "assistance", "assist", "attendance", "average", "business", "call",
        "case", "cases", "communication", "comparison", "compared", "complete", "completion",
        "contact", "cycle", "data", "datas", "decrease", "documentation", "eligibility", "eligible",
        "employmentrate", "engagementrate", "exit", "feedback", "improve", "improvement",
        "improvements", "improving", "increase", "management", "measure", "measures",
        "median", "meet", "meeting", "meetings", "opportunities", "opportunity",
        "orientation", "outcomes", "participationrate", "percent", "percentage", "performance",
        "period", "phone", "plan", "plans", "practice", "practices", "process", "procedure",
        "procedures", "program", "programs", "referral", "referrals", "requirement",
        "requirements", "resource", "resources", "results", "review", "reviewed", "reviews",
        "service", "services", "sip", "sps", "strategy", "strategies", "support", "supportive",
        "supports", "timeline", "timeliness", "wtw", "population"
    }

    # 3.2 Organizational / program terms
    org_terms = {
        "agencies", "agency", "caseworker", "caseworkers", "cbos", "cdss", "cct", "center", "centers",
        "client", "clients", "collaborator", "collaborators", "customer", "customers",
        "calsaw", "calsaws", "calworks", "cwd", "debs", "department", "departments",
        "des", "dess", "dha", "doe", "dss", "ecm", "ecms", "edc", "ess", "esss", "etw", "hhs",
        "hsa", "icdss", "kchsa", "manager", "managers", "mcdss", "oar", "ocat", "office", "offices",
        "ore", "ota", "partner", "partners", "participant", "participants", "programmatic",
        "provider", "providers", "recipient", "recipients", "sfhsa", "specialist", "specialists",
        "staff", "staffed", "staffing", "system", "tad", "team", "teams", "ts", "unit", "units",
        "vendor", "vendors", "vchsa", "ychhsd", "welfare", "wex"
    }   

    # 3.3 Geographic terms (generic + county tokens)
    geo_terms = {
        "area", "areas", "bay", "cal", "california", "central", "city", "cities",
        "coast", "coastal", "county", "counties", "district", "districts", "inland",
        "metro", "northern", "region", "regional", "southern", "valley", "valleys",
        "barbara", "benito", "bernardino", "clara", "contra", "costa", "cruz", "de", "del",
        "dorado", "diego", "el", "francisco", "glenn", "humboldt", "imperial", "inyo",
        "joaquin", "kern", "kings", "lake", "lassen", "la", "luis", "marin", "mariposa",
        "mateo", "mendocino", "merced", "mono", "monterey", "napa", "nevada", "obispo",
        "orange", "placer", "plumas", "riverside", "sacramento", "san", "santa", "shasta",
        "sierra", "siskiyou", "solano", "sonoma", "stanislaus", "sutter", "tehama", "trinity",
        "tulare", "tuolumne", "ventura", "yolo", "yuba", "alameda", "alpine", "amador",
        "angeles", "butte", "calaveras", "colusa", "fresno", "madera", "modoc", "norte", "los"
    }

    # 3.4 Common report filler words
    report_filler = {
        "additional", "additionally", "also", "among", "amongst", "and", "available",
        "bad", "best", "cause", "caused", "causes", "consider", "considered", "considering",
        "continue", "continued", "continuing", "continuum", "could", "current",
        "daily", "day", "determine", "determined", "determines", "different",
        "due", "eight", "example", "examples", "exited", 
        "figure", "five", "focused", "focusing", "focus", "four",
        "further", "good", "group", "higher", "however", "identify", "identified",
        "identifying", "impact", "impacted", "impacting", "impacts", "information",
        "initial", "internal", "issue", "january", "february", "march", "april", "may",
        "june", "july", "august", "september", "october", "november", "december",
        "lower", "month", "monthly", "months", "might", "new", "no", "note", "noted",
        "number", "numbers", "observed", "often", "ongoing", "overall", "particularly",
        "particular", "part", "parts", "people", "person", "persons", "post",
        "receive", "regard", "regarding", "regards", "relating", "related", "relatedly",
        "report", "respectively", "result", "resulted", "self", "since", "six", "seven",
        "so", "specific", "specifically", "step", "tab", "table", "ten", "three",
        "time", "typically", "typical", "two", "use", "variety", "various", "within",
        "well", "yes", "year", "yearly", "years", "one", "toward", "address",
        "section", "social", "high", "individual", "need", "etc", "rate"
    }

    # 3.5 Keep important policy-relevant terms
    KEEP_TERMS = {
        "employment","employ","employed","job","jobs","work","workforce","wage","wages","income","earnings",
        "engagement","engage","engaged","participation","participate","sanction","sanctions","sanctioned",
        "barrier","barriers","challenge","challenges","housing","homeless","homelessness","rent","cost","costs","poverty",
        "transportation","childcare","child","children","care","caregiver","family","families",
        "domestic","abuse","violence","safety","mental","health","healthcare","behavioral","behavioralhealth",
        "language","bilingual","spanish","english","access","accessibility","disparities","equity","racial","race","racism",
        "covid","pandemic","public","benefits","benefit","stabilization","stabilize","stabilizing",
        "technology","digital","internet","immigrant","immigrants","refugee","refugees","rural","urban",
        "parent","parents","parenting","eviction","evictions","bus","transit","equitable"
    }

    # Combine all sets and exclude KEEP_TERMS
    sw = base_sw.union(admin_metrics, org_terms, geo_terms, report_filler)
    sw = {w for w in sw if w not in KEEP_TERMS}
    return sw

# Instantiate STOPWORDS set
STOPWORDS = build_stopwords()

# ----------------------------------------------------
# 3.6 Lemmatization setup
# ----------------------------------------------------
_safe_download("corpora/wordnet", "wordnet")
_safe_download("corpora/omw-1.4", "omw-1.4")

LEM = WordNetLemmatizer()

def lemmatize_token(token: str) -> str:
    """
    Lemmatize a token in the sequence: noun → verb → adjective.
    Returns lemmatized token as string.
    """
    t = LEM.lemmatize(token, pos='n')
    t = LEM.lemmatize(t, pos='v')
    t = LEM.lemmatize(t, pos='a')
    return t

# ----------------------------------------------------
# 3.7 Clean and tokenize text
# ----------------------------------------------------
def clean_and_tokenize(text: str) -> list:
    """
    Clean text, tokenize, lemmatize, and filter stopwords.
    
    Steps:
    1. Lowercase the text
    2. Replace hyphens and slashes with spaces
    3. Remove non-alphabetic characters
    4. Collapse multiple spaces
    5. Tokenize
    6. Lemmatize each token
    7. Remove stopwords, short tokens (<3 chars), and numeric tokens
    """
    text = text.lower()
    text = re.sub(r"[-/]", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    
    tokens = nltk.word_tokenize(text)
    lemmas = [lemmatize_token(t) for t in tokens]
    cleaned = [t for t in lemmas if t not in STOPWORDS and len(t) > 2 and not t.isnumeric()]
    return cleaned