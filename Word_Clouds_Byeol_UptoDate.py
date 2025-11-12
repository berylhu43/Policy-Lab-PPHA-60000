########################################
# CalWORKs CSA Policy Analysis Pipeline — Merged (Byeol)
# Unifies: Word_Clouds_Ver 3..py + Word_Clouds_Byeol.py
# - Robust PDF extract (pdfplumber → PyMuPDF → PyPDF2 → OCR sample)
# - Safe font paths (Windows/macOS/Linux)
# - STOPWORDS = union of both scripts (no omissions)
# - Lemmatization + thesaurus-based collapsing
# - WordClouds (Top-20) + overall and per-word sentence sentiment
# - Statewide aggregation
# - Optional Word2Vec theme cosine + probe prompts
########################################

import os
import re
import sys
import zipfile
import logging
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import nltk
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud

# Quiet noisy PDF logs
for noisy in ("pdfminer","pdfminer.pdfinterp","pdfminer.cmapdb","pdfminer.psparser","pdfminer.pdfpage"):
    logging.getLogger(noisy).setLevel(logging.ERROR)

# Optional DOCX
try:
    import docx2txt
    from docx import Document
    _HAS_DOCX = True
except Exception:
    _HAS_DOCX = False

# Optional gensim (Word2Vec)
W2V_OK = True
try:
    from gensim.models import Word2Vec
    from gensim.downloader import load as gensim_load
except Exception:
    W2V_OK = False

# Optional PDF libs
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

# NLTK resources
def _safe_download(res_path, name):
    try:
        nltk.data.find(res_path)
    except LookupError:
        nltk.download(name)

_safe_download("tokenizers/punkt", "punkt")
_safe_download("tokenizers/punkt_tab", "punkt_tab")
_safe_download("corpora/stopwords", "stopwords")
_safe_download("sentiment/vader_lexicon", "vader_lexicon")
_safe_download("corpora/wordnet", "wordnet")
_safe_download("corpora/omw-1.4", "omw-1.4")

# ============= USER CONFIG =============
ZIP_PATH = r"C:\Users\nfnfh\OneDrive\Desktop\Python\CalWORKs data.zip"  # <-- edit this
BASE_DIR = Path(ZIP_PATH).parent

EXTRACT_DIR = BASE_DIR / "CalWORKs data_extracted"
EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_DIR = BASE_DIR / "Word Clouds Update"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=== Path Verification ===")
print("ZIP exists:", Path(ZIP_PATH).exists())
print("Extract dir:", EXTRACT_DIR)
print("Output dir:", OUTPUT_DIR)
print("=========================")

# Extract once if needed
if (not any(EXTRACT_DIR.rglob("*"))) and Path(ZIP_PATH).exists():
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(EXTRACT_DIR)
print(f"📦 Extracted ZIP to: {EXTRACT_DIR}")

# List PDFs
pdf_files = list(EXTRACT_DIR.rglob("*.pdf"))
print("🔍 Files to analyze:")
for f in pdf_files:
    try:
        print(" -", f.relative_to(EXTRACT_DIR))
    except Exception:
        print(" -", f)
print(f"\n📂 Output: {OUTPUT_DIR}\n")

# ============= Readers (PDF/DOCX) =============
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

def _ocr_pdf_pages_to_text(path: Path, max_pages: int = 5) -> str:
    """OCR first N pages (fallback)."""
    if not (_HAS_OCR and _HAS_FITZ):
        return ""
    parts = []
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
                parts.append(t)
    except Exception:
        return ""
    return "\n".join(parts)

def read_pdf_text(path: Path) -> str:
    """Try pdfplumber → PyMuPDF → PyPDF2 → OCR sample."""
    # 1) pdfplumber
    if _PDF_BACKENDS.get("pdfplumber", False):
        try:
            with pdfplumber.open(path) as pdf:
                parts = [p.extract_text() or "" for p in pdf.pages if (p.extract_text() or "").strip()]
            if parts:
                text = "\n".join(parts)
                print(f"[PDF] {path.name} → pdfplumber OK ({len(text)} chars)")
                return text
            else:
                print(f"[PDF] {path.name} → pdfplumber empty.")
        except Exception as e:
            print(f"[PDF] pdfplumber failed on {path.name}: {e}")

    # 2) PyMuPDF
    if _HAS_FITZ:
        try:
            doc = fitz.open(str(path))
            parts = [page.get_text('text') or "" for page in doc if (page.get_text('text') or "").strip()]
            if parts:
                text = "\n".join(parts)
                print(f"[PDF] {path.name} → PyMuPDF OK ({len(text)} chars)")
                return text
            else:
                print(f"[PDF] {path.name} → PyMuPDF empty.")
        except Exception as e:
            print(f"[PDF] PyMuPDF failed on {path.name}: {e}")

    # 3) PyPDF2
    if _PDF_BACKENDS.get("pypdf2", False):
        try:
            reader = PyPDF2.PdfReader(str(path))
            parts = [p.extract_text() or "" for p in reader.pages if (p.extract_text() or "").strip()]
            if parts:
                text = "\n".join(parts)
                print(f"[PDF] {path.name} → PyPDF2 OK ({len(text)} chars)")
                return text
            else:
                print(f"[PDF] {path.name} → PyPDF2 empty.")
        except Exception as e:
            print(f"[PDF] PyPDF2 failed on {path.name}: {e}")

    # 4) OCR fallback
    ocr_text = _ocr_pdf_pages_to_text(path, max_pages=5)
    if ocr_text.strip():
        print(f"[PDF] {path.name} → OCR sample OK ({len(ocr_text)} chars; first 5 pages)")
        return ocr_text

    print(f"[PDF] {path.name} → NO TEXT (scanned/secured?)")
    return ""

def read_docx_text(path: Path) -> str:
    """DOCX reader with two backends."""
    if not _HAS_DOCX:
        return ""
    text1 = ""
    try:
        doc = Document(str(path))
        text1 = "\n".join([p.text for p in doc.paragraphs])
    except Exception:
        text1 = ""
    if len(text1.strip()) < 50:
        try:
            text2 = docx2txt.process(str(path))
            if len(text2.strip()) > len(text1.strip()):
                return text2
        except Exception:
            pass
    return text1

def read_any_text(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return read_pdf_text(path)
    elif ext == ".docx":
        return read_docx_text(path)
    else:
        return ""

# ============= STOPWORDS (union, no omissions) =============
# =========================
# STRONGER FILTER PACK
# - Merge: existing stopwords + your Delete List + admin acronyms
# - Stronger general verb filter (light verbs, reporting verbs, boilerplate verbs)
# - No overlap with KEEP_TERMS
# - Optional regex to drop acronym-like tokens at tokenize-time
# =========================

# 0) Put this near other imports
import re
from nltk.stem import WordNetLemmatizer

# 1) Drop-in lists: normalized to lowercase
#    admin acronyms, and stronger general verbs
DELETE_LIST = {
    # ---- From your "Word Clouds Delete List.docx" (county-by-county) ----
    # (keep lowercase; duplicates harmless)
    "meet","exit","need","social","due","address","high","available","issue",
    "sierra","population","individual","complete","receive","different","cdss","doe",
    "table","tad","overall","wex","section","system","ota","kchsa","post",
    "use","vchsa","new","toward","etc","well","make","way","take","order",
    "reminder","text","message","icw","level"
}

ADMIN_ACRONYMS = {
    # Add all admin/agency/program acronyms you want removed (lowercased)
    "acssa","etw","etws","cdss","doe","tad","tws","wex","icw","ota","kchsa","vchsa",
    "calsaws","calsaw","oar","ess","esss","hsa","sfhsa","dha","debs","cbos",
    "cwd","ocat","dss","ts","ych","ychhsd","ecm","ecms","edc","pst"
}

# Stronger generic verb (light/reporting/boilerplate) list.
# Note: we *do not* include policy-critical verbs that overlap KEEP_TERMS.
GENERAL_VERBS = {
    "make","do","get","take","use","provide","include","ensure","allow","require",
    "meet","address","receive","complete","achieve","support","assist","help",
    "show","see","indicate","report","note","observe","focus","consider","determine",
    "continue","improve","increase","decrease","describe","compare","assess","review","update",
    "participate","engage","implement","identify","monitor","measure","evaluate",
}

# Months/dates, table artifacts, and other boilerplate
DATE_MONTH_FILLER = {
    "january","february","march","april","may","june","july","august",
    "september","october","november","december","year","years","month","months",
    "post","overall","section","table","text","message",
    "quarter","qaurterly"
}

# Optional: generic numeric words and connectors (safe to keep filtered)
REPORT_CONNECTORS = {
    "one","two","three","four","five","six","seven","eight","nine","ten",
    "first", "second","third","fourth","fifth","sixth","seventh","eighth","nineth","tenth",
    "and","also","so","well","within","various","variety","example","examples",
    "specific","specifically","particular","particularly","available",
    "different","internal","overall","since","often","typically","typical",
    "respectively","related","regarding","among","amongst",
}

# 2) Helper to recognize acronym-like tokens (after lowercasing)
#    We only drop if token is in ADMIN_ACRONYMS; we do not blanket-drop
#    all short tokens to avoid false positives.
ACRONYM_LIKE = re.compile(r"^[a-z]{2,5}\d?$")  # e.g., 'cdss','etw','tad','ecm','ota'


def build_stopwords() -> set:
    """
    Build a comprehensive, policy-tuned stopword set:
    - NLTK English stopwords
    - Administrative/metrics/organizational/geographic/report fillers (your existing sets)
    - PLUS: Delete List, admin acronyms, stronger general verbs, date/month fillers, connectors
    - Excludes KEEP_TERMS to preserve policy-relevant vocabulary
    """
    base_sw = set(stopwords.words('english'))

    # ---- Keep your existing domain stopword sets as-is (shortened for clarity) ----
    admin_metrics = {
        "achieve","analysis","app","applicable","appraisal","application","applications","apps",
        "assessment","attendance","average","business","call","case","cases","communication",
        "comparison","compared","complete","completion","contact","cycle","data","datas",
        "decrease","documentation","eligibility","eligible","employmentrate","engagementrate",
        "exit","feedback","improve","improvement","improvements","improving","increase",
        "management","measure","measures","median","orientation","outcomes","participationrate",
        "percent","percentage","performance","period","phone","plan","plans","practice","practices",
        "process","procedure","procedures","program","programs","referral","referrals",
        "requirement","requirements","resource","resources","results","review","reviewed","reviews",
        "service","services","sip","sps","strategy","strategies","support","supportive","supports",
        "timeline","timeliness","wtw","population","CSA", "csa"
    }

    org_terms = {
        "agencies","agency","caseworker","caseworkers","cbos","cct","center","centers","client","clients",
        "collaborator","collaborators","customer","customers","calworks","cwd","department",
        "departments","des","dess","dha","dss","cwes","hhs","hsa","lcdss","manager","managers","office","offices",
        "partner","partners","participant","participants","programmatic","provider","providers",
        "recipient","recipients","sfhsa","specialist","specialists","ssd","staff","staffed","staffing",
        "system","team","teams","unit","units","vendor","vendors","welfare",
    }

    geo_terms = {
        "angeles","area","areas","bay","cal","california","central","city","cities","coast","coastal",
        "county","counties","district","districts","inland","metro","northern","region","regional",
        "southern","valley","valleys","barbara","benito","bernardino","clara","contra","costa","cruz",
        "de","del","dorado","diego","el","francisco","glenn","humboldt","imperial","inyo","joaquin",
        "kern","kings","lake","lassen","la","luis","marin","mariposa","mateo","mendocino","merced",
        "mono","monterey","napa","nevada","obispo","orange","placer","plumas","riverside","sacramento",
        "san","santa","shasta","sierra","siskiyou","solano","sonoma","stanislaus","sutter","tehama",
        "trinity","tulare","tuolumne","ventura","yolo","yuba","alameda","alpine","amador","angeles",
        "butte","calaveras","colusa","fresno","madera","modoc","norte","los",
    }

    report_filler = {
        "additional","additionally","among","amongst","and","available","bad","best","cause","caused",
        "causes","consider","considered","considering","continue","continued","continuing","continuum",
        "could","current","daily","day","determine","determined","determines","different","due",
        "example","examples","exited","figure","five","focused","focusing","focus","four","further",
        "good","group","higher","however","identify","identified","identifying","impact","impacted",
        "impacting","impacts","information","initial","internal","issue","lower","month","monthly",
        "months","might","new","no","note","noted","number","numbers","observed","often","one", "ongoing", 
        "overall","particularly","particular","part","parts","people","person","persons","post",
        "receive","regard","regarding","regards","relating","related","relatedly","report","respectively",
        "result","resulted","since","six","seven","so","specific","specifically","step","tab","table",
        "ten","three","time","typically","typical","two","use","variety","various","within","well","yes",
        "year","yearly","years","one","toward","address","section","social","high","individual","need",
        "rate","text","message","describe","develop","enhance","select","lead","share","follow","conduct","small","big"
    } 

    # ---- Policy-critical terms to preserve ----
    KEEP_TERMS = {
        "employment","employ","employed","job","jobs","work","workforce","wage","wages","income","earnings",
        "engagement","engage","engaged","participation","participate","sanction","sanctions","sanctioned",
        "barrier","barriers","challenge","challenges","housing","homeless","homelessness","rent","cost","costs","poverty",
        "transportation","childcare","child","children","care","caregiver","family","families",
        "domestic","abuse","violence","safety","mental","health","healthcare","behavioral","behavioralhealth",
        "language","bilingual","spanish","english","access","accessibility","disparities","equity","racial","race","racism",
        "covid","pandemic","public","benefits","benefit","stabilization","stabilize","stabilizing",
        "technology","digital","internet","immigrant","immigrants","refugee","refugees","rural","urban",
        "parent","parents","parenting","eviction","evictions","bus","transit","equitable",
    }

    # ---- Union all filters ----
    sw = set().union(
        base_sw,
        admin_metrics,
        org_terms,
        geo_terms,
        report_filler,
        DELETE_LIST,
        ADMIN_ACRONYMS,
        GENERAL_VERBS,
        DATE_MONTH_FILLER,
        REPORT_CONNECTORS,
    )

    # ---- Exclude KEEP_TERMS explicitly ----
    sw = {w for w in sw if w not in KEEP_TERMS}
    return sw

# Rebuild STOPWORDS with the stronger filter set
STOPWORDS = build_stopwords()

# 3) OPTIONAL: during tokenization, drop acronym-like tokens if they belong to ADMIN_ACRONYMS
def _drop_if_admin_acronym(token: str) -> bool:
    """Return True if token should be dropped as an admin acronym."""
    # token is already lowercase at this point
    if token in ADMIN_ACRONYMS:
        return True
    # Optionally, drop short acronym-like tokens even if not explicitly listed:
    # return bool(ACRONYM_LIKE.match(token))
    return False

# 4) Update your clean_and_tokenize() to call _drop_if_admin_acronym:
def clean_and_tokenize(text: str) -> list[str]:
    """
    Clean, tokenize, lemmatize, and filter tokens:
    - Lowercase and strip punctuation
    - Lemmatize (n -> v -> a)
    - Remove numeric/short tokens, STOPWORDS, and admin acronyms
    """
    text = text.lower()
    text = re.sub(r"[-/]", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = nltk.word_tokenize(text)

    # Lemmatization (noun -> verb -> adjective)
    LEM = WordNetLemmatizer()
    lemmas = LEM.lemmatize(" ".join(tokens), pos='n').split()
    lemmas = [WordNetLemmatizer().lemmatize(t, pos='v') for t in lemmas]
    lemmas = [WordNetLemmatizer().lemmatize(t, pos='a') for t in lemmas]

    cleaned = []
    for t in lemmas:
        if len(t) < 3:            # too short
            continue
        if t.isnumeric():         # numeric
            continue
        if t in STOPWORDS:        # in stopword pack
            continue
        if _drop_if_admin_acronym(t):  # in admin acronyms
            continue
        cleaned.append(t)
    return cleaned


# ============= Utils: title, font, colors =============
def infer_county_name(path: str) -> str:
    base = os.path.basename(path)
    label = os.path.splitext(base)[0].replace("_", " ").strip()
    words = label.split()
    if len(words) > 7:
        label = " ".join(words[:7])
    return label

def get_safe_font_path() -> str | None:
    candidates = []
    if os.name == "nt":  # Windows
        candidates = [
            r"C:\Windows\Fonts\malgun.ttf",
            r"C:\Windows\Fonts\arial.ttf",
            r"C:\Windows\Fonts\calibri.ttf",
            r"C:\Windows\Fonts\seguiemj.ttf",
        ]
    elif sys.platform == "darwin":  # macOS
        candidates = [
            "/System/Library/Fonts/Helvetica.ttc",
            "/System/Library/Fonts/SFNS.ttf",
        ]
    else:  # Linux
        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        ]
    for fp in candidates:
        if os.path.exists(fp):
            return fp
    return None

from matplotlib import colors as mcolors

def frequency_color_func(freq_dict):
    max_freq = max(freq_dict.values())
    min_freq = min(freq_dict.values())
    def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
        freq = freq_dict[word]
        norm = (freq - min_freq) / (max_freq - min_freq + 1e-5)
        # light blue → deeper blue
        return mcolors.to_hex((0.658*(1-norm)+0.0*norm, 0.792*(1-norm)+0.2*norm, 1*(1-norm)+0.2*norm))
    return color_func

# ============= Thesaurus collapsing =============
def build_thesaurus():
    groups = {
        "employment": {"employment","employ","employed","job","jobs","work","workforce"},
        "wages": {"wage","wages","pay","earnings","income"},
        "engagement": {"engagement","engage","engaged","participation","participate","outreach","attendance","orientation"},
        "sanctions": {"sanction","sanctions","sanctioned","compliance","penalty","penalties","noncompliance","good","cause","goodcause"},
        "housing": {"housing","rent","rents","eviction","evictions","homeless","homelessness","shelter"},
        "childcare": {"childcare","child","children","care","parent","parents","parenting","caregiver"},
        "transportation": {"transportation","bus","buses","transit","distance","commute"},
        "mental_health": {"mental","health","healthcare","behavioral","counseling","therapy"},
        "language_access": {"language","bilingual","spanish","english","interpreter","translation","access","accessibility"},
        "equity": {"equity","equitable","disparities","racial","race","racism"},
        "poverty": {"poverty","lowincome","low","income","cost","costs"},
        "violence": {"domestic","violence","abuse","dv","safety"},
        "technology": {"technology","digital","internet","online","device","devices"},
        "immigration": {"immigrant","immigrants","refugee","refugees"},
    }
    v2c = {v: c for c, vs in groups.items() for v in vs}
    return groups, v2c

THESAURUS, VAR2CANON = build_thesaurus()

def collapse_tokens_to_canon(tokens: list[str]) -> list[str]:
    return [VAR2CANON.get(t, t) for t in tokens]

# ============= Stats & plots =============
def compute_word_stats(tokens: list[str]) -> pd.DataFrame:
    total = len(tokens)
    if total == 0:
        return pd.DataFrame(columns=["word","freq","relative_freq"])
    freq = Counter(tokens)
    rows = [{"word": w, "freq": c, "relative_freq": c/total} for w, c in freq.items()]
    return pd.DataFrame(rows).sort_values("freq", ascending=False).reset_index(drop=True)

def build_wordcloud(freq_df: pd.DataFrame, outpath: Path, title: str | None = None, top_n: int = 20, show: bool = False):
    if freq_df.empty:
        print(f"[WARN] No words to plot for {outpath}")
        return
    freq_df_top = freq_df.head(top_n)
    freq_dict = dict(zip(freq_df_top["word"], freq_df_top["freq"]))

    wc_kwargs = dict(
        width=1200, height=900, background_color="white",
        prefer_horizontal=0.9, margin=4, max_words=top_n,
        collocations=False, contour_color="#003366", contour_width=1,
        color_func=frequency_color_func(freq_dict),
    )
    _font = get_safe_font_path()
    if _font:
        wc_kwargs["font_path"] = _font

    wc = WordCloud(**wc_kwargs).generate_from_frequencies(freq_dict)
    fig = plt.figure(figsize=(10, 8))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")

    if title is None:
        title = infer_county_name(str(outpath))

    if title:
        plt.suptitle(title, fontsize=20, fontweight="bold", color="#003366", y=1.03)
    plt.title(f"Top {top_n} Most Frequent Words in Report", fontsize=14, color="#444444", style="italic", pad=12)
    plt.tight_layout()

    if show:
        plt.show()
    else:
        print(f"Saving wordcloud → {outpath}")
        plt.savefig(outpath, dpi=600, bbox_inches="tight")
    plt.close(fig)

# Simple theme extractors
CHALLENGE_TERMS = {
    "barrier","barriers","challenge","challenges","transportation","housing",
    "homeless","homelessness","childcare","care","domestic","abuse","violence",
    "mental","health","poverty","cost","costs","wage","wages","language",
    "bilingual","access","disparities","equity","sanction","sanctions","sanctioned",
    "rent","eviction","evictions"
}
CONTEXT_TERMS = {
    "family","families","parent","parents","child","children","bilingual","spanish","english",
    "immigrant","immigrants","refugee","refugees","rural","urban","poverty","wage","wages",
    "income","earnings","participation","engagement","engage","engaged","work","employment",
    "job","jobs","workforce"
}

def extract_policy_themes(tokens: list[str], keywords: set[str], top_n: int = 5) -> list[str]:
    freq = Counter(t for t in tokens if t in keywords)
    return [w for w, _ in freq.most_common(top_n)]

# ============= Word2Vec utils (optional) =============
def build_sentences_for_w2v(raw_text: str) -> list[list[str]]:
    sentences = []
    for sent in nltk.sent_tokenize(raw_text):
        toks = clean_and_tokenize(sent)
        if toks:
            sentences.append(toks)
    return sentences

def get_w2v_model(sentences: list[list[str]]):
    if not W2V_OK:
        print("⚠️ gensim not available — Word2Vec disabled.")
        return None, None
    try:
        wv = gensim_load("glove-wiki-gigaword-100")
        print("[INFO] Loaded pretrained GloVe (100D).")
        return None, wv
    except Exception:
        print("[WARN] Could not load GloVe — training Word2Vec locally.")
    try:
        model = Word2Vec(sentences=sentences, vector_size=100, window=5, min_count=3,
                         workers=4, sg=1, negative=10, epochs=15)
        print("[INFO] Trained local Word2Vec.")
        return model, model.wv
    except Exception as e:
        print(f"[ERROR] Word2Vec training failed: {e}")
        return None, None

def nearest_terms(wv, term: str, topn: int = 10) -> list[str]:
    if (wv is None) or (not hasattr(wv, "key_to_index")) or (term not in wv.key_to_index):
        return []
    try:
        return [w for w, _ in wv.most_similar(term, topn=topn)]
    except Exception:
        return []

def vector_mean(wv, tokens: list[str]) -> np.ndarray | None:
    if (wv is None) or (not hasattr(wv, "key_to_index")):
        return None
    vectors = [wv[t] for t in tokens if t in wv.key_to_index]
    return np.mean(vectors, axis=0) if vectors else None

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return np.nan
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return np.nan
    return float(np.dot(a, b) / (na * nb))

THEMATIC_SEEDS = {
    "employment": ["employment","job","workforce","earnings"],
    "childcare": ["childcare","child","care","parent"],
    "housing": ["housing","rent","homelessness","shelter"],
    "transport": ["transportation","bus","transit","distance"],
    "mental_health": ["mental","behavioral","health","counseling"],
    "sanctions": ["sanction","compliance","penalty","good","cause"],
    "language_access": ["language","translation","interpreter","english","spanish"],
    "dv": ["domestic","violence","abuse","safety"],
    "equity": ["equity","disparities","racial"],
    "poverty": ["poverty","lowincome","income","cost","costs"],
}

def build_theme_vectors(wv) -> dict[str, np.ndarray | None]:
    theme_vectors = {}
    for theme, seeds in THEMATIC_SEEDS.items():
        valid = [s for s in seeds if s in (wv.key_to_index if wv else {})]
        theme_vectors[theme] = vector_mean(wv, valid)
    return theme_vectors

def document_vector(wv, sentences: list[list[str]]) -> np.ndarray | None:
    flat = [t for s in sentences for t in s]
    return vector_mean(wv, flat)

def build_probe_prompt(county: str, theme_scores: dict[str, dict[str, float]], wv, top_k: int = 3) -> str:
    if county not in theme_scores:
        return f"No theme scores found for {county}."
    scores = theme_scores[county]
    ranked = sorted(scores.items(), key=lambda x: (-x[1] if x[1] == x[1] else 1))
    top_themes = ranked[:top_k]

    lines = []
    for theme, score in top_themes:
        seeds = THEMATIC_SEEDS[theme]
        neighbors = []
        for s in seeds[:2]:
            neighbors += nearest_terms(wv, s, topn=5)
        uniq, seen = [], set()
        for n in neighbors:
            if n not in seen:
                uniq.append(n); seen.add(n)
            if len(uniq) >= 6: break
        suggestion = ", ".join(uniq) if uniq else "related terms"
        lines.append(f"- {theme} (score={score:.2f}): consider {suggestion}")
    return f"Key areas to probe for {county}:\n" + "\n".join(lines)

# ============= Sentiment (overall + per-word) =============
def compute_sentiment_sentence_avg(raw_text: str, sia) -> float:
    sents = nltk.sent_tokenize(raw_text)
    scores = []
    for s in sents:
        s = s.strip()
        if len(s) < 5:
            continue
        scores.append(sia.polarity_scores(s)["compound"])
    return (sum(scores) / len(scores)) if scores else 0.0

def interpret_sentiment(score: float) -> str:
    if score < 0.05:
        return "problem-focused tone (risks/gaps/barriers)"
    elif score < 0.15:
        return "mixed tone (barriers & solutions)"
    else:
        return "strength-oriented tone (successes/improvements)"

def plot_sentiment_bar(sentiment_avg: float, outpath: Path, title: str | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    fig = plt.figure(figsize=(4, 5))
    ax = fig.add_subplot(111)
    ax.bar([0], [sentiment_avg], color="skyblue", edgecolor="black")
    ax.set_ylim(-1, 1)
    ax.set_xticks([0])
    ax.set_xticklabels(["sentiment"])
    ax.axhline(0, lw=1, color="gray")
    ax.set_ylabel("VADER compound (–1..1)")
    if title:
        ax.set_title(title)
    plt.tight_layout()
    print(f"📁 Saving sentiment bar: {outpath}")
    plt.savefig(outpath, dpi=300)
    plt.close(fig)

def _sentence_tokens_canonical(sentence: str) -> set[str]:
    toks = clean_and_tokenize(sentence)
    return set(collapse_tokens_to_canon(toks))

def compute_word_level_sentiment(raw_text: str, freq_df: pd.DataFrame, top_n: int = 20, sia: SentimentIntensityAnalyzer | None = None) -> pd.DataFrame:
    if sia is None:
        sia = SentimentIntensityAnalyzer()
    if freq_df is None or freq_df.empty:
        return pd.DataFrame(columns=["word", "avg_sentiment", "n_sentences"])

    top_words = list(freq_df.head(top_n)["word"].values)
    sentences = [s.strip() for s in nltk.sent_tokenize(raw_text) if len(s.strip()) >= 5]
    sent_token_sets = [_sentence_tokens_canonical(s) for s in sentences]
    sent_scores = [sia.polarity_scores(s)["compound"] for s in sentences]

    rows = []
    for w in top_words:
        relevant_scores = [sc for stoks, sc in zip(sent_token_sets, sent_scores) if w in stoks]
        avg_sent = (sum(relevant_scores) / len(relevant_scores)) if relevant_scores else np.nan
        rows.append({"word": w, "avg_sentiment": avg_sent, "n_sentences": len(relevant_scores)})
    return pd.DataFrame(rows)

def plot_word_sentiment_bar(word_sent_df: pd.DataFrame, outpath: Path, title: str | None = None) -> None:
    if word_sent_df is None or word_sent_df.empty:
        print(f"[WARN] No per-word sentiment data available for {outpath}.")
        return
    plot_df = word_sent_df.copy()
    plot_df["avg_sentiment"] = pd.to_numeric(plot_df["avg_sentiment"], errors="coerce").fillna(0.0)
    plot_df["n_sentences"] = pd.to_numeric(plot_df["n_sentences"], errors="coerce").fillna(0).astype(int)

    x = plot_df["word"].values
    y = plot_df["avg_sentiment"].values
    counts = plot_df["n_sentences"].values
    if len(x) == 0:
        print(f"[WARN] No valid words to plot for {outpath}.")
        return

    import matplotlib
    matplotlib.use("Agg")
    fig = plt.figure(figsize=(max(8, len(x) * 0.6), 5))
    ax = fig.add_subplot(111)
    ax.bar(range(len(x)), y, color="skyblue", edgecolor="black")
    ax.set_ylim(-1, 1)
    ax.axhline(0, lw=1, color="gray")
    ax.set_xticks(range(len(x)))
    ax.set_xticklabels(x, rotation=45, ha="right")
    ax.set_ylabel("Avg sentence sentiment (VADER)")
    ax.set_xlabel("Top frequent words")
    for i, (val, n) in enumerate(zip(y, counts)):
        try:
            ax.text(
                i,
                float(val) + (0.02 if val >= 0 else -0.05),
                f"n={int(n)}",
                ha="center",
                va="bottom" if val >= 0 else "top",
                fontsize=8,
            )
        except Exception:
            continue
    if title:
        ax.set_title(title)
    plt.tight_layout()
    print(f"📁 Saving per-word sentiment bar: {outpath}")
    plt.savefig(outpath, dpi=300)
    plt.close(fig)

# ============= Main analysis =============
def analyze_reports(file_paths: list[Path], output_dir: Path) -> dict:
    sia = SentimentIntensityAnalyzer()
    results = {}
    statewide_tokens = []
    statewide_texts = []
    county_sent_tokens = {}

    for path in file_paths:
        path = Path(path)
        county = infer_county_name(str(path))
        print(f"\n=== Processing {county} ===")

        raw_text = read_any_text(path)
        print(f"[INFO] Extracted {len(raw_text)} characters from {county}.")

        tokens = clean_and_tokenize(raw_text)
        tokens_collapsed = collapse_tokens_to_canon(tokens)
        statewide_tokens.extend(tokens_collapsed)

        sent_score = compute_sentiment_sentence_avg(raw_text, sia)
        sent_label = interpret_sentiment(sent_score)

        top_challenges = extract_policy_themes(tokens, CHALLENGE_TERMS, top_n=5)
        top_context = extract_policy_themes(tokens, CONTEXT_TERMS, top_n=5)

        stats_df = compute_word_stats(tokens_collapsed)
        csv_out = output_dir / f"stats_{county}_collapsed.csv"
        stats_df.head(200).to_csv(csv_out, index=False)
        print(f"[INFO] Saved word frequency table → {csv_out.name}")

        wc_out = output_dir / f"wordcloud_{county}.png"
        build_wordcloud(
            stats_df,
            wc_out,
            title=f"{county} – Collapsed Top Terms (Top-20)",
            top_n=20,
        )

        # Per-word sentence sentiment for Top-20
        word_sent_df = compute_word_level_sentiment(raw_text, stats_df, top_n=20, sia=sia)
        ws_csv = output_dir / f"word_sentiment_{county}.csv"
        word_sent_df.to_csv(ws_csv, index=False)

        ws_png = output_dir / f"word_sentiment_{county}.png"
        plot_word_sentiment_bar(
            word_sent_df,
            ws_png,
            title=f"{county} – Avg sentence sentiment per Top-20 word (VADER)"
        )

        sent_img = output_dir / f"sentiment_{county}.png"
        plot_sentiment_bar(sent_score, sent_img, title=f"{county} – Overall tone (VADER)")

        statewide_texts.append(raw_text)
        county_sent_tokens[county] = build_sentences_for_w2v(raw_text)

        results[county] = {
            "tokens": tokens_collapsed,
            "sentiment_score": sent_score,
            "sentiment_label": sent_label,
            "top_challenges": top_challenges,
            "top_context": top_context,
            "freq_table": stats_df,
        }

    # Statewide aggregate
    statewide_stats = compute_word_stats(statewide_tokens)
    statewide_csv = output_dir / "stats_California_statewide_collapsed.csv"
    statewide_stats.head(200).to_csv(statewide_csv, index=False)

    wc_state = output_dir / "wordcloud_California_statewide.png"
    build_wordcloud(
        statewide_stats,
        wc_state,
        title="California statewide – Collapsed Top Terms (Top-20)",
        top_n=20
    )

    statewide_sent = compute_sentiment_sentence_avg("\n".join(statewide_texts), sia)
    wc_state_sent = output_dir / "sentiment_California_statewide.png"
    plot_sentiment_bar(statewide_sent, wc_state_sent, title="California statewide – Overall tone (VADER)")

    # Optional: Word2Vec theme scores & prompts
    theme_scores, prompts = {}, {}
    if W2V_OK:
        all_sentences = []
        for _, sents in county_sent_tokens.items():
            all_sentences += sents
        model, wv = get_w2v_model(all_sentences)
        if wv is not None:
            tvecs = build_theme_vectors(wv)
            cvecs = {c: document_vector(wv, sents) for c, sents in county_sent_tokens.items()}
            for c, cvec in cvecs.items():
                scores = {t: cosine_similarity(cvec, tvecs[t]) for t in THEMATIC_SEEDS.keys()}
                theme_scores[c] = scores
            if theme_scores:
                pd.DataFrame(theme_scores).T.to_csv(output_dir / "theme_cosine_by_county.csv")
            for c in county_sent_tokens.keys():
                prompts[c] = build_probe_prompt(c, theme_scores, wv, top_k=3)
                with open(output_dir / f"prompt_{c}.txt", "w", encoding="utf-8") as f:
                    f.write(prompts[c])
        else:
            print("⚠️ Word2Vec unavailable (gensim not loaded/trained). Skipping theme scores & prompts.")
    else:
        print("⚠️ gensim not installed. Skipping Word2Vec/theme scoring/prompt generation.")

    return {
        "counties": results,
        "statewide_sentiment": statewide_sent,
        "theme_scores": theme_scores if theme_scores else None,
        "prompts": prompts if prompts else None
    }

# ============= Entry point =============
if __name__ == "__main__":
    try:
        FILE_PATHS = [str(p) for p in pdf_files]
    except NameError:
        raise RuntimeError("Variable 'pdf_files' not defined. Ensure input PDF list is initialized before running.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Launching full analysis pipeline...")
    print(f"Total files to analyze: {len(FILE_PATHS)}")
    print(f"Output directory: {OUTPUT_DIR}\n")

    results = analyze_reports(FILE_PATHS, OUTPUT_DIR)

    print("==== County Summary (Collapsed Tokens) ====")
    for county, info in results["counties"].items():
        print(f"\n[{county}]")
        print(f"Sentiment: {info['sentiment_score']:.3f} → {info['sentiment_label']}")
        print("Top challenge themes:", ", ".join(info.get("top_challenges", [])) or "N/A")
        print("Top context themes:", ", ".join(info.get("top_context", [])) or "N/A")
        print("Top terms (collapsed):")
        freq_table = info.get("freq_table")
        if freq_table is not None and not freq_table.empty:
            for _, row in freq_table.head(5).iterrows():
                print(f" - {row['word']} (freq={row['freq']})")
        else:
            print(" - N/A")

    ss = results.get("statewide_sentiment", None)
    if ss is not None:
        print(f"\nStatewide sentiment (average across counties): {ss:.3f}")

    if results.get("theme_scores"):
        print("\n==== Theme Cosine Similarity (sample) ====")
        sample_c = next(iter(results["theme_scores"].keys()))
        print(f"[{sample_c}]", results["theme_scores"][sample_c])

    if results.get("prompts"):
        print("\n==== Prompt Example ====")
        sample_c = next(iter(results["prompts"].keys()))
        print(f"[{sample_c}]\n{results['prompts'][sample_c]}")

    print("\nAnalysis complete. All outputs saved to:", OUTPUT_DIR)
