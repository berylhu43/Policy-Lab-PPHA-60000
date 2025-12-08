import os
import re
import pymupdf
import pymupdf.layout
import pymupdf4llm
import pandas as pd
import json
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_DIRECTORY = ""  # Set this to your PDF folder path
OUTPUT_DIRECTORY = os.path.join(BASE_DIR, "..", "data", "chunked_sip_output.xlsx") # Define output directory

county_names = [
    "Alameda", "Alpine", "Amador", "Butte", "Calaveras", "Colusa", "Contra Costa",
    "Del Norte", "El Dorado", "Fresno", "Glenn", "Humboldt", "Imperial", "Inyo",
    "Kern", "Kings", "Lake", "Lassen", "Los Angeles", "Madera", "Marin", "Mariposa",
    "Mendocino", "Merced", "Modoc", "Mono", "Monterey", "Napa", "Nevada", "Orange",
    "Placer", "Plumas", "Riverside", "Sacramento", "San Benito", "San Bernardino",
    "San Diego", "San Francisco", "San Joaquin", "San Luis Obispo", "San Mateo",
    "Santa Barbara", "Santa Clara", "Santa Cruz", "Shasta", "Sierra", "Siskiyou",
    "Solano", "Sonoma", "Stanislaus", "Sutter", "Tehama", "Trinity", "Tulare",
    "Tuolumne", "Ventura", "Yolo", "Yuba"
]

# Aliases for counties based on common filename patterns
alias_map = {
    "icdss": "Imperial",
    "lacdss": "Los Angeles",
    "ocss": "Orange",
    "scc": "Santa Clara",
    "sbcs": "San Bernardino",
    # Add more aliases as needed
}

def infer_metadata_from_filename(filename):
    # Clean and normalize
    name_clean = filename.lower().replace("_", " ").replace("-", " ").replace(".pdf", "")
    name_compressed = re.sub(r"\s+", "", name_clean)

    # Report type logic
    report_type = "Unknown" # Initialize report_type
    if "sip_pr" in name_clean:
        report_type = "Cal-SIP-PR"
    elif "sip" in name_clean or "system improvement" in name_clean:
        report_type = "Cal-SIP"
    elif (
        "csa" in name_clean
        or "self-assessment" in name_clean
        or "calworks self-assessment" in name_clean
        or "county self-assessment" in name_clean
        or "fatal flaw" in name_clean
    ):
        report_type = "Cal-CSA"


    # Try full county match
    normalized_counties = [c.lower().replace(" ", "") for c in county_names]
    county = "Unknown"
    for orig, compressed in zip(county_names, normalized_counties):
        if compressed in name_compressed:
            county = orig
            break

    # If not found, try alias map
    if county == "Unknown":
        for alias, full_name in alias_map.items():
            if alias in name_compressed:
                county = full_name
                break

    return {
        "file": filename,
        "county": county,
        "report_type": report_type
    }

def clean_br(s: str) -> str:
    if not s:
        return ""
    return re.sub(r"<[^>]*br[^>]*>", " ", s, flags=re.IGNORECASE)

def table_to_markdown(table):
    """Convert a pymupdf4llm dict-style table to markdown."""
    if not table:
        return ""


    # Prefer built-in markdown
    md = table.get("markdown") if isinstance(table, dict) else None
    if isinstance(md, str) and md.strip():
        return clean_br(md)

    # Fallback: manually build markdown
    rows = table.get("cells") if isinstance(table, dict) else []
    if not rows:
        return ""

    lines = []
    for r_idx, row in enumerate(rows):
        cells = []
        for cell in row:
            if isinstance(cell, dict):
                text = clean_br((cell.get("text") or "").replace("\n", " ").strip())
            elif isinstance(cell, list):
                text = ""
            else:
                text = clean_br(str(cell))
            cells.append(text if text else " ")
        lines.append("| " + " | ".join(cells) + " |")
        if r_idx == 0:
            lines.append("|" + " --- |" * len(cells))

    return "\n".join(lines)

def extract_sections_with_layout(pdf_path, county, report_type):

    doc = pymupdf.open(pdf_path)

    try:
        raw = pymupdf4llm.to_json(doc, header=False, footer=False)
        data = json.loads(raw) if isinstance(raw, str) else raw

    except Exception as e:
        print(f"⚠️ layout failed for {pdf_path} — switching to fallback text mode. Error: {e}")

        data = {"pages": []}
        for i in range(len(doc)):
            page = doc[i]
            txt = page.get_text("text") or ""
            lines = txt.split("\n")
            data["pages"].append({
                "page_number": i + 1,
                "boxes": [{
                    "boxclass": "text",
                    "textlines": [{"spans": [{"text": line}]} for line in lines]
                }]
            })

    sections = []
    current_section = None

    def norm_text(s: str) -> str:
        if not s:
            return ""
        # normalize bullets and whitespace
        s = clean_br(s)
        s = s.replace("\u25cf", "•")
        s = re.sub(r"[\t\r]+", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    SECTION_KEYWORDS = {"introduction", "summary", "executive summary", "demographics"}

    def page_text(page):
        lines = []
        for b in page.get("boxes", []):
            for tl in b.get("textlines", []) or []:
                for sp in tl.get("spans", []) or []:
                    t = sp.get("text")
                    if t: lines.append(t)
        return " ".join(lines)

    for page in data.get("pages", []):
        page_num = page.get("page_number")
        if not page.get("boxes"):
            continue
        text_for_detect = page_text(page).lower()
        if "table of contents" in text_for_detect:
            continue
        boxes = page.get("boxes", [])

        # Ensure we always have a section to attach front-matter
        if current_section is None:
            current_section = {
                "county": county,
                "report_type": report_type,
                "section_header": "Front Matter",
                "start_page": page_num,
                "content": []
            }
            sections.append(current_section)

        for box in boxes:
            btype = (box.get("boxclass") or "").strip()

            # Skip page decorations unless you want them
            if btype in {"page-header", "page-footer"}:
                continue

            # TABLES
            if btype == "table" and box.get("table"):
                table_obj = box["table"]
                table_md = table_obj.get("markdown", "") if isinstance(table_obj, dict) else ""
                if not table_md:
                    table_md = table_to_markdown(table_obj)  # fallback
                table_md = clean_br(table_md)
                current_section["content"].append({
                    "type": "table",
                    "text": table_md,
                    "page": page_num,
                    "bbox": [box.get("x0"), box.get("y0"), box.get("x1"), box.get("y1")]
                })
                continue


            textlines = box.get("textlines") or []
            if textlines:
                text_parts = []
                for line in textlines:
                    for span in line.get("spans", []):
                        t = span.get("text")
                        if t:
                            text_parts.append(t)
                raw_text = " ".join(text_parts)
                text = norm_text(raw_text)
                if not text:
                    continue

                # SECTION HEADER
                if btype == "section-header":
                    lower_text = text.lower()
                    if re.match(r'^\d+(\.\d+)*\.', lower_text) or lower_text in SECTION_KEYWORDS:
                        current_section = {
                            "county": county,
                            "report_type": report_type,
                            "section_header": text,
                            "start_page": page_num,
                            "content": []
                        }
                        sections.append(current_section)
                        continue

                # CAPTIONS / LISTS / PARAGRAPHS
                if btype in {"caption", "list-item", "text", "heading"}:
                    normalized_type = (
                        "list" if btype == "list-item" else
                        "paragraph" if btype in {"text", "heading"} else
                        "caption"
                    )
                    current_section["content"].append({
                        "type": normalized_type,
                        "text": text,
                        "page": page_num,
                        "bbox": [box.get("x0"), box.get("y0"), box.get("x1"), box.get("y1")]
                    })
                    continue

    return sections

def chunk_sections_layout(sections, max_chunk_chars=900):
    import unicodedata

    def slugify(text: str) -> str:
        text = unicodedata.normalize("NFKD", text)
        text = re.sub(r"[^\w\-\s]", "", text)
        text = re.sub(r"\s+", "_", text).strip("_")
        return text[:80] if len(text) > 80 else text

    chunks = []
    chunk_id = 0

    for sec in sections:
        buf = []
        pages = set()

        def flush_text_buf():
            nonlocal chunk_id, buf, pages
            if not buf:
                return
            text = "\n\n".join(buf).strip()
            if not text:
                buf = []
                pages = set()
                return
            chunks.append({
                "county": sec.get("county"),
                "report_type": sec.get("report_type"),
                "section": sec.get("section_header"),
                "chunk_id": f"{slugify(sec.get('section_header','Section'))}_chunk{chunk_id}",
                "type": "text",
                "text": text,
                "pages": sorted(pages)
            })
            chunk_id += 1
            buf = []
            pages = set()

        for block in sec.get("content", []):
            btype = block.get("type")
            btext = block.get("text", "")
            bpage = block.get("page")

            if btype == "table":
                flush_text_buf()
                chunks.append({
                    "county": sec.get("county"),
                    "report_type": sec.get("report_type"),
                    "section": sec.get("section_header"),
                    "chunk_id": f"{slugify(sec.get('section_header','Section'))}_chunk{chunk_id}",
                    "type": btype,
                    "text": btext,
                    "pages": [bpage] if bpage is not None else []
                })
                chunk_id += 1
                continue

            # caption treated as text; keep it but mark inline
            if btype in {"paragraph", "list", "caption"}:
                candidate = ("[CAPTION] " + btext) if btype == "caption" else btext
                # flush if oversize
                if sum(len(x) + 2 for x in buf) + len(candidate) > max_chunk_chars:
                    flush_text_buf()
                buf.append(candidate)
                if bpage is not None:
                    pages.add(bpage)

        flush_text_buf()

    return chunks


def process_pdf(pdf_path):
    meta = infer_metadata_from_filename(os.path.basename(pdf_path))
    try:
        sections = extract_sections_with_layout(pdf_path, meta["county"], meta["report_type"])
    except ValueError as e:
        # skip empty pages
        print(f"reading-order error, skip: {pdf_path} | {e}")
        return []
    return chunk_sections_layout(sections)

def save_chunks(chunks, output):
    df = pd.DataFrame(chunks)
    preferred_cols = [
        "county", "report_type", "section", "chunk_id", "type", "text", "pages"
    ]
    cols = [c for c in preferred_cols if c in df.columns] + [c for c in df.columns if c not in preferred_cols]
    df = df[cols]
    df.to_excel(output, index=False)
    print(f"Saved {len(df)} chunks to {output}")

all_chunks = []
for fname in os.listdir(PDF_DIRECTORY):
    if fname.lower().endswith(".pdf"):
        pdf_path = os.path.join(PDF_DIRECTORY, fname)
        print(f"Processing {pdf_path}")
        chunks = process_pdf(pdf_path)
        all_chunks.extend(chunks)

save_chunks(all_chunks, OUTPUT_DIRECTORY)