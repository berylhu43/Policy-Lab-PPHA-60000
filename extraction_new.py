import os
import re
import pymupdf
import pymupdf.layout
import pymupdf4llm
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_DIRECTORY = "/Users/yuxuanhu/gdrive/CalWorks/Vector Database/PDFs"  # Set this to your PDF folder path
OUTPUT_DIRECTORY = os.path.join(BASE_DIR, "..", "test_output.xlsx") # Define output directory

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


def table_to_markdown(table):
    """Convert pymupdf-layout dict-style table to markdown."""
    md = []

    # table["cells"] is a list of rows
    rows = table.get("cells", [])

    for r_idx, row in enumerate(rows):
        cells = []
        for cell in row:
            text = cell.get("text", "").replace("\n", " ").strip()
            cells.append(text if text else " ")
        md.append("| " + " | ".join(cells) + " |")
        if r_idx == 0:
            md.append("|" + " --- |" * len(cells))
    return "\n".join(md)

def extract_sections_with_layout(pdf_path, county, report_type):
    import json
    doc = pymupdf.open(pdf_path)

    raw = pymupdf4llm.to_json(doc, header=False, footer=False)

    # parse json
    data = json.loads(raw) if isinstance(raw, str) else raw

    sections = []
    current_section = None

    for page in data.get("pages", []):
        page_num = page.get("page_number")
        boxes = page.get("boxes", [])

        for box in boxes:

            btype = box.get("boxclass", "")

            # TABLE

            if btype == "table" and box.get("table"):
                if current_section:
                    table_md = box["table"].get("markdown", "")
                    current_section["content"].append({
                        "type": "table",
                        "text": table_md,
                        "page": page_num
                    })
                continue


            textlines = box.get("textlines")

            if textlines:
                text = ""
                for line in box["textlines"]:
                    for span in line.get("spans", []):
                        text += span.get("text", "")
                    text += "\n"
                text = text.strip()

                # SECTION HEADER
                if btype == "section-header":
                    current_section = {
                        "county": county,
                        "report_type": report_type,
                        "section_header": text,
                        "content": []
                    }
                    sections.append(current_section)
                    continue

                #  NORMAL TEXT (paragraph, caption, etc.)
                if btype in ["text", "caption", "list-item", "page-header"]:
                    if current_section:
                        current_section["content"].append({
                            "type": "paragraph",
                            "text": text,
                            "page": page_num
                        })
                    continue

    return sections

def chunk_sections_layout(sections, max_chunk_chars=900):
    chunks = []
    chunk_id = 0

    for sec in sections:
        buf = ""
        pages = set()

        for block in sec["content"]:
            btype = block["type"]
            btext = block["text"]
            bpage = block["page"]

            #  TABLE
            if btype == "table":
                if buf.strip():
                    chunks.append({
                        "county": sec["county"],
                        "report_type": sec["report_type"],
                        "section": sec["section_header"],
                        "chunk_id": f"{sec['section_header']}_chunk{chunk_id}",
                        "type": "text",
                        "text": buf.strip(),
                        "pages": sorted(list(pages))
                    })
                    chunk_id += 1
                    buf = ""
                    pages = set()

                chunks.append({
                    "county": sec["county"],
                    "report_type": sec["report_type"],
                    "section": sec["section_header"],
                    "chunk_id": f"{sec['section_header']}_chunk{chunk_id}",
                    "type": "table",
                    "text": btext,
                    "pages": [bpage]
                })
                chunk_id += 1
                continue

            #  PARAGRAPH
            if btype == "paragraph":
                if len(buf) + len(btext) > max_chunk_chars:
                    chunks.append({
                        "county": sec["county"],
                        "report_type": sec["report_type"],
                        "section": sec["section_header"],
                        "chunk_id": f"{sec['section_header']}_chunk{chunk_id}",
                        "type": "text",
                        "text": buf.strip(),
                        "pages": sorted(list(pages))
                    })
                    chunk_id += 1
                    buf = ""
                    pages = set()

                buf += btext + "\n\n"
                pages.add(bpage)

        #  END OF SECTION
        if buf.strip():
            chunks.append({
                "county": sec["county"],
                "report_type": sec["report_type"],
                "section": sec["section_header"],
                "chunk_id": f"{sec['section_header']}_chunk{chunk_id}",
                "type": "text",
                "text": buf.strip(),
                "pages": sorted(list(pages))
            })
            chunk_id += 1

    return chunks

def process_pdf(pdf_path):
    meta = infer_metadata_from_filename(os.path.basename(pdf_path))
    sections = extract_sections_with_layout(pdf_path, meta["county"], meta["report_type"])
    chunks = chunk_sections_layout(sections)
    return chunks

def save_chunks(chunks, output):
    df = pd.DataFrame(chunks)
    df.to_excel(output, index=False)
    print(f"✅ Saved {len(df)} chunks to {output}")

chunks = process_pdf("/Users/yuxuanhu/gdrive/CalWorks/Vector Database/PDFs/CSA-Summary-Santa-Clara-Fatal-Flaw.pdf")
save_chunks(chunks, OUTPUT_DIRECTORY)