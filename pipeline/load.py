import os
import re
import shutil
import pandas as pd
import chromadb
import openpyxl
from chromadb.config import Settings
from langchain_core.documents import Document  # Langchain Document storage and retrieval
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
XLSX_PATH = os.path.join(BASE_DIR, "..", "chunked_sip_csa_output.xlsx")
PERSIST_DIR = os.path.join(BASE_DIR, "..", "chroma_sip_csa_db[Huggingface Embedding]")
# PERSIST_DIR = "../chroma_sip_csa_db[Huggingface Embedding]"
COLLECTION  = "sip_csa_chunks"

# Normalize Text
def normalize_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"(\$?\d{1,3}(?:,\d{3})*(?:\.\d+)?%?)", lambda m: f"[NUM:{m.group(1)}]", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"‐|–|—", "-", text)
    text = re.sub(r"“|”|\"|''", '"', text)
    text = re.sub(r"’|‘|`", "'", text)
    return text


def denormalize_text(text: str) -> str:
    """
    Reverses numeric normalization by replacing [NUM:x] placeholders
    with their original numeric values (x).
    Example:
        "The rate is [NUM:15.4%]" → "The rate is 15.4%"
    """
    return re.sub(r"\[NUM:([^\]]+)\]", r"\1", text)

def build_vectorstore(refresh = False):
    # Load Excel Data
    df = pd.read_excel(XLSX_PATH).dropna(subset=["text"])
    df["chunk_id"] = df.apply(
        lambda
            row: f"{row['county'].replace(' ', '')}_{row['report_type'].replace('-', '')}_{row['section'].replace(':', '').replace('.', '').replace(' ', '')}_chunk{row.name}",
        axis=1
    )
    df["text"] = df["text"].apply(normalize_text)
    df["section"] = df["section"].astype(str).apply(normalize_text)

    # Use HuggingFace embeddings
    embedding_func = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # Prepare Documents for LangChain's Chroma.from_documents
    documents = [
        Document(
            page_content=row["text"],
            metadata={
                "county": row["county"],
                "report_type": row["report_type"],
                "section": row["section"],
                "page": row["page"],
                "chunk_id": row["chunk_id"]  # Include chunk_id in metadata
            }
        )
        for _, row in df.iterrows()
    ]

    # refresh toggle which clear the persistence directory if it exists to avoid conflicts

    if refresh and os.path.exists(PERSIST_DIR):
        print(f"Clearing existing Chroma directory: {PERSIST_DIR}")
        shutil.rmtree(PERSIST_DIR)
        print(f"Creating Chroma collection '{COLLECTION}' at '{PERSIST_DIR}'...")
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embedding_func,
            collection_name=COLLECTION,
            persist_directory=PERSIST_DIR
        )
        vectorstore.persist()
        print(f"Chroma collection '{COLLECTION}' created and populated.")
        print(f"Total documents added to collection: {vectorstore._collection.count()}")
    else:
        vectorstore = Chroma(
            collection_name=COLLECTION,
            persist_directory=PERSIST_DIR,
            embedding_function=embedding_func
        )

    # Create retriever
    # retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    # Run a query
    # docs = retriever.invoke("childcare support")
    # for i, doc in enumerate(docs):
    #     print(f"\n--- Document {i + 1} ---")
    #     print(f"Metadata: {doc.metadata}")
    #     print(f"Content: {doc.page_content[:300]}...")

    print("Retriever ready for queries.")
    return vectorstore

if __name__ == "__main__":
    retriever, vectorstore = build_vectorstore(refresh=False)
    try:
        results = vectorstore.similarity_search("childcare support", k=2)
        print("\nSample Query Result:")
        for i, r in enumerate(results):
            print(f"Result {i+1}: {r.page_content[:300]}...\n---")
    except Exception as e:
        print(f"Error during sample query: {e}")


