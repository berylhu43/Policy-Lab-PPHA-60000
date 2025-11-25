import os
import json
import re
import shutil
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
JSONL_DIR = os.path.join(BASE_DIR, "..", "data", "dashboard_quant_docs.json")
PERSIST_DIR = os.path.join(BASE_DIR, "..", "embedding", "chroma_jsonl_db[Huggingface Embedding]")
COLLECTION = "dashboard_json"


def build_jsonl_vectorstore(refresh=False):

    embedding_func = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )


    if os.path.exists(PERSIST_DIR) and not refresh:
        print("Loading existing Chroma DB (no rebuild).")
        return Chroma(
            collection_name=COLLECTION,
            persist_directory=PERSIST_DIR,
            embedding_function=embedding_func
        )


    if refresh and os.path.exists(PERSIST_DIR):
        shutil.rmtree(PERSIST_DIR)


    with open(JSONL_DIR, "r") as f:
        quant_docs = json.load(f)

    documents = [
        Document(
            page_content=item["text"],
            metadata=item["metadata"]
        )
        for item in quant_docs
    ]
    print(f"Loaded {len(documents)} quantitative documents for embedding.")

    vectorstore = Chroma(
        collection_name=COLLECTION,
        persist_directory=PERSIST_DIR,
        embedding_function=embedding_func
    )

    texts = [doc.page_content for doc in documents]
    metas = [doc.metadata for doc in documents]

    vectorstore.add_texts(texts, metas)
    vectorstore.persist()

    print("Finished building JSONL vectorstore.")
    return vectorstore

if __name__ == "__main__":
    vs = build_jsonl_vectorstore(refresh=False)
    results = vs.similarity_search("orientation attendance Alameda", k=2)
    for i, r in enumerate(results):
        print(f"\nResult {i+1}: {r.page_content[:300]}")
        print(f"Metadata: {r.metadata}")