import os
import json
import re
from pipeline.extraction_transformation import county_names
from pipeline.load_xl import denormalize_text
from datetime import datetime
from collections import defaultdict
from langchain_community.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains.summarize import load_summarize_chain

from langchain_classic.chains.llm import LLMChain
from langchain_core.documents import Document as LCDocument

from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOpenAI
from langchain_community.utilities import SerpAPIWrapper
from chromadb.config import Settings
import subprocess
import textwrap


# Load API keys
os.environ["OPENAI_API_KEY"] = ""
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
serp = SerpAPIWrapper() if SERPAPI_API_KEY else None

# Configuration
PERSIST_HF = "embedding/chroma_sip_csa_db[Huggingface Embedding]"
PERSIST_JS = "embedding/chroma_jsonl_db[Huggingface Embedding]"
PERSIST_OPENAI = "embedding/chroma_sip_csa_db[openai_embed3]"
COLLECTION_1_NAME = "sip_csa_chunks"
COLLECTION_2_NAME = "dashboard_json"
QUERY_LOG_PATH = "query_log.json"
TOP_K_DEFAULT = 5
MAX_CHAR_LIMIT = 80000

# Prompt template
# we can improve the template
qa_prompt = PromptTemplate(
    input_variables=["context", "question", "external", "user_context"],
    template="""
You are a policy data analyst summarizing county self-assessment reports for the California Department of Social Services (CalWORKs).
Use the information in the **context** to answer the **question** accurately, with correct statistics and clear comparisons.

Guidelines:
1. Follow the task described in the user context (e.g., single-county summary or multi-county comparison).
2. Preserve all numeric values (%, $, counts) exactly as written — do not round or alter them.
3. If comparing counties or statewide data, explicitly state which county has higher or lower values, and quantify the difference if possible.
4. Use short, factual paragraphs or bullet points. 
5. When multiple counties are involved, group findings under clear headers for each county or show a structured comparison table.
6. Do not invent information. If the data is missing or unclear, state that directly.
7. If data from CSA (text) and Dashboard JSON contradict each other, DO NOT merge or average the findings.
Instead, explain that the discrepancy is due to different time ranges and data definitions. Present both trends separately.
8. Dashboard JSON typically covers quantitative trend data from 2018–2023, while CSA text may reference different reporting periods (often only 2019–2021 or a limited quarter range).

Context:
{context}


User Context (instructions and query type):
{user_context}

Question:
{question}

Answer:
"""
)

# Utility functions -----------------------------------------------------------

# need to install ollama in local environment
def start_ollama():
    try:
        subprocess.Popen(
            ['ollama', 'serve'],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
            )
    except Exception as e:
        print(f"❌ Could not start Ollama: {e}")


def clean_text(text: str) -> str:
    return text.encode("utf-8","ignore").decode("utf-8")


def load_log():
    try:
        return json.load(open(QUERY_LOG_PATH))
    except:
        return {}


def save_log(log):
    json.dump(log, open(QUERY_LOG_PATH, "w"), indent=2)


def top_queries(n=10):
    '''Return the n top queries from the query log'''
    log = load_log()
    freqs = defaultdict(int)
    for v in log.values():
        freqs[v.get("query","")] += 1
     # does this break if less than 10 queries? No but the .json file already stored more than 10
     # queries so shouldn't be a problem
    top = sorted(freqs.items(), key=lambda x: -x[1])[:n]
    return "\n".join(f"{i+1}. {q} — {c}x" for i,(q,c) in enumerate(top)) or "No queries."

# Global placeholders
retriever = None
summarizer = None
qa_chain = None
log = load_log()
current_settings = {
    "embed_backend": None,
    "embed_model": None,
    "llm_backend": None,
    "llm_model": None
}

# def merged_search(query, k, retriever_sip, retriever_json):
#     # retrieve from both db
#     docs1 = retriever_sip.invoke(query)
#     docs2 = retriever_json.invoke(query)
#
#     # add source name
#     for d in docs1:
#         d.metadata["source"] = "sip_csa"
#
#     for d in docs2:
#         d.metadata["source"] = "dashboard_json"
#
#     # merge two search result
#     merged = docs1 + docs2
#
#     # return in sorted score
#     merged_sorted = sorted(merged, key=lambda d: d.metadata.get("score", 0), reverse=True)
#
#     # get top k
#     return merged_sorted[:k]

# Initialization function supporting Ollama and OpenAI
def init_engine(embed_backend, embed_model, llm_backend, llm_model):
    '''Initialize engine suppporting Ollama and OpenAI

    Inputs:
        embed_backend (str): which backend to use (MiniLM or OpenAI Embeddings)
        embed_model (str): embed model name
        llm_backend (str): which llm backend to use (OpenAI or Ollama)
        llm_model (str): llm model name

    Returns:
        (None) updates the global variables retriever, summarizer and qa_chain
    '''
    # Update the global version of the variable
    global retriever1, retriever2, summarizer, qa_chain

    # Use MiniLM if avaiable, otherwise use open AI
    if embed_backend == "BAAI":
        embedder = HuggingFaceEmbeddings(
            model_name=embed_model,
            model_kwargs={"device": "cpu"})
    else:
        embedder = OpenAIEmbeddings(
            model=embed_model,
            openai_api_key=OPENAI_API_KEY)

    # Set up retriever to vector database
    store1 = Chroma(
        collection_name=COLLECTION_1_NAME,
        persist_directory=(PERSIST_HF if embed_backend == "BAAI" else PERSIST_OPENAI),
        embedding_function=embedder
)
    store2 = Chroma(
        collection_name=COLLECTION_2_NAME,
        persist_directory=PERSIST_JS,
        embedding_function=embedder
    )

    retriever1 = store1.as_retriever(search_type="similarity",
                                     search_kwargs={"k": TOP_K_DEFAULT})
    retriever2 = store2.as_retriever(search_type="similarity",
                                     search_kwargs={"k": TOP_K_DEFAULT})



    if llm_backend == "OpenAI":
        llm = ChatOpenAI(
            model_name=llm_model,
            temperature=0,
            openai_api_key=OPENAI_API_KEY
            )
    else:
      # temperature 0 so no hallucination
        llm = Ollama(model=llm_model, temperature=0)

    summarizer = load_summarize_chain(llm, chain_type="map_reduce")
    qa_chain = LLMChain(llm=llm, prompt=qa_prompt)


# Document summarization
def summarize_docs(docs):
    """Summaries SIP/CSA + Dashboard JSON documents with correct headers."""
    pages = []

    for i, doc in enumerate(docs):
        meta = doc.metadata
        source = meta.get("source", "")

        if source == "sip_csa":
            header = (
                f"[{i+1}] {meta.get('county', 'Unknown')} | "
                f"Section: {meta.get('section', '?')} | "
                f"Page: {meta.get('page', '?')}"
            )

        elif source == "dashboard_json":
            header = (
                f"[{i+1}] {meta.get('county', 'Unknown')} | "
                f"Indicator: {meta.get('indicator', 'Unknown')} | "
                f"Category: {meta.get('category', 'Unknown')} | "
                f"Subcategory: {meta.get('subcategory', 'Unknown')}"
            )

        else:
            header = f"[{i+1}] {meta.get('county', 'Unknown')} | Unknown Source"

        pages.append(
            LCDocument(page_content=header + "\n" + doc.page_content)
        )

    return summarizer.invoke({"input_documents": pages})["output_text"]



# Main QA function
def ask(
        query,
        k,
        embed_backend,
        embed_model,
        llm_backend,
        llm_model
        ):
    '''Query the LLM using the embed/llm models provided

    Inputs:
        query (str): The question asked of the LLM
        k (int): Number of most probable tokens (for top-k sampling)
        use_ext (bool): Use external query or not
        ext_query (str): External query
        embed_backend (str): embedding backend to use (MiniLM or OpenAI
            Embeddings)
        embed_model (str): embed model name
        llm_backend (str): llm backed to use (Ollama or OpenAI)
        llm_model (str): llm model name

    Returns:
        (tuple) of:
            response (str),
            top queries (list of strings),
            external response (str)
        '''
    global log

    needs_reload = (
            retriever is None or
            embed_backend != current_settings["embed_backend"] or
            embed_model != current_settings["embed_model"] or
            llm_backend != current_settings["llm_backend"] or
            llm_model != current_settings["llm_model"]
    )

    if needs_reload:
        print(f"Reinitializing engine with {llm_backend} - {llm_model}")
        init_engine(embed_backend, embed_model, llm_backend, llm_model)
        current_settings.update({
            "embed_backend": embed_backend,
            "embed_model": embed_model,
            "llm_backend": llm_backend,
            "llm_model": llm_model
        })

    mentioned = [c for c in county_names if re.search(rf"\b{c}\b", query, re.IGNORECASE)]
    if len(mentioned) == 1:
        query_type = "single"
    elif len(mentioned) > 1:
        query_type = "multi"
    else:
        query_type = "unspecified"
    #
    # docs_sip = retriever1.invoke(query)
    # docs_json = retriever2.invoke(query)


    docs_sip_with_score = retriever1.vectorstore.similarity_search_with_score(query, k=k)
    docs_json_with_score = retriever2.vectorstore.similarity_search_with_score(query, k=k)

    query_lower = query.lower()

    QUANT_KEYWORDS = [
        "trend", "trends", "over time", "change", "changes", "rate", "rates",
        "increase", "decrease", "improve", "decline", "growth", "drop"
    ]
    use_quant = any(k in query_lower for k in QUANT_KEYWORDS)

    GROUP_KEYWORDS = {
        "gender": ["gender", "female", "women", "man", "male"],
        "au_type": ["au type", "all other", "two parent", "non moe", "tanf timed_out"],
        "language": ["language", "english", "other language", "spanish"],
        "race": ["race", "white", "asian", "black", "hispanic", "other race", "native american"],
        "total": ["total", "overall", "total"]
    }
    mentioned_group = None
    for group, keys in GROUP_KEYWORDS.items():
        if any(k in query_lower for k in keys):
            mentioned_group = group
            break

    docs_sip = []
    for d, score in docs_sip_with_score:
        d.metadata["_distance"] = score
        docs_sip.append(d)

    docs_json = []
    for d, score in docs_json_with_score:
        if use_quant:
            if mentioned_group:
                if d.metadata.get("category","").lower() == mentioned_group:
                    d.metadata["_distance"] = score * 0.8
                else:
                    d.metadata["_distance"] = score
            else:
                if d.metadata.get("category","").lower() == "total":
                    d.metadata["_distance"] = score * 0.8
        else:
            d.metadata["_distance"] = score

        docs_json.append(d)


    for d in docs_sip:
        d.metadata["source"] = "sip_csa"
    for d in docs_json:
        d.metadata["source"] = "dashboard_json"

    print("\n=== DEBUG JSON FULL METADATA ===")
    for d in docs_json[:3]:
        print(d.metadata)

    print("\n=== DEBUG CSA FULL METADATA ===")
    for d in docs_sip[:3]:
        print(d.metadata)

    merged_docs = docs_sip + docs_json

    def sim(doc):
        return doc.metadata.get("_distance", 999)

    merged_docs = sorted(merged_docs, key=sim)
    docs = merged_docs[:k]


    if query_type == "single":
        county = mentioned[0].lower()
        docs = [d for d in docs if d.metadata.get("county", "").lower() == county]
        summary = summarize_docs(docs)

    elif query_type == "multi":
        docs_by_county = defaultdict(list)
        for d in docs:
            docs_by_county[d.metadata.get("county", "Unknown")].append(d)

        county_summaries = []
        for county, county_docs in docs_by_county.items():
            summary_text = summarize_docs(county_docs)
            county_summaries.append(f"County: {county}\n{summary_text}")
        summary = "\n\n".join(county_summaries)

    else:
        summary = summarize_docs(docs)

    print("\n=== Retrieved Chunks ===")
    for d in docs:
        meta = d.metadata
        if d.metadata["source"] == "dashboard_json":
            print(
                f"sheet={meta.get('indicator')}, "
                f"county={meta.get('county')}, "
                f"category={meta.get('category')}, "
                f"subcategory={meta.get('subcategory')}"
            )
        elif d.metadata["source"] == "sip_csa":
            print(
                f"chunk_id={meta.get('chunk_id')}, "
                f"county={meta.get('county')}, "
                f"section={meta.get('section')}, "
                f"page={meta.get('page')}"
            )
        else:
            print("Unknown source, minimal metadata:")
            print(f"county={meta.get('county')}")

    if not docs:
      # better communication to users?
        return f"No docs found. No relevant sections found for '{query}'. Try being specific", "", ""

    task_instruction = {
        "single": "Summarize the key findings for this county.",
        "multi": "Compare and contrast the findings between the mentioned counties. Summarize the key findings for these counties",
        "unspecified": "Provide a concise summary based on the context."
    }[query_type]


    resp = qa_chain.run({
        "context": clean_text(summary),
        "question": clean_text(query),
        "user_context": f"Query type: {query_type}. {task_instruction} | Counties: {', '.join(mentioned) or 'unspecified'}"
    })
    resp = denormalize_text(resp)

    t = datetime.now().isoformat()
    log[t] = {"query": query}
    save_log(log)

    excerpts = "\n\n---\n\n".join([
        textwrap.dedent
        (f'''[{i+1}] 📍 {
            doc.metadata.get('county', 'Unknown')
            } | {
            doc.metadata.get('report_type', 'Unknown')
            } | Section: {
            doc.metadata.get('section', 'Unknown')
            } | Page {
            doc.metadata.get('page', '?')
            }\n{doc.page_content.strip()}'''
        if doc.metadata["source"] == "sip_csa"
        else
        f'''[{i+1}] 📍 {
            doc.metadata.get('county', 'Unknown')
            } | Cal-OAR Dashboard | Indicator: {
            doc.metadata.get('indicator', 'Unknown')
            } | Category {
            doc.metadata.get('category', 'Unknown')
            } | Subcategory {
            doc.metadata.get('subcategory', 'Unknown')
            } \n{doc.page_content.strip()}'''
    )
        for i, doc in enumerate(docs)
    ])

    excerpts = denormalize_text(excerpts)

    full_response = (
        f"<div style='font-size:18px; line-height:1.6;'>"
        f"{resp.strip()}"
        "<h3>📚 Used Excerpts:</h3>"
        "<pre style='white-space:pre-wrap; font-size:16px;'>"
        f"{excerpts}"
        "</pre>"
        "</div>"
    )

    return full_response
