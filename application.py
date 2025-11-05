import os
import json
import re
from pipeline.extraction_transformation import county_names
from pipeline.load import denormalize_text
from datetime import datetime
from collections import defaultdict
from langchain_community.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains.summarize import load_summarize_chain

from langchain_classic.chains.llm import LLMChain
from langchain_community.utilities import SerpAPIWrapper
from langchain_core.documents import Document as LCDocument

from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOpenAI
from langchain_community.utilities import SerpAPIWrapper
from chromadb.config import Settings
import subprocess


# Load API keys
os.environ["OPENAI_API_KEY"] = ""
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
serp = SerpAPIWrapper() if SERPAPI_API_KEY else None

# Configuration
PERSIST_HF = "chroma_sip_csa_db[Huggingface Embedding]"
PERSIST_OPENAI = "chroma_sip_csa_db[openai_embed3]"
COLLECTION_NAME = "sip_csa_chunks"
QUERY_LOG_PATH = "query_log.json"
TOP_K_DEFAULT = 5
MAX_CHAR_LIMIT = 80000

# Prompt template
# we can imporve the template
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

Context:
{context}

External Info (if available):
{external}

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
    global retriever, summarizer, qa_chain

    # Use MiniLM if avaiable, otherwise use open AI
    if embed_backend == "MiniLM":
        embedder = HuggingFaceEmbeddings(
            model_name=embed_model,
            model_kwargs={"device": "cpu"})
    else:
        embedder = OpenAIEmbeddings(
            model=embed_model,
            openai_api_key=OPENAI_API_KEY)

    # Set up retriever to vector database
    store = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=(PERSIST_HF if embed_backend == "MiniLM" else PERSIST_OPENAI),
        embedding_function=embedder
)
    retriever = store.as_retriever()

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
    # splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    # summarizer = create_map_reduce_chain(llm, text_splitter=splitter)
    qa_chain = LLMChain(llm=llm, prompt=qa_prompt)


# Document summarization

def summarize_docs(docs):
    """For list of documents, summarize them into one single summary.

    Inputs:
        docs (list): List of documents to summarize

    Returns (str) summary
    """
    pages = []
    for i, doc in enumerate(docs):
        meta = doc.metadata
        header = f'''[{i+1}] {
            meta.get('county','Unknown')
            } | Section: {
            meta.get('section','?')
            }'''
        pages.append(LCDocument(page_content=header+"\n"+doc.page_content))
    return summarizer.invoke({"input_documents": pages})["output_text"]


# File analysis (text only)

def extract_text_from_file(path):
    return open(path, 'r').read()


def analyze_file(file_path, query=""):
    '''Analyze the file (currently only returns the text in the file?)'''
    # i think this function only return the first 80000 characters now
    # definitely need to be improved
    text = extract_text_from_file(file_path)
    return text[:MAX_CHAR_LIMIT]


# Main QA function
def ask(
        query,
        k,
        use_ext,
        ext_query,
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

    retriever.search_kwargs = {"k": k}

    if query_type == "single":
        county = mentioned[0]
        filter_fn = lambda doc: doc.metadata.get("county", "").lower() == county.lower()
        docs = [d for d in retriever.invoke(query) if filter_fn(d)]
        summary = summarize_docs(docs)

    elif query_type == "multi":
        docs = retriever.invoke(query)
        # Optional: group by county for later structured summary
        docs_by_county = defaultdict(list)
        for d in docs:
            docs_by_county[d.metadata.get("county", "Unknown")].append(d)
        # For each county, summarize its documents individually
        county_summaries = []
        for county, county_docs in docs_by_county.items():
            summary_text = summarize_docs(county_docs)
            county_summaries.append(f"County: {county}\n{summary_text}")
        # Concatenate the county summaries into a structured context string
        summary = "\n\n".join(county_summaries)
    else:
        docs = retriever.invoke(query)
        summary = summarize_docs(docs)


    if not docs:
      # better communication to users?
        return f"No docs found. No relevant sections found for '{query}'. Try being specific", "", ""

    task_instruction = {
        "single": "Summarize the key findings for this county and note comparisons with statewide averages.",
        "multi": "Compare and contrast the findings between the mentioned counties. Summarize the key findings for these counties",
        "unspecified": "Provide a concise summary based on the context."
    }[query_type]


    external = ""
    if use_ext:
        if not serp:
            external = "[Web search disabled]"
        elif ext_query:
            try:
                external = serp.run(ext_query)
            except Exception as e:
                external = f"[Web search error: {e}]"

    resp = qa_chain.run({
        "context": clean_text(summary),
        "question": clean_text(query),
        "external": clean_text(external),
        "user_context": f"Query type: {query_type}. {task_instruction} | Counties: {', '.join(mentioned) or 'unspecified'}"
    })
    resp = denormalize_text(resp)

    t = datetime.now().isoformat()
    log[t] = {"query": query}
    save_log(log)

    excerpts = "\n\n---\n\n".join([
        f'''[{i+1}] 📍 {
            doc.metadata.get('county', 'Unknown')
            } | {
            doc.metadata.get('report_type', 'Unknown')
            } | Section: {
            doc.metadata.get('section', 'Unknown')
            } | Page {
            doc.metadata.get('page', '?')
            }\n{doc.page_content.strip()}'''
        for i, doc in enumerate(docs)
    ])

    full_response = resp.strip() + "\n\n📚 Used Excerpts:\n\n" + excerpts
    return full_response, top_queries(), external.strip()
