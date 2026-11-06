import math
import os
import tempfile
import time
import uuid

import pymupdf
import streamlit as st

# Import your production pipeline functions
from config import get_secret
from ingest import default_parser, file_digest, ingest_pdf
from agent import CITATION, build_agent, cited_numbers, normalize_citations, run
from rerank import JinaReranker
from retrieval import DEFAULT_EMBED_MODEL, LIBRARY_OWNER, HybridIndex, Retriever, connect, library_collection

# Uploads are parsed and embedded on free tiers (Gemini embeds ~100 chunks a
# minute); this keeps one upload to a few minutes and a fraction of the quota.
MAX_UPLOAD_PAGES = 25

st.set_page_config(page_title="RAG Research Assistant", page_icon="📄", layout="wide")

REQUIRED_SECRETS = ["GROQ_API_KEY", "GEMINI_API_KEY", "QDRANT_URL", "QDRANT_API_KEY", "JINA_API_KEY"]
missing = [name for name in REQUIRED_SECRETS if not get_secret(name)]
if missing:
    st.error(f"Missing API keys: {', '.join(missing)}. Add them to .env locally, "
             "or under Settings → Secrets on Streamlit Cloud.")
    st.stop()


@st.cache_resource
def get_index():
    # Shared across all visitors and reruns: one Qdrant connection per server.
    index = HybridIndex(connect(), library_collection(DEFAULT_EMBED_MODEL), DEFAULT_EMBED_MODEL)
    index.ensure_collection()
    index.delete_expired_uploads()  # runs once per server start
    return index


@st.cache_resource
def get_retriever():
    # Hybrid search (dense + BM25) over Qdrant, then Jina reranking.
    return Retriever(get_index(), mode="hybrid", reranker=JinaReranker())


# --- SESSION STATE SETUP ---
# Each browser session gets its own owner tag, so visitors only ever see the
# shared library plus their own uploads.
if "owner" not in st.session_state:
    st.session_state.owner = f"session-{uuid.uuid4().hex}"
if "indexed_upload" not in st.session_state:
    st.session_state.indexed_upload = None

owner = st.session_state.owner
visible_owners = [LIBRARY_OWNER, owner]
index = get_index()


def add_upload(uploaded_file):
    # Save the uploaded file to a temporary path for the parser to read
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    try:
        with pymupdf.open(tmp_file_path) as pdf:
            if pdf.page_count > MAX_UPLOAD_PAGES:
                st.error(f"Please upload a PDF of at most {MAX_UPLOAD_PAGES} pages (this one has {pdf.page_count}).")
                return

        if index.has_document(file_digest(tmp_file_path), owners=visible_owners):
            st.info(f"{uploaded_file.name} is already in your library.")
            return

        parser = default_parser()
        # The local layout parser takes a few seconds per page on CPU.
        wait_hint = "" if parser == "llamaparse" else " Local parsing takes a few seconds per page."
        with st.spinner(f"Parsing the PDF ({parser})...{wait_hint}"):
            chunks, scanned = ingest_pdf(tmp_file_path, source_name=uploaded_file.name, parser=parser)
        if not chunks:
            st.error("No text could be extracted from this PDF. If it is a scanned document, "
                     "set LLAMA_CLOUD_API_KEY so it can be read with OCR.")
            return
        minutes = math.ceil(len(chunks) / 100)  # free-tier embedding rate
        with st.spinner(f"Embedding {len(chunks)} chunks (about {minutes} min on the free tier)..."):
            index.add(chunks, owner=owner, uploaded_at=time.time())

        st.success(f"Added {uploaded_file.name}: {len(chunks)} chunks.")
        if scanned:
            # Image-only pages have no text layer; only LlamaParse runs OCR.
            st.warning(f"Pages {', '.join(map(str, scanned))} look scanned (no text found) and were skipped. "
                       "Set LLAMA_CLOUD_API_KEY to read them with OCR.")
    finally:
        # Clean up the temporary file so we don't leak storage
        os.remove(tmp_file_path)


# --- SIDEBAR: document library ---
with st.sidebar:
    st.header("Library")
    uploaded_file = st.file_uploader("Add your own PDF", type=["pdf"],
                                     help="Only you can search it; it is deleted after 24 hours.")
    if uploaded_file:
        # Index each upload once, not on every rerun of the script.
        upload_key = (uploaded_file.name, uploaded_file.size)
        if st.session_state.indexed_upload != upload_key:
            add_upload(uploaded_file)
            st.session_state.indexed_upload = upload_key

    documents = sorted(index.documents(owners=visible_owners),
                       key=lambda d: (d["owner"] == LIBRARY_OWNER, d.get("paper") or d["source"]))
    st.caption("Search in:")
    selected = [
        d for d in documents
        if st.checkbox((d.get("paper") or d["source"]) + (" (yours)" if d["owner"] == owner else ""),
                       value=True, key=f"doc-{d['doc_id']}")
    ]

    st.divider()
    if st.button("New chat", width="stretch"):
        st.session_state.messages = []


def link_citations(text, message_index):
    # [3] -> a link that jumps to source 3 of this answer.
    return CITATION.sub(lambda m: f"[[{m.group(1)}]](#src-{message_index}-{m.group(1)})", text)


def render_sources(sources, answer, message_index):
    """Cited sources first, each with an anchor the answer's [n] links point at."""
    cited = [n for n in cited_numbers(answer) if 1 <= n <= len(sources)]
    for n in cited:
        chunk = sources[n - 1]
        section = f" · {chunk['section']}" if chunk.get("section") else ""
        st.markdown(f'<a id="src-{message_index}-{n}"></a>', unsafe_allow_html=True)
        with st.expander(f"[{n}] {chunk.get('paper') or chunk['source']}, page {chunk['page']}{section}"):
            # Chunks are markdown, so tables render as tables.
            st.markdown(chunk["text"])
    others = [(n, c) for n, c in enumerate(sources, start=1) if n not in cited]
    if others:
        with st.expander(f"Other passages searched ({len(others)})"):
            for n, chunk in others:
                st.markdown(f"**[{n}] {chunk.get('paper') or chunk['source']}, page {chunk['page']}**")
                st.markdown(chunk["text"])


# --- MAIN: chat ---
st.title("RAG Research Assistant")
st.caption("Ask questions across the research papers in the library, or add your own PDF in the sidebar. "
           "Answers come only from the documents and cite their sources; follow-up questions work too.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Replay the conversation so far
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            st.caption(message["trace"])
            st.markdown(link_citations(message["content"], i))
            render_sources(message["sources"], message["content"], i)
        else:
            st.markdown(message["content"])

question = st.chat_input("Ask about the papers, e.g. How does ColBERT's late interaction differ from DPR?")
if question:
    if not selected:
        st.warning("Select at least one document in the sidebar.")
        st.stop()

    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        agent = build_agent(get_retriever(),
                            filters={"owners": visible_owners, "doc_ids": [d["doc_id"] for d in selected]},
                            library_titles=[d.get("paper") or d["source"] for d in selected])
        status = st.status("Researching...")
        placeholder = st.empty()
        text, final = "", {}
        try:
            for kind, value in run(agent, question, history):
                if kind == "status":
                    status.update(label=value + "...")
                    status.write(value)
                elif kind == "token":
                    text += value
                    placeholder.markdown(normalize_citations(text) + "▌")
                else:
                    final = value
        except Exception as e:
            status.update(label="Something went wrong", state="error")
            st.error(f"An error occurred during execution: {e}")
            st.stop()

        searches = final.get("searched", [])
        trace = (f"Searched {len(searches)} quer{'y' if len(searches) == 1 else 'ies'}, "
                 f"read {len(final.get('sources', []))} passages" if searches else "No search needed")
        status.update(label=trace, state="complete")
        answer = final.get("answer", normalize_citations(text))
        message_index = len(st.session_state.messages)
        placeholder.markdown(link_citations(answer, message_index))
        render_sources(final.get("sources", []), answer, message_index)
        st.session_state.messages.append({"role": "assistant", "content": answer,
                                          "sources": final.get("sources", []), "trace": trace})
