# RAG Research Assistant

A Retrieval-Augmented Generation (RAG) app for **asking plain-English questions across a library of research papers**, including PDFs you upload yourself. Answers are grounded strictly in the documents and cite the **paper, section and page**, so every response is verifiable.

🔗 **Live demo:** deployed on [Streamlit Community Cloud](https://streamlit.io/cloud)

---

## What it does

1. The app ships with a shared library of four retrieval papers (RAG, DPR, ColBERT, BEIR); you can add your own PDF, which only you can search and which is deleted after 24 hours.
2. Each PDF is parsed into markdown (tables included, OCR for scanned pages), split into chunks along the paper's own sections, and stored in Qdrant with two vectors: a dense embedding and BM25 keyword weights.
3. You ask a question; hybrid search finds candidates by meaning *and* by exact terms, and a cross-encoder reranks them.
4. An LLM answers **using only those chunks** and cites its sources. If the answer isn't in the documents, it says so instead of guessing.

---

## How it works

**Indexing (once per document)**
```
PDF → parse to per-page markdown (LlamaParse, or local pymupdf4llm)
    → split at headings, tracking the section path across pages
    → ~192-token chunks tagged with paper, section and page
    → dense vector (gemini-embedding-2) + sparse BM25 vector
    → Qdrant Cloud, tagged with an owner (shared library or one visitor)
```

**Query (per question)**
```
question → dense search + BM25 search in Qdrant (top 30 each)
         → Reciprocal Rank Fusion → Jina reranker → top 5 chunks
         → LLM generates a grounded, cited answer
```

If the embedding API is unavailable, search falls back to BM25 + reranking (still Hit@5 1.00 / MRR 0.93 on the eval set), and if the reranker is unavailable the fused order is kept.

---

## Tech stack

| Component | Choice | Why |
|---|---|---|
| PDF parsing | **LlamaParse** (`llama-cloud`), fallback `pymupdf4llm` | Markdown tables and headings; LlamaParse also OCRs scanned pages |
| Chunking | Heading-aware sections + LlamaIndex `SentenceSplitter` (192 tokens) | Chunks never mix sections; each knows its paper and section for citations |
| Embeddings | **`gemini-embedding-2`** (768-dim) | Best on the eval set; free tier (100 texts/min), every vector cached on disk |
| Keyword search | BM25 via `fastembed`, IDF computed by Qdrant | Finds exact terms (dataset names, metrics, numbers) that embeddings blur |
| Vector database | **Qdrant Cloud** (free tier), hybrid query with RRF fusion | Persistent library; dense + sparse search and fusion in one request |
| Reranker | **Jina `jina-reranker-v3`** | Cross-encoder reads query and chunk together; lifts weaker first stages a lot |
| LLM | `gpt-oss-20b` via **Groq** | Fast inference with a free tier |
| UI | `Streamlit` | Quickest path to a shareable web app |

---

## Project structure

```
├── app.py            # Streamlit UI — library, uploads, questions
├── ingest.py         # PDF parsing (LlamaParse / pymupdf4llm) + section-aware chunking
├── embeddings.py     # Gemini and local embedding models behind one interface, disk-cached
├── retrieval.py      # Qdrant hybrid index (dense + BM25, RRF) and the Retriever
├── rerank.py         # Jina cross-encoder reranking, cached, with fallback
├── rag_app.py        # Prompt construction + LLM answer generation
├── config.py         # API keys from .env or Streamlit secrets
├── index_library.py  # Adds PDFs to the shared library in Qdrant Cloud
├── pipeline.py       # Original pdfplumber + sentence-window pipeline (baseline)
├── evaluate.py       # Retrieval evaluation harness (Hit@k, MRR)
├── eval/
│   ├── testset.json         # labelled questions (paper + page)
│   ├── download_docs.py     # fetches the 4 eval papers
│   ├── generate_testset.py  # drafts questions with an LLM
│   └── review_app.py        # click-through review of drafted questions
├── requirements.txt       # app
└── requirements-dev.txt   # app + local embedding models for evaluation
```

---

## Getting started

### 1. Clone and install
```bash
git clone https://github.com/Udit024-code/Rag_Assistant.git
cd Rag_Assistant
pip install -r requirements-dev.txt
```

### 2. Add your API keys
Create a `.env` file in the project root (all have free tiers):
```
GROQ_API_KEY=...          # console.groq.com
GEMINI_API_KEY=...        # aistudio.google.com
QDRANT_URL=...            # cloud.qdrant.io (free cluster URL, ending :6333)
QDRANT_API_KEY=...
JINA_API_KEY=...          # jina.ai
LLAMA_CLOUD_API_KEY=...   # optional: cloud.llamaindex.ai, for LlamaParse + OCR
```
Without a LlamaCloud key the app parses locally with `pymupdf4llm` (slower, no OCR). The `.env` file is gitignored and never committed.

### 3. Build the shared library and run the app
```bash
python eval/download_docs.py
python index_library.py
streamlit run app.py
```
Then open `http://localhost:8501` and start asking questions.

---

## Deployment (Streamlit Cloud)

1. Push the repo to GitHub.
2. On [share.streamlit.io](https://share.streamlit.io), create a new app pointing at `app.py`.
3. Under **Settings → Secrets**, add every key from `.env` in TOML format, e.g. `GEMINI_API_KEY = "..."`.
4. Deploy. The app reads keys from Streamlit Secrets in the cloud and from `.env` locally, and tells you which are missing.

---

## Evaluation

Retrieval is scored on four public RAG papers (RAG, DPR, ColBERT, BEIR) with a test set in `eval/testset.json`, where each question is labelled with the paper and page that answer it. Every configuration is indexed into an in-memory Qdrant, so evaluation runs the app's own search code.

- **Hit@k**: did a chunk from a correct page appear in the top k results?
- **MRR**: how high the first correct chunk ranks, on average (1.0 = always first).

```bash
python evaluate.py --models BAAI/bge-small-en-v1.5 gemini-embedding-2 --modes dense bm25 hybrid --rerank
python evaluate.py --parser legacy    # the original pipeline
python evaluate.py --answers          # also score generated answers (uses Groq quota)
```

`eval/generate_testset.py` drafts more questions with an LLM; each is marked unverified until reviewed by hand.

| Change | Hit@1 | Hit@5 | MRR |
|---|---|---|---|
| Baseline (MiniLM, pdfplumber defaults) | – | 0.52 | 0.40 |
| Fix PDF word spacing (`x_tolerance=1.5`) | – | 0.78 | 0.65 |
| Switch embeddings to bge-small | – | 0.85 | 0.66 |
| LlamaParse + section-aware 192-token chunks | 0.65 | 0.89 | 0.76 |
| + Jina reranker | 0.85 | 0.96 | 0.91 |
| Hybrid (dense + BM25) + reranker | 0.89 | 1.00 | 0.93 |
| **gemini-embedding-2, hybrid (+ reranker)** | **0.91** | **1.00** | **0.95** |

*Preliminary: 46 LLM-drafted questions, not yet all hand-verified. The top configurations now tie at Hit@5 1.00, so the set is too easy to separate them; harder questions are next.*

Things that did **not** help, kept here because they shaped the design:
- **Bigger chunks** (384-512 tokens): small embedding models squeeze a whole chunk into one vector, and longer chunks blur it (Hit@5 fell to ~0.70).
- **Prepending "paper: ... section: ..." to the embedded text**: lower in all five pairings tested with bge-small; the shared title words make one paper's chunks look alike. Still recorded per chunk for citations and the LLM prompt.
- **Reranking on top of gemini-embedding-2 hybrid**: no gain (MRR 0.95 either way). It is kept as a safety net, because it brings the BM25-only fallback from 0.85 to 0.93.

### Known limitation

Questions comparing two papers ("How does ColBERT's late interaction differ from DPR?") can retrieve chunks from only one of them, and the LLM then fills the gap from its own memory, sometimes wrongly. Splitting such questions into one search per paper, and measuring answer faithfulness, are the next steps.

---

## Possible improvements

- Agentic retrieval: decompose comparison and multi-part questions into sub-queries
- Conversational memory for multi-turn follow-ups, streamed answers
- Inline [1][2] citations linked to the exact passage
- LLM-judged answer faithfulness in the evaluation

---

## License

Released under the MIT License.
