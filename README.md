# RAG Research Assistant

A Retrieval-Augmented Generation (RAG) **research chatbot** for a library of papers, including PDFs you upload yourself. An agent plans the searches each question needs, and answers stream in with **numbered citations** that link to the exact passage (paper, section and page), so every claim is verifiable. Follow-up questions work.

🔗 **Live demo:** deployed on [Streamlit Community Cloud](https://streamlit.io/cloud)

---

## What it does

1. The app ships with a shared library of four retrieval papers (RAG, DPR, ColBERT, BEIR); you can add your own PDF, which only you can search and which is deleted after 24 hours.
2. Each PDF is parsed into markdown (tables included, OCR for scanned pages), split into chunks along the paper's own sections, and stored in Qdrant with two vectors: a dense embedding and BM25 keyword weights.
3. You ask a question in the chat. A LangGraph agent rewrites follow-ups into standalone questions, splits comparisons into one search per item, runs hybrid search (meaning *and* exact terms) with cross-encoder reranking, and checks the evidence covers every part before answering.
4. An LLM answers **using only those passages**, citing them as [1], [2]. If something isn't in the documents, it says so instead of guessing.

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

**Query (per chat message): a LangGraph agent**
```
plan ──► retrieve ──► check ──► answer (streamed, cites [1] [2] ...)
  │          ▲          │
  │          └──────────┘  evidence missing → one more search round
  └──► answer              greetings and thanks skip the search

plan:     rewrite follow-ups ("and BM25?") into a standalone question;
          split comparisons / multi-part questions into one search each
retrieve: per search: dense + BM25 in Qdrant (top 30 each) → RRF fusion
          → Jina reranker → top 4; results interleaved, up to 10 passages
check:    does the evidence cover every part? if not, which searches would
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
| Agent | **LangGraph** state machine (plan → retrieve → check → answer) | Explicit, inspectable control flow; streams progress and tokens to the UI |
| LLM | `gpt-oss-120b` for answers, `gpt-oss-20b` for planning, via **Groq** | Free tier; each model has its own rate limit, so either falls back to the other |
| UI | `Streamlit` | Quickest path to a shareable web app |

---

## Project structure

```
├── app.py            # Streamlit chat UI — library, uploads, streamed cited answers
├── agent.py          # LangGraph research agent: plan, retrieve, check, answer
├── llm.py            # Groq calls with model fallback (JSON plans, streamed answers)
├── ingest.py         # PDF parsing (LlamaParse / pymupdf4llm) + section-aware chunking
├── embeddings.py     # Gemini and local embedding models behind one interface, disk-cached
├── retrieval.py      # Qdrant hybrid index (dense + BM25, RRF) and the Retriever
├── rerank.py         # Jina cross-encoder reranking, cached, with fallback
├── rag_app.py        # Single-search RAG pipeline (used by evaluate.py --answers)
├── config.py         # API keys from .env or Streamlit secrets
├── index_library.py  # Adds PDFs to the shared library in Qdrant Cloud
├── pipeline.py       # Original pdfplumber + sentence-window pipeline (baseline)
├── evaluate.py       # Retrieval evaluation harness (Hit@k, MRR)
├── eval/
│   ├── testset.json         # labelled questions (paper + page)
│   ├── comparisons.json     # questions needing evidence from 2-3 papers
│   ├── compare_agent.py     # paper coverage: single search vs the agent
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

### Comparison questions: single search vs the agent

The test set above only has single-page questions. `eval/compare_agent.py` scores 10 questions that each need evidence from 2-3 papers ("How does ColBERT's late interaction differ from DPR?"):

| Method | All needed papers in the evidence | Passages for the least-covered paper |
|---|---|---|
| Single search, top 5 (Phase 2 app) | 4/10 | 0.6 |
| Single search, top 10 | 10/10 | 2.3 |
| **Agent** (decomposed searches, up to 10 passages) | **10/10** | **3.7** |

The Phase 2 app answered "ColBERT vs DPR" from ColBERT passages alone and made up DPR's half; most of that was the top-5 cut. At an equal 10-passage budget, single search already reaches every paper, and the agent's gain is balance: ~60% more evidence for the weakest side of each comparison. Its other benefits (follow-up rewriting, skipping search for small talk, a second round when evidence is missing) cost two small LLM calls per question.

---

## Possible improvements

- LLM-judged answer faithfulness and citation accuracy in the evaluation
- Harder, hand-verified questions (the current set saturates at Hit@5 1.00)
- Highlighting the exact sentence each citation supports

---

## License

Released under the MIT License.
