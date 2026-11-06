# RAG Research Assistant

A Retrieval-Augmented Generation (RAG) app that lets you **upload a PDF and ask questions about it in plain English**. Answers are grounded strictly in the document's content and include **page citations**, so every response is verifiable.

🔗 **Live demo:** deployed on [Streamlit Community Cloud](https://streamlit.io/cloud)

---

## What it does

1. You upload a research PDF.
2. The app parses it into markdown (tables included, OCR for scanned pages), splits it into chunks along the paper's own sections, and embeds each chunk into a vector index.
3. You ask a question; the app finds the most semantically relevant chunks.
4. An LLM answers **using only those chunks**, and cites the page number. If the answer isn't in the document, it says so instead of guessing.

---

## How it works

The system has two phases:

**Indexing (once per document)**
```
PDF → parse to per-page markdown (LlamaParse, or local pymupdf4llm)
    → split at headings, tracking the section path across pages
    → ~192-token chunks tagged with paper, section and page
    → embed each chunk → FAISS index in memory
```

**Query (per question)**
```
question → embed → FAISS similarity search (top-k) → build augmented prompt
         → LLM generates a grounded, page-cited answer
```

---

## Tech stack

| Component | Choice | Why |
|---|---|---|
| PDF parsing | **LlamaParse** (`llama-cloud`), fallback `pymupdf4llm` | Markdown tables and headings; LlamaParse also OCRs scanned pages |
| Chunking | Heading-aware sections + LlamaIndex `SentenceSplitter` (192 tokens) | Chunks never mix sections; each knows its paper and section for citations |
| Embeddings | `BAAI/bge-small-en-v1.5` (384-dim) | Beat MiniLM on the eval set and reads 512 tokens instead of 256; runs on CPU |
| Vector search | `FAISS` (`IndexFlatIP`, cosine) | Exact nearest-neighbor search; cosine similarity on normalized vectors |
| LLM | `gpt-oss-20b` via **Groq** | Fast inference with a free tier |
| UI | `Streamlit` | Quickest path to a shareable web app |

---

## Project structure

```
├── app.py          # Streamlit UI — upload, index, ask
├── ingest.py       # PDF parsing (LlamaParse / pymupdf4llm) + section-aware chunking
├── pipeline.py     # Original pdfplumber + sentence-window pipeline (baseline)
├── retrieval.py    # Embeddings, FAISS index build, similarity search
├── rag_app.py      # Prompt construction + LLM answer generation
├── config.py       # API keys from .env or Streamlit secrets
├── evaluate.py     # Retrieval evaluation harness (Hit@k, MRR)
├── eval/
│   ├── testset.json         # labelled questions (paper + page)
│   ├── download_docs.py     # fetches the 4 eval papers
│   ├── generate_testset.py  # drafts questions with an LLM
│   └── review_app.py        # click-through review of drafted questions
└── requirements.txt
```

---

## Getting started

### 1. Clone and install
```bash
git clone https://github.com/Udit024-code/Rag_Assistant.git
cd Rag_Assistant
pip install -r requirements.txt
```

### 2. Add your API keys
Create a `.env` file in the project root:
```
GROQ_API_KEY=your_key_here
LLAMA_CLOUD_API_KEY=your_key_here   # optional: LlamaParse parsing + OCR
```
Get free keys at [console.groq.com](https://console.groq.com) and [cloud.llamaindex.ai](https://cloud.llamaindex.ai). Without a LlamaCloud key the app parses locally with `pymupdf4llm` (slower, no OCR). The `.env` file is gitignored and never committed.

### 3. Run the app
```bash
streamlit run app.py
```
Then open `http://localhost:8501`, upload a PDF, and start asking questions.

---

## Deployment (Streamlit Cloud)

1. Push the repo to GitHub.
2. On [share.streamlit.io](https://share.streamlit.io), create a new app pointing at `app.py`.
3. Under **Advanced settings → Secrets**, add your key in TOML format:
   ```toml
   GROQ_API_KEY = "your_key_here"
   ```
4. Deploy. The app reads the key from Streamlit Secrets in the cloud and from `.env` locally.

---

## Evaluation

Retrieval is scored on four public RAG papers (RAG, DPR, ColBERT, BEIR) with a test set in `eval/testset.json`, where each question is labelled with the paper and page that answer it.

- **Hit@k**: did a chunk from a correct page appear in the top k results?
- **MRR**: how high the first correct chunk ranks, on average.
- **Truncated**: share of chunks longer than the embedding model can read.

```bash
python eval/download_docs.py          # fetch the papers (pinned arXiv versions)
python evaluate.py --models all-MiniLM-L6-v2 BAAI/bge-small-en-v1.5
python evaluate.py --answers          # also score generated answers (uses Groq quota)
```

`eval/generate_testset.py` drafts more questions with an LLM; each is marked unverified until reviewed by hand.

| Change | Hit@5 | MRR |
|---|---|---|
| Baseline (MiniLM, pdfplumber defaults) | 0.52 | 0.40 |
| Fix PDF word spacing (`x_tolerance=1.5`) | 0.78 | 0.65 |
| Switch embeddings to bge-small | 0.85 | 0.66 |
| LlamaParse + section-aware 192-token chunks | **0.89** | **0.76** |

*Preliminary: 46 LLM-drafted questions, not yet all hand-verified.*

Things that did **not** help, kept here because they shaped the design:
- **Bigger chunks** (384-512 tokens): small embedding models squeeze a whole chunk into one vector, and longer chunks blur it (Hit@5 fell to ~0.70).
- **Prepending "paper: ... section: ..." to the embedded text**: lower in all five pairings tested; the shared title words make one paper's chunks look alike. Still recorded per chunk for citations and the LLM prompt.

---

## Possible improvements

- A cross-encoder **reranker** for higher retrieval precision
- **Hybrid search** (semantic + keyword/BM25)
- Multi-document support with metadata filtering
- Conversational memory for multi-turn follow-ups
- A persistent vector database (Chroma / Pinecone) instead of rebuilding the index each session

---

## License

Released under the MIT License.
