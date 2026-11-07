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
    → prose: ~192-token chunks; tables: their own chunks, caption + header
      repeated in every piece; all tagged with paper, section and page
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
| Chunking | Heading-aware sections + LlamaIndex `SentenceSplitter` (192 tokens); tables chunked separately (512 tokens) with caption and header in every piece | Chunks never mix sections; each knows its paper and section; no table row is separated from its column names |
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
│   ├── unanswerable.json    # plausible questions the papers don't answer
│   ├── judge_answers.py     # LLM-judged answer quality
│   ├── table_questions.py   # exact-answer questions generated from the papers' tables
│   ├── table_eval.py        # table answers: old vs table-aware chunking
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
python evaluate.py --difficulty hard  # newcomer-phrased questions only
python eval/compare_agent.py          # paper coverage on comparison questions
python eval/judge_answers.py          # LLM-judged answer quality, single search vs agent
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

*Standard set: LLM-drafted questions, not yet all hand-verified. The top configurations tie at Hit@5 1.00, so it can no longer separate them; see the hard set below.*

#### Hard questions: phrased the way a newcomer asks

The standard questions reuse 76% of their answer page's content words (median), which flatters keyword search. `generate_testset.py --hard` asks for newcomer phrasing and keeps only questions sharing at most 50% of their words with the page; even when told to paraphrase, gpt-oss-120b passed that filter 15% of the time and Qwen 52%. On 50 such questions (`--difficulty hard`):

| Config | Hit@1 | Hit@5 | MRR (hard) | MRR (standard) |
|---|---|---|---|---|
| BM25 | 0.32 | 0.68 | 0.48 | 0.85 |
| bge-small dense | 0.40 | 0.72 | 0.54 | 0.76 |
| bge-small hybrid + reranker | 0.68 | 0.98 | 0.81 | 0.93 |
| gemini-embedding-2 dense | **0.76** | 0.94 | **0.85** | 0.92 |
| gemini-embedding-2 hybrid | 0.70 | 0.94 | 0.80 | 0.95 |
| **gemini-embedding-2 hybrid + reranker (the app)** | 0.68 | **1.00** | 0.81 | **0.95** |

Keyword search collapses on paraphrased questions, and on them BM25 even pulls the hybrid's first result down slightly. The app keeps hybrid + reranker because it is the only setup with the right page in the top 5 on both sets (the agent passes several passages to the LLM, so Hit@5 matters more than Hit@1), and because it still works when the embedding API is down. Differences of ~0.04 MRR are about 2 of 50 questions.

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

### Answer quality, judged by an LLM

Retrieval scores don't show what users read. `eval/judge_answers.py` generates answers for 26 questions (10 hard, 10 comparisons, 6 **unanswerable**: plausible questions the papers don't answer, e.g. "What results does ColBERT report on Natural Questions?", where DPR does) and has a judge from another model family (Qwen 3.8 27B; the answers come from gpt-oss) split each answer into claims and check every claim against the passages that system retrieved.

| System | Faithfulness | Citation accuracy | Correct (hard) | Complete (comparisons) | Declines unanswerable | Gives up on answerable |
|---|---|---|---|---|---|---|
| Phase 2: single search, top 5, gpt-oss-20b | 0.91 | – | 0.70 | 0.55 | 5/6 | 5/20 |
| **Phase 3: agent, gpt-oss-120b** | **0.94** | **0.94** | **0.85** | **1.00** | **6/6** | **0/20** |

- *Faithfulness*: share of claims supported by the retrieved passages. *Citation accuracy*: share of cited claims whose cited passage supports them. *Gives up*: answers nothing even though the documents contain the answer (saying one part is missing is fine).
- The two systems differ in both architecture and answer model, so this measures the whole upgrade, not the agent alone.
- Checking the judge: the agent's lowest-scoring answer quoted real numbers under the wrong labels: ColBERT's MS MARCO score (0.401) presented as its BEIR average, and a 0.884 from a 12-column Recall@100 table credited to the wrong model. The judge was right. The cause was chunking: 50 of the 90 chunks holding table rows had been cut off from their header row and 75 from their caption, so the LLM saw grids of numbers without column names or the metric. See the table fix below.
- Building this evaluation found two agent bugs, both fixed: the planner skipped searching for a real question that merely sounded off-topic (one of BEIR's example queries is about Game of Thrones), and gpt-oss could spend its whole token budget reasoning and return empty JSON, which would crash a request.
- Small sample (26 questions × 2 systems) and one judge; useful for direction, not decimals.

### Tables: the fix for misread numbers

Before: tables were chunked like prose, so 50 of the 90 chunks holding table rows had lost their header row and 75 their caption. Now each table is chunked on its own, with its caption and header repeated in every piece (a table continued on the next page inherits them), and the LLM sees table chunks as labelled records instead of a pipe grid:

```
- Dataset = MS MARCO; Lexical BM25: 0.228; ... Late-Interaction ColBERT: 0.401; Re-ranking BM25+CE: 0.413
```

`eval/table_questions.py` generates questions straight from the papers' tables, so every one has an exact answer ("In the ColBERT paper, what is reported for MRR@10 on DeepCT in the table of results on TREC CAR?" → 33.2) and needs no judge. `eval/table_eval.py` runs the full agent on 20 of them:

| Tables | Exact value in the answer | Value in the retrieved passages |
|---|---|---|
| Chunked like prose, shown as markdown | 8/20 | 18/20 |
| **Table-aware chunks, shown as records** | **14/20** | **20/20** |

All six remaining misses trace back to ambiguous generated questions rather than misreading (a classification made after seeing the results): two ask about one of two BEIR tables with near-identical captions, and the agent answered from the other table while naming its metric; three ask about grouped rows ("Single", "Multi", "Gold") that contain several sub-rows; one asks about "Wikipedia", the corpus of several BEIR datasets. On the 14 unambiguous questions the score went from 7/14 to 14/14. A stricter generator should require unique row labels and distinct captions.

---

## Possible improvements

- Table questions with unique row labels and distinct captions, for a cleaner table benchmark
- Hand-verify the test sets (the review app in `eval/review_app.py` makes it a click per question)
- Run the single-search pipeline with gpt-oss-120b too, to separate the agent's effect from the bigger model's
- Weighted fusion that leans on dense search for paraphrased questions

---

## License

Released under the MIT License.
