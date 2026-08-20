# RAG Research Assistant

A Retrieval-Augmented Generation (RAG) app that lets you **upload a PDF and ask questions about it in plain English**. Answers are grounded strictly in the document's content and include **page citations**, so every response is verifiable.

🔗 **Live demo:** deployed on [Streamlit Community Cloud](https://streamlit.io/cloud)

---

## What it does

1. You upload a research PDF.
2. The app extracts the text, splits it into overlapping chunks, and embeds each chunk into a vector index.
3. You ask a question; the app finds the most semantically relevant chunks.
4. An LLM answers **using only those chunks**, and cites the page number. If the answer isn't in the document, it says so instead of guessing.

---

## How it works

The system has two phases:

**Indexing (once per document)**
```
PDF → extract text per page → chunk into overlapping sentence windows
    → embed each chunk → store vectors in FAISS + metadata in a pickle
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
| PDF extraction | `pdfplumber` | Reliable page-level text extraction |
| Chunking | `NLTK` sentence tokenizer | Sentence-aware chunks stay semantically coherent |
| Embeddings | `all-MiniLM-L6-v2` (384-dim) | Small, fast, strong quality/size trade-off; runs on CPU |
| Vector search | `FAISS` (`IndexFlatIP`, cosine) | Exact nearest-neighbor search; cosine similarity on normalized vectors |
| LLM | `Llama 3.1 8B` via **Groq** | Very fast inference with a free tier |
| UI | `Streamlit` | Quickest path to a shareable web app |

---

## Project structure

```
├── app.py          # Streamlit UI — upload, index, ask
├── pipeline.py     # PDF extraction + sentence-window chunking
├── retrieval.py    # Embeddings, FAISS index build, similarity search
├── rag_app.py      # Prompt construction + LLM answer generation
├── evaluate.py     # Retrieval recall & answer-quality evaluation
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

### 2. Add your Groq API key
Create a `.env` file in the project root:
```
GROQ_API_KEY=your_key_here
```
Get a free key at [console.groq.com](https://console.groq.com). The `.env` file is gitignored and never committed.

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

`evaluate.py` measures quality against a test set of 10 questions:

- **Retrieval recall@k** — did the top-k retrieved chunks include a correct page?
- **End-to-end answer quality** — does the generated answer contain the expected keywords?

```bash
python evaluate.py
```

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
