"""Embedding models behind one interface, with a disk cache.

    embedder = get_embedder('gemini-embedding-2')
    vectors = embedder.embed_documents(chunks)   # (n, dim) float32, unit length
    vector = embedder.embed_query('how does DPR train?')

Local models (sentence-transformers) cost nothing per call. Gemini models are
stronger but the free tier allows 100 texts per minute per model, so every
vector is cached on disk by (model, text) and requests wait out 429s.
"""
import hashlib
import os
import re
import sqlite3
import time
from functools import lru_cache

import numpy as np

from config import get_secret

CACHE_PATH = os.path.join('.cache', 'embeddings.sqlite')

# Some local models are trained with an instruction prepended to the *query*
# (never the documents). Leaving it off quietly costs retrieval quality.
QUERY_PREFIXES = {
    'BAAI/bge-small-en-v1.5': 'Represent this sentence for searching relevant passages: ',
    'BAAI/bge-base-en-v1.5': 'Represent this sentence for searching relevant passages: ',
}

GEMINI_MODELS = ('gemini-embedding-001', 'gemini-embedding-2')
GEMINI_DIM = 768        # Matryoshka-truncated from 3072; a quarter of the storage
GEMINI_BATCH = 50       # texts per request; each text counts against the per-minute quota


class EmbeddingCache:
    def __init__(self, path=CACHE_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute('CREATE TABLE IF NOT EXISTS vectors (key TEXT PRIMARY KEY, vec BLOB)')

    @staticmethod
    def key(model, text):
        return hashlib.sha256(f'{model}\x00{text}'.encode('utf-8')).hexdigest()

    def get_many(self, keys):
        found = {}
        for i in range(0, len(keys), 500):
            batch = keys[i:i + 500]
            rows = self.db.execute(f"SELECT key, vec FROM vectors WHERE key IN ({','.join('?' * len(batch))})", batch)
            found.update({k: np.frombuffer(v, dtype=np.float32) for k, v in rows})
        return found

    def put_many(self, items):
        self.db.executemany('INSERT OR REPLACE INTO vectors VALUES (?, ?)',
                            [(k, v.astype(np.float32).tobytes()) for k, v in items])
        self.db.commit()


def _normalize(vectors):
    vectors = np.asarray(vectors, dtype=np.float32)
    return vectors / np.linalg.norm(vectors, axis=-1, keepdims=True)


class Embedder:
    """Base class: subclasses turn texts into vectors; this handles caching."""
    name = None
    dim = None

    def __init__(self, cache=None):
        self.cache = cache or EmbeddingCache()

    def document_text(self, chunk):
        return chunk.get('embed_text', chunk['text'])

    def query_text(self, query):
        return query

    def _embed(self, texts, is_query):
        raise NotImplementedError

    def _cached(self, texts, is_query):
        keys = [EmbeddingCache.key(f"{self.name}:{'q' if is_query else 'd'}", t) for t in texts]
        found = self.cache.get_many(keys)
        missing = [i for i, k in enumerate(keys) if k not in found]
        if missing:
            new = self._embed([texts[i] for i in missing], is_query)
            self.cache.put_many([(keys[i], v) for i, v in zip(missing, new)])
            found.update({keys[i]: v for i, v in zip(missing, new)})
        return np.stack([found[k] for k in keys])

    def embed_documents(self, chunks):
        return self._cached([self.document_text(c) for c in chunks], is_query=False)

    def embed_query(self, query):
        return self._cached([self.query_text(query)], is_query=True)[0]


class LocalEmbedder(Embedder):
    def __init__(self, name, cache=None):
        super().__init__(cache)
        from sentence_transformers import SentenceTransformer
        self.name = name
        self.model = SentenceTransformer(name)
        self.dim = self.model.get_embedding_dimension()

    def query_text(self, query):
        return QUERY_PREFIXES.get(self.name, '') + query

    def _embed(self, texts, is_query):
        return self.model.encode(texts, normalize_embeddings=True, batch_size=32)


class GeminiEmbedder(Embedder):
    def __init__(self, name, cache=None):
        super().__init__(cache)
        from google import genai
        self.name = name
        self.dim = GEMINI_DIM
        self.client = genai.Client(api_key=get_secret('GEMINI_API_KEY'))

    def document_text(self, chunk):
        if self.name == 'gemini-embedding-2':
            # embedding-2 has no task_type; the task is written into the text.
            return f"title: {chunk.get('paper') or 'none'} | text: {chunk['text']}"
        return chunk['text']

    def query_text(self, query):
        if self.name == 'gemini-embedding-2':
            return f'task: search result | query: {query}'
        return query

    def _request(self, texts, is_query):
        from google.genai import errors, types

        config = types.EmbedContentConfig(output_dimensionality=self.dim)
        if self.name == 'gemini-embedding-001':
            config.task_type = 'RETRIEVAL_QUERY' if is_query else 'RETRIEVAL_DOCUMENT'
        # One Content per text: embedding-2 merges a plain list of strings
        # into a single multimodal input and returns ONE vector for all of them.
        contents = [types.Content(parts=[types.Part(text=t)]) for t in texts]
        for attempt in range(8):
            try:
                result = self.client.models.embed_content(model=self.name, contents=contents, config=config)
                return [e.values for e in result.embeddings]
            except errors.ClientError as e:
                if e.code != 429 or attempt == 7:
                    raise
                # Free tier: 100 texts/minute. The error says how long to wait.
                wait = re.search(r'retry in ([\d.]+)s', str(e))
                time.sleep(float(wait.group(1)) + 1 if wait else 30)

    def _embed(self, texts, is_query):
        vectors = []
        for i in range(0, len(texts), GEMINI_BATCH):
            vectors += self._request(texts[i:i + GEMINI_BATCH], is_query)
        # embedding-001 vectors truncated below 3072 dims aren't unit length.
        return _normalize(vectors)


@lru_cache(maxsize=None)
def get_embedder(name):
    # Loading a model takes seconds, so build each embedder once and reuse it.
    return GeminiEmbedder(name) if name in GEMINI_MODELS else LocalEmbedder(name)
