"""Hybrid retrieval over Qdrant: dense vectors + BM25, fused with RRF, then reranked.

Each chunk is stored once with two vectors:
  dense - an embedding of its meaning (finds paraphrases: "how fast" ~ "latency")
  bm25  - sparse keyword weights (finds exact terms: "CPHS", "MRR@10", "Table 2")
Qdrant runs both searches and merges them with Reciprocal Rank Fusion, which
combines rankings rather than raw scores, so the two scales never need tuning.
A cross-encoder reranker (rerank.py) then reorders the top candidates.
"""
import logging
import re
import time
import uuid
from functools import lru_cache

from qdrant_client import QdrantClient, models
from qdrant_client.local.qdrant_local import QdrantLocal

from config import get_secret
from embeddings import get_embedder

# Best on the eval set (hybrid: Hit@1 0.91, MRR 0.95 vs 0.76/0.85 for bge-small).
# Free tier: 100 texts/minute, so uploads embed at ~100 chunks per minute.
DEFAULT_EMBED_MODEL = 'gemini-embedding-2'
LIBRARY_OWNER = 'library'   # owner tag of the shared demo documents
MODES = ('dense', 'bm25', 'hybrid')
CANDIDATES = 30   # first-stage results handed to the reranker

# Chunk fields stored in Qdrant; embed_text is only needed at indexing time.
PAYLOAD_FIELDS = ('chunk_id', 'doc_id', 'text', 'source', 'paper', 'section', 'page', 'char_count', 'is_table')


@lru_cache(maxsize=None)
def get_bm25():
    from fastembed import SparseTextEmbedding
    return SparseTextEmbedding('Qdrant/bm25')


def connect(local=False):
    """Qdrant Cloud (persistent library), or an in-memory instance for evaluation."""
    if local:
        return QdrantClient(':memory:')
    return QdrantClient(url=get_secret('QDRANT_URL'), api_key=get_secret('QDRANT_API_KEY'), timeout=60)


def library_collection(embed_model=DEFAULT_EMBED_MODEL):
    # One collection per embedding model: vectors from different models live
    # in different spaces (and sizes) and must never be mixed.
    return 'library-' + re.sub(r'[^a-z0-9]+', '-', embed_model.lower()).strip('-')


def _sparse(embedding):
    return models.SparseVector(indices=embedding.indices.tolist(), values=embedding.values.tolist())


def _filter(owners=None, doc_ids=None):
    conditions = []
    if owners:
        conditions.append(models.FieldCondition(key='owner', match=models.MatchAny(any=list(owners))))
    if doc_ids:
        conditions.append(models.FieldCondition(key='doc_id', match=models.MatchAny(any=list(doc_ids))))
    return models.Filter(must=conditions) if conditions else None


class HybridIndex:
    def __init__(self, client, collection, embed_model=DEFAULT_EMBED_MODEL):
        self.client = client
        self.collection = collection
        self.embedder = get_embedder(embed_model)

    def ensure_collection(self):
        if self.client.collection_exists(self.collection):
            return
        self.client.create_collection(
            self.collection,
            vectors_config={'dense': models.VectorParams(size=self.embedder.dim, distance=models.Distance.COSINE)},
            # IDF modifier: Qdrant computes BM25's inverse document frequency
            # from the collection itself, so it stays right as documents are added.
            sparse_vectors_config={'bm25': models.SparseVectorParams(modifier=models.Modifier.IDF)},
        )
        if not isinstance(self.client._client, QdrantLocal):
            # Indexes let Qdrant Cloud filter by owner/document quickly; the
            # in-memory instance used for evaluation doesn't support them.
            for field in ('owner', 'doc_id'):
                self.client.create_payload_index(self.collection, field, models.PayloadSchemaType.KEYWORD)
            self.client.create_payload_index(self.collection, 'uploaded_at', models.PayloadSchemaType.FLOAT)

    def add(self, chunks, owner=LIBRARY_OWNER, batch_size=64, **extra_payload):
        """Embed and store chunks. owner tags who may search them (see app.py)."""
        self.ensure_collection()
        dense = self.embedder.embed_documents(chunks)
        sparse = list(get_bm25().embed([c.get('embed_text', c['text']) for c in chunks]))
        points = [
            models.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{c['doc_id']}/{c['chunk_id']}")),
                vector={'dense': d.tolist(), 'bm25': _sparse(s)},
                payload={**{k: c[k] for k in PAYLOAD_FIELDS if k in c}, 'owner': owner, **extra_payload},
            )
            for c, d, s in zip(chunks, dense, sparse)
        ]
        for i in range(0, len(points), batch_size):
            self.client.upsert(self.collection, points[i:i + batch_size])

    def has_document(self, doc_id, owners=None):
        if not self.client.collection_exists(self.collection):
            return False
        return self.client.count(self.collection, count_filter=_filter(owners, [doc_id]), exact=True).count > 0

    def documents(self, owners=None):
        """One entry per indexed document: {doc_id, source, paper, owner}."""
        if not self.client.collection_exists(self.collection):
            return []
        docs, offset = {}, None
        while True:
            points, offset = self.client.scroll(self.collection, scroll_filter=_filter(owners), limit=256, offset=offset,
                                                with_payload=['doc_id', 'source', 'paper', 'owner'])
            for p in points:
                docs.setdefault(p.payload['doc_id'], p.payload)
            if offset is None:
                return list(docs.values())

    def delete_document(self, doc_id, owner):
        self._delete(_filter([owner], [doc_id]))

    def delete_expired_uploads(self, max_age_hours=24):
        """Remove visitors' uploads older than max_age_hours; the shared library stays."""
        self._delete(models.Filter(
            must=[models.FieldCondition(key='uploaded_at', range=models.Range(lt=time.time() - max_age_hours * 3600))],
            must_not=[models.FieldCondition(key='owner', match=models.MatchValue(value=LIBRARY_OWNER))],
        ))

    def _delete(self, flt):
        if self.client.collection_exists(self.collection):
            self.client.delete(self.collection, points_selector=models.FilterSelector(filter=flt))

    def search(self, query, k, mode='hybrid', owners=None, doc_ids=None):
        if mode not in MODES:
            raise ValueError(f'unknown mode {mode!r}; choose from {MODES}')
        flt = _filter(owners, doc_ids)
        dense = lambda: models.Prefetch(query=self.embedder.embed_query(query).tolist(), using='dense', limit=k, filter=flt)
        bm25 = lambda: models.Prefetch(query=_sparse(next(get_bm25().query_embed(query))), using='bm25', limit=k, filter=flt)

        if mode == 'hybrid':
            try:
                prefetch = [dense(), bm25()]
            except Exception as e:
                # The query embedding comes from an API with a free-tier quota;
                # keyword search alone still answers most questions.
                logging.warning('Dense search unavailable, using BM25 only: %s', e)
                mode, prefetch = 'bm25', None
        if mode == 'hybrid':
            result = self.client.query_points(self.collection, prefetch=prefetch,
                                              query=models.FusionQuery(fusion=models.Fusion.RRF), limit=k)
        else:
            single = dense() if mode == 'dense' else bm25()
            result = self.client.query_points(self.collection, query=single.query, using=single.using,
                                              query_filter=flt, limit=k)
        return [{**p.payload, 'score': p.score} for p in result.points]


class Retriever:
    """First-stage search, optionally followed by reranking. Call it like a function."""

    def __init__(self, index, mode='hybrid', reranker=None, candidates=CANDIDATES):
        self.index = index
        self.mode = mode
        self.reranker = reranker
        self.candidates = candidates

    def __call__(self, query, k, **filters):
        if not self.reranker:
            return self.index.search(query, k, self.mode, **filters)
        candidates = self.index.search(query, max(k, self.candidates), self.mode, **filters)
        return self.reranker.rerank(query, candidates, k)
