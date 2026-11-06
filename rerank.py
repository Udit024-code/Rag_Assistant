"""Cross-encoder reranking with Jina's hosted reranker.

First-stage retrieval (dense + BM25) compares a query vector with chunk
vectors computed separately. A cross-encoder reads the query and each chunk
*together*, which is far more accurate but too slow to run over the whole
library - so it reorders only the top ~30 candidates.
"""
import hashlib
import json
import logging
import os
import sqlite3
import time

import requests

from config import get_secret

JINA_URL = 'https://api.jina.ai/v1/rerank'
JINA_MODEL = 'jina-reranker-v3'
CACHE_PATH = os.path.join('.cache', 'rerank.sqlite')


class JinaReranker:
    name = JINA_MODEL

    def __init__(self, model=JINA_MODEL, cache_path=CACHE_PATH, strict=False, max_retries=6):
        # strict: raise instead of falling back. Evaluation uses it, so a
        # rate-limited run can't quietly report un-reranked results as reranked.
        self.model = model
        self.strict = strict
        self.max_retries = max_retries
        self.api_key = get_secret('JINA_API_KEY')
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        self.db = sqlite3.connect(cache_path, check_same_thread=False)
        self.db.execute('CREATE TABLE IF NOT EXISTS scores (key TEXT PRIMARY KEY, scores TEXT)')

    def _scores(self, query, texts):
        key = hashlib.sha256(json.dumps([self.model, query, texts]).encode('utf-8')).hexdigest()
        row = self.db.execute('SELECT scores FROM scores WHERE key = ?', (key,)).fetchone()
        if row:
            return json.loads(row[0])

        for attempt in range(self.max_retries + 1):
            response = requests.post(JINA_URL, timeout=60, headers={'Authorization': f'Bearer {self.api_key}'},
                                     json={'model': self.model, 'query': query, 'documents': texts,
                                           'top_n': len(texts), 'return_documents': False})
            if response.status_code != 429 or attempt == self.max_retries:
                break
            # Free tier limits tokens per minute; wait as told, else back off.
            retry_after = response.headers.get('Retry-After')
            time.sleep(float(retry_after) if retry_after else min(60, 5 * 2 ** attempt))
        response.raise_for_status()
        scores = [0.0] * len(texts)
        for r in response.json()['results']:
            scores[r['index']] = r['relevance_score']

        self.db.execute('INSERT OR REPLACE INTO scores VALUES (?, ?)', (key, json.dumps(scores)))
        self.db.commit()
        return scores

    def rerank(self, query, chunks, top_n):
        """Return the top_n chunks reordered by relevance to the query.

        If the API fails (rate limit, outage, missing key) the first-stage
        order is kept, so answers degrade instead of erroring.
        """
        if not chunks:
            return chunks
        try:
            scores = self._scores(query, [c['text'] for c in chunks])
        except Exception as e:
            if self.strict:
                raise
            logging.warning('Reranking skipped, keeping retrieval order: %s', e)
            return chunks[:top_n]
        ranked = sorted(zip(scores, chunks), key=lambda pair: pair[0], reverse=True)
        return [{**chunk, 'rerank_score': score} for score, chunk in ranked[:top_n]]
