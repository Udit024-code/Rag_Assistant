"""Add PDFs to the shared demo library in Qdrant Cloud.

Every visitor of the app can search these. Run once, and again after changing
DEFAULT_EMBED_MODEL (each model gets its own collection):
    python index_library.py                  # the four eval papers
    python index_library.py paper1.pdf ...   # specific PDFs
"""
import glob
import os
import sys

from ingest import ingest_pdf
from retrieval import DEFAULT_EMBED_MODEL, LIBRARY_OWNER, HybridIndex, connect, library_collection

if __name__ == '__main__':
    paths = sys.argv[1:] or sorted(glob.glob(os.path.join('eval', 'docs', '*.pdf')))
    index = HybridIndex(connect(), library_collection(DEFAULT_EMBED_MODEL), DEFAULT_EMBED_MODEL)
    print(f'Collection {index.collection} ({DEFAULT_EMBED_MODEL})')
    for path in paths:
        chunks, scanned = ingest_pdf(path)
        if index.has_document(chunks[0]['doc_id'], owners=[LIBRARY_OWNER]):
            print(f'{os.path.basename(path)}: already indexed')
            continue
        index.add(chunks)
        print(f'{os.path.basename(path)}: {len(chunks)} chunks added' + (f', scanned pages {scanned}' if scanned else ''))
