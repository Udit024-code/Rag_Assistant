"""Download the arXiv papers the eval test set is written for into eval/docs/.

The PDFs aren't committed to the repo; run this once before evaluate.py:
    python eval/download_docs.py
"""
import os
import urllib.request

DOCS_DIR = os.path.join('eval', 'docs')

# filename -> versioned arXiv id. Filenames must match the "doc" field in
# testset.json, and versions are pinned so its page numbers stay correct.
PAPERS = {
    'rag_lewis2020.pdf': '2005.11401v4',        # Retrieval-Augmented Generation (Lewis et al.)
    'dpr_karpukhin2020.pdf': '2004.04906v3',    # Dense Passage Retrieval (Karpukhin et al.)
    'colbert_khattab2020.pdf': '2004.12832v2',  # ColBERT (Khattab & Zaharia)
    'beir_thakur2021.pdf': '2104.08663v4',      # BEIR benchmark (Thakur et al.)
}

if __name__ == '__main__':
    os.makedirs(DOCS_DIR, exist_ok=True)
    for filename, arxiv_id in PAPERS.items():
        path = os.path.join(DOCS_DIR, filename)
        if os.path.exists(path):
            print(f'{filename}: already present')
            continue
        urllib.request.urlretrieve(f'https://arxiv.org/pdf/{arxiv_id}', path)
        print(f'{filename}: downloaded')
