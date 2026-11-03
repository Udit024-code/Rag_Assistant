"""Evaluation harness for the RAG pipeline.

Measures retrieval quality against a labelled test set (eval/testset.json),
where each question lists the page(s) that contain its answer. Labels are
page-level on purpose: they stay valid when the chunking strategy changes,
so every future pipeline can be scored against the same ground truth.

Usage:
    python evaluate.py --pdf paper.pdf
    python evaluate.py --pdf paper.pdf --models all-MiniLM-L6-v2 BAAI/bge-small-en-v1.5
    python evaluate.py --pdf paper.pdf --answers      # also score generated answers (uses Groq quota)
"""
import argparse
import json
import os
import time
from datetime import datetime

from pipeline import build_pipeline
from retrieval import DEFAULT_EMBED_MODEL, build_index, embed_chunks, get_embed_model, retrieve

TESTSET_PATH = os.path.join('eval', 'testset.json')
RESULTS_DIR = os.path.join('eval', 'results')

KS = (1, 3, 5)  # cut-offs for Hit@k
MAX_K = 10      # depth searched when computing MRR


def load_testset(path=TESTSET_PATH, include_unverified=False):
    with open(path, encoding='utf-8') as f:
        testset = json.load(f)
    if not include_unverified:
        testset = [t for t in testset if t.get('verified', False)]
    return testset


def first_relevant_rank(retrieved_chunks, expected_pages):
    """1-based rank of the first chunk from a correct page, or None if missing."""
    for rank, chunk in enumerate(retrieved_chunks, start=1):
        if chunk['page'] in expected_pages:
            return rank
    return None


def evaluate_retrieval(testset, retrieve_fn):
    """Score any retriever: retrieve_fn(question, k) -> ranked list of chunks.

    Hit@k  - share of questions with a correct page somewhere in the top k.
    MRR    - mean of 1/rank of the first correct chunk (0 if not in top MAX_K);
             rewards putting the right chunk first, not just somewhere.
    """
    per_question = []
    for test in testset:
        results = retrieve_fn(test['question'], MAX_K)
        rank = first_relevant_rank(results, test['expected_pages'])
        per_question.append({
            'question': test['question'],
            'expected_pages': test['expected_pages'],
            'retrieved_pages': [r['page'] for r in results],
            'rank': rank,
        })

    n = len(per_question)
    metrics = {f'hit@{k}': sum(1 for q in per_question if q['rank'] and q['rank'] <= k) / n for k in KS}
    metrics['mrr'] = sum(1 / q['rank'] for q in per_question if q['rank']) / n
    return metrics, per_question


def truncation_rate(chunks, model_name):
    """Share of chunks longer than the model's input limit (the excess is silently dropped)."""
    model = get_embed_model(model_name)
    limit = model.max_seq_length
    too_long = sum(1 for c in chunks if len(model.tokenizer.tokenize(c['text'])) > limit)
    return too_long / len(chunks)


def build_faiss_retriever(chunks, model_name):
    embeddings = embed_chunks(chunks, model_name)
    index = build_index(embeddings)
    return index, (lambda question, k: retrieve(question, index, chunks, model_name, top_k=k))


def evaluate_answers(testset, index, chunks):
    """Keyword check on generated answers: did ANY expected keyword appear?

    A crude baseline - it can't tell a correct answer from one that merely
    mentions the right words. LLM-judged faithfulness replaces it later.
    """
    # Imported here so retrieval-only runs don't need a Groq key.
    from rag_app import rag_pipeline

    hits = 0
    for test in testset:
        answer, _sources = rag_pipeline(test['question'], index, chunks)
        answer = answer.lower()
        if any(kw.lower() in answer for kw in test['expected_keywords']):
            hits += 1
    return hits / len(testset)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--pdf', required=True, help='PDF the test set was written for')
    parser.add_argument('--models', nargs='+', default=[DEFAULT_EMBED_MODEL], help='embedding models to compare')
    parser.add_argument('--chunk-size', type=int, default=5, help='sentences per chunk')
    parser.add_argument('--overlap', type=int, default=1, help='sentences shared between chunks')
    parser.add_argument('--include-unverified', action='store_true', help='also use LLM-drafted questions not yet reviewed')
    parser.add_argument('--answers', action='store_true', help=f'also score generated answers with {DEFAULT_EMBED_MODEL} (uses Groq quota)')
    args = parser.parse_args()

    testset = load_testset(include_unverified=args.include_unverified)
    chunks = build_pipeline(args.pdf, chunk_size=args.chunk_size, overlap=args.overlap)
    print(f'{len(testset)} questions, {len(chunks)} chunks from {os.path.basename(args.pdf)}\n')

    runs = []
    for model_name in args.models:
        start = time.perf_counter()
        index, retrieve_fn = build_faiss_retriever(chunks, model_name)
        index_seconds = time.perf_counter() - start

        metrics, per_question = evaluate_retrieval(testset, retrieve_fn)
        run = {
            'model': model_name,
            **metrics,
            'truncated_chunks': truncation_rate(chunks, model_name),
            'index_seconds': index_seconds,
            'per_question': per_question,
        }
        if args.answers and model_name == DEFAULT_EMBED_MODEL:
            run['answer_keyword_score'] = evaluate_answers(testset, index, chunks)
        runs.append(run)

    header = ['model', *[f'hit@{k}' for k in KS], 'mrr', 'truncated', 'index time']
    print('| ' + ' | '.join(header) + ' |')
    print('|' + '---|' * len(header))
    for run in runs:
        row = [run['model'], *[f"{run[f'hit@{k}']:.2f}" for k in KS], f"{run['mrr']:.2f}",
               f"{run['truncated_chunks']:.0%}", f"{run['index_seconds']:.1f}s"]
        print('| ' + ' | '.join(row) + ' |')

    for run in runs:
        if 'answer_keyword_score' in run:
            print(f"\nAnswer keyword score ({run['model']}): {run['answer_keyword_score']:.0%}")

    # Misses are the most useful output: they show *why* retrieval fails.
    for run in runs:
        misses = [q for q in run['per_question'] if q['rank'] is None]
        if misses:
            print(f"\nMissed in top {MAX_K} by {run['model']}:")
            for q in misses:
                print(f"  - {q['question']}\n    expected pages {q['expected_pages']}, got {q['retrieved_pages']}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, datetime.now().strftime('%Y%m%d-%H%M%S') + '.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({'pdf': os.path.basename(args.pdf), 'chunk_size': args.chunk_size, 'overlap': args.overlap,
                   'questions': len(testset), 'runs': runs}, f, indent=2)
    print(f'\nSaved {out_path}')


if __name__ == '__main__':
    main()
