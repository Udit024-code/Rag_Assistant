"""Do table answers get the numbers right? Old vs table-aware chunking.

Runs the full agent on eval/table_questions.json (each asks for one table
cell with a known value) against two in-memory libraries:

  before - tables chunked like prose (split mid-table, headers and captions
           lost), shown to the LLM as markdown
  after  - tables chunked on their own with caption + header repeated in
           every piece, shown to the LLM as "column: value" records

Scores need no judge: an answer is right if it contains the exact value.
"retrieved" = some retrieved passage contains the value at all.

Answers are appended to eval/results/table_answers.jsonl, so an interrupted
run resumes:
    python eval/table_eval.py
"""
import glob
import json
import os
import sys
import time

import httpx
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import llm
from agent import build_agent, run
from ingest import ingest_corpus
from rerank import JinaReranker
from retrieval import DEFAULT_EMBED_MODEL, HybridIndex, Retriever, connect

QUESTIONS_PATH = os.path.join('eval', 'table_questions.json')
RESULTS_PATH = os.path.join('eval', 'results', 'table_answers.jsonl')
VARIANTS = {'before': {'table_aware': False, 'table_records': False},
            'after': {'table_aware': True, 'table_records': True}}


def contains_value(text, value):
    # Tables write 79169 or 77.8%; answers may say "79,169" or "77.8 %".
    normalize = lambda s: s.replace(',', '').replace(' %', '%')
    return value.rstrip('%') in normalize(text)


def main():
    with open(QUESTIONS_PATH, encoding='utf-8') as f:
        questions = json.load(f)
    done = []
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, encoding='utf-8') as f:
            done = [json.loads(line) for line in f if line.strip()]
    finished = {(r['variant'], r['question']) for r in done}

    # Evaluate the answer model as designed: wait out rate limits instead of
    # silently falling back to the smaller model.
    llm.ANSWER_MODELS = ['openai/gpt-oss-120b']
    llm.client = llm.client.with_options(max_retries=8)
    pdfs = sorted(glob.glob(os.path.join('eval', 'docs', '*.pdf')))
    client = connect(local=True)
    reranker = JinaReranker(strict=True)

    for name, variant in VARIANTS.items():
        index = HybridIndex(client, f'tables-{name}', DEFAULT_EMBED_MODEL)
        chunks = ingest_corpus(pdfs, 'llamaparse', table_aware=variant['table_aware'])
        index.add(chunks)
        titles = sorted({c['paper'] for c in chunks})
        retriever = Retriever(index, 'hybrid', reranker)
        for q in questions:
            if (name, q['question']) in finished:
                continue
            final = {}
            for attempt in range(4):
                try:
                    for kind, value in run(build_agent(retriever, None, titles, table_records=variant['table_records']),
                                           q['question']):
                        if kind == 'done':
                            final = value
                    break
                except (httpx.TransportError, requests.ConnectionError) as e:
                    # Brief network drops shouldn't end a 20-minute run.
                    if attempt == 3:
                        raise
                    print(f'    network error ({type(e).__name__}), retrying in 30s')
                    time.sleep(30)
            row = {'variant': name, 'question': q['question'], 'gold': q['answer'], 'answer': final.get('answer', ''),
                   'correct': contains_value(final.get('answer', ''), q['answer']),
                   'retrieved': any(contains_value(s['text'], q['answer']) for s in final.get('sources', []))}
            done.append(row)
            os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
            with open(RESULTS_PATH, 'a', encoding='utf-8') as f:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
            print(f"{name:6} correct={row['correct']!s:5} retrieved={row['retrieved']!s:5} gold={q['answer']:>8} | {q['question'][:80]}")

    print('\n| variant | answer has the exact value | value was in the retrieved passages |\n|---|---|---|')
    for name in VARIANTS:
        rows = [r for r in done if r['variant'] == name]
        print(f"| {name} | {sum(r['correct'] for r in rows)}/{len(rows)} | {sum(r['retrieved'] for r in rows)}/{len(rows)} |")


if __name__ == '__main__':
    main()
