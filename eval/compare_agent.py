"""Does the agent gather evidence from every paper a question needs?

The page-level test set (evaluate.py) only has single-page questions, so it
can't see the failure that motivated the agent: a comparison question whose
single search lands on one paper, leaving the LLM to invent the other half.
This compares, on eval/comparisons.json (questions that need 2-3 papers):

  single@5      - one hybrid + reranked search, top 5 (the Phase 2 app)
  single@10     - the same search, top 10 (the agent's source budget)
  agent         - plan (decompose) -> retrieve -> check, no answer step

Paper coverage = share of the needed papers with at least one passage in the
evidence; "all covered" = every needed paper is present; "weakest paper" =
passages for the least-represented needed paper (evidence balance).

Run from the project root against the shared library in Qdrant Cloud:
    python eval/compare_agent.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import MAX_SOURCES, build_agent, run
from rerank import JinaReranker
from retrieval import DEFAULT_EMBED_MODEL, LIBRARY_OWNER, HybridIndex, Retriever, connect, library_collection

QUESTIONS_PATH = os.path.join('eval', 'comparisons.json')


def coverage(chunks, needed):
    found = {c['source'] for c in chunks}
    return sum(doc in found for doc in needed) / len(needed)


def weakest(chunks, needed):
    return min(sum(c['source'] == doc for c in chunks) for doc in needed)


def main():
    with open(QUESTIONS_PATH, encoding='utf-8') as f:
        questions = json.load(f)

    index = HybridIndex(connect(), library_collection(DEFAULT_EMBED_MODEL), DEFAULT_EMBED_MODEL)
    retriever = Retriever(index, 'hybrid', JinaReranker(strict=True))
    filters = {'owners': [LIBRARY_OWNER]}
    titles = [d.get('paper') or d['source'] for d in index.documents(owners=[LIBRARY_OWNER])]

    rows = []
    for q in questions:
        single = retriever(q['question'], MAX_SOURCES, **filters)
        evidence = {'single@5': single[:5], 'single@10': single}
        final = {}
        for kind, value in run(build_agent(retriever, filters, titles, with_answer=False), q['question']):
            if kind == 'done':
                final = value
        evidence['agent'] = final.get('sources', [])
        rows.append({m: (coverage(c, q['docs']), weakest(c, q['docs'])) for m, c in evidence.items()})
        print(' | '.join(f'{m} {cov:.0%}' for m, (cov, _) in rows[-1].items()), '|', q['question'])
        print('    agent searched:', final.get('searched', []))

    n = len(rows)
    print('\n| method | paper coverage | all papers covered | weakest paper (avg passages) |\n|---|---|---|---|')
    for method in ('single@5', 'single@10', 'agent'):
        cov = [r[method][0] for r in rows]
        print(f"| {method} | {sum(cov) / n:.2f} | {sum(c == 1 for c in cov)}/{n} | "
              f"{sum(r[method][1] for r in rows) / n:.1f} |")


if __name__ == '__main__':
    main()
