"""LLM-judged answer quality: the Phase 2 single-search pipeline vs the Phase 3 agent.

Retrieval metrics (evaluate.py) say whether the right passages were found;
this checks what users actually read - the answers. Three question types:

  hard           newcomer-phrased questions with a reference answer (testset.json, difficulty "hard")
  comparison     questions needing 2-3 papers (comparisons.json)
  unanswerable   plausible questions the papers don't answer (unanswerable.json)

Systems:
  single  - hybrid search + rerank, top 5, one gpt-oss-20b call (rag_app.rag_pipeline)
  agent   - plan -> retrieve -> check -> answer with gpt-oss-120b (agent.py)

A judge from a different model family than the answer models (Qwen on Groq
by default, so it doesn't grade its own family's writing) splits each answer
into claims and checks every claim against the passages that system
retrieved. One judge grades every answer, so scores are comparable.

  faithfulness      share of claims supported by the retrieved passages
  citation accuracy share of cited claims whose cited passage supports them (agent only)
  correctness       hard questions: matches the reference answer (yes 1, partial 0.5)
  completeness      comparisons: addresses every part (yes 1, partial 0.5)
  declined          unanswerable: says it's not in the documents and invents nothing
  gave up           answerable: answers nothing at all (saying one PART is missing is fine)

Answers (answers.jsonl) and verdicts (verdicts-<judge>.jsonl) are stored as
they finish, so runs stopped by free-tier limits resume where they stopped,
and answers can be re-judged without being regenerated:
    python eval/judge_answers.py                    # generate, judge, report
    python eval/judge_answers.py --redo agent       # regenerate one system's answers
    python eval/judge_answers.py --judge gemini-3.5-flash
    python eval/judge_answers.py --report-only
"""
import argparse
import hashlib
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import llm
from agent import build_agent, run
from config import get_secret
from rag_app import rag_pipeline
from rerank import JinaReranker
from retrieval import DEFAULT_EMBED_MODEL, LIBRARY_OWNER, HybridIndex, Retriever, connect, library_collection

RESULTS_DIR = os.path.join('eval', 'results')
ANSWERS_PATH = os.path.join(RESULTS_DIR, 'answers.jsonl')
DEFAULT_JUDGE = 'qwen/qwen3.8-27b'
SYSTEMS = ('single', 'agent')
PASSAGE_CHARS = 1000   # per passage shown to the judge; keeps a verdict around 3K tokens

JUDGE_PROMPT = """You are grading an answer from a research assistant that must answer ONLY from the retrieved passages below.

QUESTION ({kind}): {question}
{reference}
RETRIEVED PASSAGES (what the assistant saw):
{passages}

ANSWER TO GRADE:
{answer}

Return JSON:
{{"claims": [{{"text": "...", "supported": true, "cited": [1], "citation_ok": true}}],
  "addresses_all_parts": "yes",
  "matches_reference": "n/a",
  "answered_nothing": false}}

- claims: every factual statement in the answer, one per item (skip pleasantries and statements that the documents lack information).
  supported: is it backed by ANY retrieved passage? Judge strictly: true only if a passage states or directly implies it.
  cited: the passage numbers the answer cites for it, e.g. "[2]" -> [2]; [] if none.
  citation_ok: if cited is non-empty, does at least one cited passage support it? null if nothing cited.
- addresses_all_parts: "yes", "partial" or "no" - does the answer address every part of the question (saying a part isn't in the documents counts as addressing it)?
- matches_reference: "yes", "partial", "no" against the reference answer, or "n/a" if there is none.
- answered_nothing: true if the answer gives no substantive information and only says the documents don't contain it."""


def load_questions(n_hard):
    with open(os.path.join('eval', 'testset.json'), encoding='utf-8') as f:
        hard = [t for t in json.load(f) if t.get('difficulty') == 'hard']
    # Spread the sample across papers instead of taking one paper's questions.
    by_doc = {}
    for t in hard:
        by_doc.setdefault(t['doc'], []).append(t)
    picked = []
    while len(picked) < n_hard and any(by_doc.values()):
        for doc in sorted(by_doc):
            if by_doc[doc] and len(picked) < n_hard:
                picked.append(by_doc[doc].pop(0))
    questions = [{'kind': 'hard', 'question': t['question'], 'reference': t.get('reference_answer', '')} for t in picked]
    with open(os.path.join('eval', 'comparisons.json'), encoding='utf-8') as f:
        questions += [{'kind': 'comparison', 'question': q['question']} for q in json.load(f)]
    with open(os.path.join('eval', 'unanswerable.json'), encoding='utf-8') as f:
        questions += [{'kind': 'unanswerable', 'question': q['question']} for q in json.load(f)]
    return questions


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path, item):
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')


def answer_key(item):
    # Verdicts belong to one specific answer: regenerating it invalidates them.
    return hashlib.sha256(f"{item['system']}\x00{item['question']}\x00{item['answer']}".encode()).hexdigest()[:16]


# ---------------------------------------------------------------- generation

def answer_single(retriever, question):
    # rag_app turns API failures into an "API Error" answer; wait out rate limits.
    for attempt in range(6):
        answer, sources = rag_pipeline(question, retriever, k=5, owners=[LIBRARY_OWNER])
        if not answer.startswith('API Error'):
            return answer, sources
        print(f'    answer retry: {answer[:100]}')
        time.sleep(30)
    raise RuntimeError(answer)


def answer_agent(retriever, titles, question):
    final = {}
    for kind, value in run(build_agent(retriever, {'owners': [LIBRARY_OWNER]}, titles), question):
        if kind == 'done':
            final = value
    return final['answer'], final.get('sources', [])


def generate(questions, answers, redo):
    # Evaluate the answer model as designed: wait out rate limits instead of
    # silently falling back to the smaller model.
    llm.ANSWER_MODELS = ['openai/gpt-oss-120b']
    llm.client = llm.client.with_options(max_retries=8)
    index = HybridIndex(connect(), library_collection(DEFAULT_EMBED_MODEL), DEFAULT_EMBED_MODEL)
    retriever = Retriever(index, 'hybrid', JinaReranker(strict=True))
    titles = [d.get('paper') or d['source'] for d in index.documents(owners=[LIBRARY_OWNER])]

    if redo:
        answers = [a for a in answers if a['system'] not in redo]
        with open(ANSWERS_PATH, 'w', encoding='utf-8') as f:
            f.writelines(json.dumps(a, ensure_ascii=False) + '\n' for a in answers)
    have = {(a['system'], a['question']) for a in answers}
    for q in questions:
        for system in SYSTEMS:
            if (system, q['question']) in have:
                continue
            answer, sources = (answer_single(retriever, q['question']) if system == 'single'
                               else answer_agent(retriever, titles, q['question']))
            item = {**q, 'system': system, 'answer': answer,
                    'sources': [{k: s[k] for k in ('source', 'paper', 'page', 'section', 'text') if k in s} for s in sources]}
            answers.append(item)
            append_jsonl(ANSWERS_PATH, item)
            print(f"answered  {system:6} {q['kind']:12} | {q['question'][:70]}")
    return answers


# ---------------------------------------------------------------- judging

def judge(item, judge_model):
    passages = '\n\n'.join(
        f"[{i}] {s.get('paper') or s['source']}, page {s['page']}\n{s['text'][:PASSAGE_CHARS]}"
        for i, s in enumerate(item['sources'], start=1)) or '(none)'
    reference = f"REFERENCE ANSWER: {item['reference']}\n" if item.get('reference') else ''
    prompt = JUDGE_PROMPT.format(kind=item['kind'], question=item['question'], reference=reference,
                                 passages=passages, answer=item['answer'])
    if not judge_model.startswith('gemini'):
        return llm.complete_json([{'role': 'user', 'content': prompt}], models=[judge_model], max_tokens=4096)

    from google import genai
    from google.genai import types
    client = genai.Client(api_key=get_secret('GEMINI_API_KEY'))
    for attempt in range(4):
        try:
            response = client.models.generate_content(
                model=judge_model, contents=prompt,
                config=types.GenerateContentConfig(response_mime_type='application/json', temperature=0))
            return json.loads(response.text)
        except Exception as e:
            # Free-tier Gemini allows ~20 requests/day per model: give up on
            # daily limits rather than waiting hours.
            if attempt == 3 or 'PerDay' in str(e):
                raise
            time.sleep(20)


def score(verdict):
    claims = verdict.get('claims') or []
    cited = [c for c in claims if c.get('cited')]
    level = {'yes': 1.0, 'partial': 0.5, 'no': 0.0}
    return {
        'claims': len(claims),
        'faithfulness': sum(bool(c.get('supported')) for c in claims) / len(claims) if claims else None,
        'citation_accuracy': sum(bool(c.get('citation_ok')) for c in cited) / len(cited) if cited else None,
        'complete': level.get(verdict.get('addresses_all_parts')),
        'correct': level.get(verdict.get('matches_reference')),
        'answered_nothing': bool(verdict.get('answered_nothing')),
        'unsupported_claims': sum(not c.get('supported') for c in claims),
    }


def judge_all(answers, judge_model, verdicts_path):
    judged = {v['key'] for v in read_jsonl(verdicts_path)}
    for item in answers:
        key = answer_key(item)
        if key in judged:
            continue
        verdict = judge(item, judge_model)
        append_jsonl(verdicts_path, {'key': key, 'system': item['system'], 'question': item['question'],
                                     'verdict': verdict, 'score': score(verdict)})
        print(f"judged    {item['system']:6} {item['kind']:12} | {item['question'][:70]}")


# ---------------------------------------------------------------- report

def report(answers, verdicts_path, judge_model):
    scores = {v['key']: v['score'] for v in read_jsonl(verdicts_path)}
    rows = [{**a, 'score': scores[answer_key(a)]} for a in answers if answer_key(a) in scores]

    def mean(values):
        values = [v for v in values if v is not None]
        return f'{sum(values) / len(values):.2f}' if values else '-'

    print(f'\nJudge: {judge_model}, {len(rows)}/{len(answers)} answers judged')
    print('| system | faithfulness | citation accuracy | correctness (hard) | completeness (comparisons) '
          '| declines unanswerable | gives up on answerable |')
    print('|---|---|---|---|---|---|---|')
    for system in SYSTEMS:
        r = [x for x in rows if x['system'] == system]
        answerable = [x for x in r if x['kind'] != 'unanswerable']
        unanswerable = [x for x in r if x['kind'] == 'unanswerable']
        good_declines = sum(x['score']['unsupported_claims'] == 0 for x in unanswerable)
        print(f"| {system} | {mean(x['score']['faithfulness'] for x in answerable)} "
              f"| {mean(x['score']['citation_accuracy'] for x in r) if system == 'agent' else '-'} "
              f"| {mean(x['score']['correct'] for x in r if x['kind'] == 'hard')} "
              f"| {mean(x['score']['complete'] for x in r if x['kind'] == 'comparison')} "
              f"| {good_declines}/{len(unanswerable)} "
              f"| {sum(x['score']['answered_nothing'] for x in answerable)}/{len(answerable)} |")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--hard', type=int, default=10, help='hard questions to include')
    parser.add_argument('--judge', default=DEFAULT_JUDGE, help='Groq model id, or a gemini-* model')
    parser.add_argument('--redo', nargs='+', default=[], choices=SYSTEMS, help='regenerate these systems\' answers')
    parser.add_argument('--report-only', action='store_true')
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    verdicts_path = os.path.join(RESULTS_DIR, f"verdicts-{re.sub(r'[^a-z0-9.]+', '-', args.judge.lower())}.jsonl")
    # Only answer fields: earlier runs stored verdicts inline.
    answers = [{k: v for k, v in a.items() if k not in ('verdict', 'score')} for a in read_jsonl(ANSWERS_PATH)]

    if not args.report_only:
        answers = generate(load_questions(args.hard), answers, args.redo)
        judge_all(answers, args.judge, verdicts_path)
    report(answers, verdicts_path, args.judge)


if __name__ == '__main__':
    main()
