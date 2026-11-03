"""Draft extra test questions from the eval PDFs with an LLM, for a human to review.

For a sample of pages in each PDF in eval/docs/, the LLM writes questions that
page answers. Drafts are appended to eval/testset.json with "verified": false;
evaluate.py ignores them until you check each one and flip it to true. The LLM
can write questions that are vague, wrong, or answered on another page too -
reviewing is what makes the test set trustworthy.

Usage (from the project root):
    python eval/generate_testset.py --pages-per-doc 6 --per-page 2
"""
import argparse
import glob
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import extract_text_by_page
from rag_app import client

TESTSET_PATH = os.path.join('eval', 'testset.json')
DOCS_DIR = os.path.join('eval', 'docs')
MIN_PAGE_CHARS = 1500    # skip title pages, figure-only and near-empty pages
MAX_PAGE_CHARS = 6000    # keeps each request well under Groq's free 8K tokens/minute

# Clean word spacing for the LLM, independent of what the pipeline under test uses.
X_TOLERANCE = 1.5

PROMPT = """You are writing an evaluation set for a search system over a collection of research papers.
Below is the text of page {page} of the paper in file "{doc}". Write {n} question(s) a reader might ask that this page answers.

Rules:
- Each question must be answerable from this page alone.
- The collection holds several papers on related topics, so if a question could apply to another paper, name the specific method, model or benchmark so it is unambiguous.
- Paraphrase: do NOT copy distinctive phrases from the text into the question. Real users don't know the paper's wording.
- Never refer to "this page", "the text" or "the paper" - write the question as a standalone query.
- Mix question types: facts and numbers (including from tables), definitions, reasons/explanations, comparisons.
- For each question give a short answer and 2-4 keywords that a correct answer must contain (lowercase, as they appear in the page).

Reply with JSON only: {{"questions": [{{"question": "...", "answer": "...", "keywords": ["..."]}}]}}

PAGE {page} TEXT:
{text}"""


def looks_like_bibliography(text):
    # Reference lists are mostly lines carrying a publication year.
    lines = [l for l in text.splitlines() if l.strip()]
    with_year = sum(1 for l in lines if re.search(r'\b(19|20)\d{2}\b', l))
    return lines and with_year / len(lines) > 0.4


def draft_questions(doc, page_number, text, n, model):
    response = client.with_options(max_retries=8).chat.completions.create(
        model=model,
        messages=[{'role': 'user', 'content': PROMPT.format(doc=doc, page=page_number, n=n, text=text[:MAX_PAGE_CHARS])}],
        response_format={'type': 'json_object'},
        reasoning_effort='low',
        max_completion_tokens=2048,
    )
    return json.loads(response.choices[0].message.content)['questions']


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--pages-per-doc', type=int, default=6, help='pages sampled from each PDF')
    parser.add_argument('--per-page', type=int, default=2, help='questions to draft per page')
    parser.add_argument('--model', default='openai/gpt-oss-120b')
    parser.add_argument('--seed', type=int, default=0, help='page sampling seed, for reproducible drafts')
    args = parser.parse_args()

    with open(TESTSET_PATH, encoding='utf-8') as f:
        testset = json.load(f)
    seen = {t['question'].strip().lower() for t in testset}
    rng = random.Random(args.seed)

    added = 0
    for path in sorted(glob.glob(os.path.join(DOCS_DIR, '*.pdf'))):
        doc = os.path.basename(path)
        pages = [p for p in extract_text_by_page(path, x_tolerance=X_TOLERANCE)
                 if len(p['text']) >= MIN_PAGE_CHARS and not looks_like_bibliography(p['text'])]
        sample = sorted(rng.sample(pages, min(args.pages_per_doc, len(pages))), key=lambda p: p['page_number'])

        for page in sample:
            try:
                drafts = draft_questions(doc, page['page_number'], page['text'], args.per_page, args.model)
            except Exception as e:
                print(f"{doc} p{page['page_number']}: skipped ({e})")
                continue

            for d in drafts:
                if d['question'].strip().lower() in seen:
                    continue
                seen.add(d['question'].strip().lower())
                testset.append({
                    'question': d['question'],
                    'doc': doc,
                    'expected_pages': [page['page_number']],
                    'expected_keywords': d.get('keywords', []),
                    'reference_answer': d.get('answer', ''),
                    'origin': f"llm-draft:{args.model}",
                    'verified': False,
                })
                added += 1
            print(f"{doc} p{page['page_number']}: {len(drafts)} drafted")

            # Save after every page so a rate-limit failure midway loses nothing.
            with open(TESTSET_PATH, 'w', encoding='utf-8') as f:
                json.dump(testset, f, indent=2, ensure_ascii=False)

    print(f'\nAdded {added} unverified questions to {TESTSET_PATH}.')
    print('Review each one: fix or delete bad ones, add any other pages that also answer it, then set "verified": true.')


if __name__ == '__main__':
    main()
