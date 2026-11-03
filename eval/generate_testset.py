"""Draft extra test questions from a PDF with an LLM, for a human to review.

For each page with enough text, the LLM writes questions that page answers.
Drafts are appended to eval/testset.json with "verified": false; evaluate.py
ignores them until you check each one and flip it to true. The LLM can write
questions that are vague, wrong, or answered on another page too - reviewing
is what makes the test set trustworthy.

Usage (from the project root):
    python eval/generate_testset.py --pdf paper.pdf --per-page 2
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import extract_text_by_page
from rag_app import client

TESTSET_PATH = os.path.join('eval', 'testset.json')
MIN_PAGE_CHARS = 400     # skip title pages, reference lists and near-empty pages
MAX_PAGE_CHARS = 6000    # keeps each request well under Groq's free 8K tokens/minute

PROMPT = """You are writing an evaluation set for a search system over a research document.
Below is the text of page {page} of that document. Write {n} question(s) a reader might ask that this page answers.

Rules:
- Each question must be answerable from this page alone, and be specific enough that it isn't answered equally well by most other pages.
- Paraphrase: do NOT copy distinctive phrases from the text into the question. Real users don't know the document's wording.
- Never refer to "this page", "the text" or "the document" - write the question as a standalone query.
- Mix question types: facts and numbers, definitions, reasons/explanations, comparisons.
- For each question give a short answer and 2-4 keywords that a correct answer must contain (lowercase, as they appear in the page).

Reply with JSON only: {{"questions": [{{"question": "...", "answer": "...", "keywords": ["..."]}}]}}

PAGE {page} TEXT:
{text}"""


def draft_questions(page_number, text, n, model):
    response = client.chat.completions.create(
        model=model,
        messages=[{'role': 'user', 'content': PROMPT.format(page=page_number, n=n, text=text[:MAX_PAGE_CHARS])}],
        response_format={'type': 'json_object'},
        reasoning_effort='low',
        max_completion_tokens=2048,
    )
    return json.loads(response.choices[0].message.content)['questions']


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--pdf', required=True)
    parser.add_argument('--per-page', type=int, default=2, help='questions to draft per page')
    parser.add_argument('--model', default='openai/gpt-oss-120b')
    args = parser.parse_args()

    with open(TESTSET_PATH, encoding='utf-8') as f:
        testset = json.load(f)
    seen = {t['question'].strip().lower() for t in testset}

    added = 0
    for page in extract_text_by_page(args.pdf):
        if len(page['text']) < MIN_PAGE_CHARS:
            continue
        try:
            drafts = draft_questions(page['page_number'], page['text'], args.per_page, args.model)
        except Exception as e:
            print(f"page {page['page_number']}: skipped ({e})")
            continue

        for d in drafts:
            if d['question'].strip().lower() in seen:
                continue
            seen.add(d['question'].strip().lower())
            testset.append({
                'question': d['question'],
                'expected_keywords': d.get('keywords', []),
                'expected_pages': [page['page_number']],
                'reference_answer': d.get('answer', ''),
                'source': f"llm-draft:{args.model}",
                'verified': False,
            })
            added += 1
        print(f"page {page['page_number']}: {len(drafts)} drafted")

        # Save after every page so a rate-limit failure midway loses nothing.
        with open(TESTSET_PATH, 'w', encoding='utf-8') as f:
            json.dump(testset, f, indent=2, ensure_ascii=False)

    print(f'\nAdded {added} unverified questions to {TESTSET_PATH}.')
    print('Review each one: fix or delete bad ones, add any other pages that also answer it, then set "verified": true.')


if __name__ == '__main__':
    main()
