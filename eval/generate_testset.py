"""Draft extra test questions from the eval PDFs with an LLM, for a human to review.

For a sample of pages in each PDF in eval/docs/, the LLM writes questions that
page answers. Drafts are appended to eval/testset.json with "verified": false;
evaluate.py ignores them until you check each one and flip it to true. The LLM
can write questions that are vague, wrong, or answered on another page too -
reviewing is what makes the test set trustworthy.

--hard asks for questions in a newcomer's own words and keeps only those
sharing at most --max-overlap of their content words with the answer page.
The default drafts reuse ~76% of the page's words (median), which makes keyword
search look better than it is with real users.

Usage (from the project root):
    python eval/generate_testset.py --pages-per-doc 6 --per-page 2
    python eval/generate_testset.py --hard --seed 1
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


HARD_PROMPT = """You are writing a HARD evaluation set for a search system over a collection of research papers.
Below is the text of page {page} of the paper in file "{doc}". Write {n} question(s) this page answers, phrased the way a curious newcomer would ask them: someone who understands the idea but has never read the paper.

Rules:
- Each question must be answerable from this page alone.
- Describe concepts in your OWN everyday words. Avoid the paper's technical terms, section titles and distinctive phrases; only names of methods, models, datasets or papers may be kept (needed to make the question unambiguous in a collection of related papers).
- Prefer questions about why/how something works, what a result implies, or what a number in a table means, over copying a sentence into a question.
- Never refer to "this page", "the text" or "the paper".
- For each question give a short answer and 2-4 keywords that a correct answer must contain (lowercase, as they appear in the page).

Reply with JSON only: {{"questions": [{{"question": "...", "answer": "...", "keywords": ["..."]}}]}}

PAGE {page} TEXT:
{text}"""

STOPWORDS = set("""the a an of to in on for and or is are was were be been by with as at from that this which what
how does do did its it their they them than into about can could would should has have had per each when where who
whom why between within using used use based also not no more most other such these those there over under via vs
paper method model approach""".split())


def content_words(text):
    return {w for w in re.findall(r'[a-z0-9]+', text.lower()) if len(w) > 2 and w not in STOPWORDS}


def word_overlap(question, page_text):
    """Share of the question's content words that appear on the page (1.0 = all copied)."""
    words = content_words(question)
    return len(words & content_words(page_text)) / len(words) if words else 1.0


BIBLIOGRAPHY_LINE = re.compile(r'\b(19|20)\d{2}\b|arxiv|proceedings|et al\.|conference|journal|association for|in advances',
                               re.IGNORECASE)


def looks_like_bibliography(text):
    # Reference-list lines carry years, venues and "et al."; content pages
    # score <= 0.1, reference pages 0.4-0.9 on the eval papers.
    lines = [l for l in text.splitlines() if l.strip()]
    return lines and sum(1 for l in lines if BIBLIOGRAPHY_LINE.search(l)) / len(lines) > 0.35


def draft_questions(doc, page_number, text, n, model, prompt=PROMPT):
    response = client.with_options(max_retries=8).chat.completions.create(
        model=model,
        messages=[{'role': 'user', 'content': prompt.format(doc=doc, page=page_number, n=n, text=text[:MAX_PAGE_CHARS])}],
        response_format={'type': 'json_object'},
        # gpt-oss takes low/medium/high; Groq's Qwen models take none/default.
        reasoning_effort='low' if 'gpt-oss' in model else 'none',
        max_completion_tokens=2048,
    )
    return json.loads(response.choices[0].message.content)['questions']


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--pages-per-doc', type=int, default=6, help='pages sampled from each PDF')
    parser.add_argument('--per-page', type=int, default=2, help='questions to draft per page')
    parser.add_argument('--model', default='openai/gpt-oss-120b')
    parser.add_argument('--seed', type=int, default=0, help='page sampling seed, for reproducible drafts')
    parser.add_argument('--hard', action='store_true', help='newcomer phrasing, filtered by word overlap')
    parser.add_argument('--max-overlap', type=float, default=0.5, help='--hard only: max share of words shared with the page')
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
                drafts = draft_questions(doc, page['page_number'], page['text'], args.per_page, args.model,
                                         HARD_PROMPT if args.hard else PROMPT)
            except Exception as e:
                print(f"{doc} p{page['page_number']}: skipped ({e})")
                continue

            kept = 0
            for d in drafts:
                overlap = word_overlap(d['question'], page['text'])
                if d['question'].strip().lower() in seen or (args.hard and overlap > args.max_overlap):
                    continue
                seen.add(d['question'].strip().lower())
                testset.append({
                    'question': d['question'],
                    'doc': doc,
                    'expected_pages': [page['page_number']],
                    'expected_keywords': d.get('keywords', []),
                    'reference_answer': d.get('answer', ''),
                    'difficulty': 'hard' if args.hard else 'standard',
                    'word_overlap': round(overlap, 2),
                    'origin': f"llm-draft:{args.model}",
                    'verified': False,
                })
                added += 1
                kept += 1
            print(f"{doc} p{page['page_number']}: {len(drafts)} drafted, {kept} kept")

            # Save after every page so a rate-limit failure midway loses nothing.
            with open(TESTSET_PATH, 'w', encoding='utf-8') as f:
                json.dump(testset, f, indent=2, ensure_ascii=False)

    print(f'\nAdded {added} unverified questions to {TESTSET_PATH}.')
    print('Review each one: fix or delete bad ones, add any other pages that also answer it, then set "verified": true.')


if __name__ == '__main__':
    main()
