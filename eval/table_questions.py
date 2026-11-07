"""Generate table-lookup questions with exact answers from the papers' own tables.

Every question asks for one cell ("what is the <caption metric> for <column>
on <row>?"), so it has a single correct value and needs no judge: an answer
is right if it contains that value. Cells are only used when their value is
unique within the table, so quoting a neighbouring cell can't pass by luck.

    python eval/table_questions.py          # writes eval/table_questions.json
"""
import glob
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingest import clean_inline, parse_pdf

OUT_PATH = os.path.join('eval', 'table_questions.json')
PAPER_NAMES = {'beir': 'BEIR', 'colbert': 'ColBERT', 'dpr': 'DPR (Dense Passage Retrieval)', 'rag': 'RAG (Retrieval-Augmented Generation)'}
CAPTION = re.compile(r'^\W*(Table\s+\d+)\s*[:.]\s*(.+)', re.IGNORECASE)
SEPARATOR = re.compile(r'^\|(\s*:?-+:?\s*\|)+\s*$')
NUMBER = re.compile(r'^-?\d+(\.\d+)?%?$')
PER_TABLE = 2
TOTAL = 20


def cells(row):
    # Drop markdown escapes (a backslash before @, _ etc.) and in-cell line breaks.
    return [clean_inline(c.replace('<br/>', ' ').replace('<br>', ' ')).replace(chr(92), '')
            for c in row.strip().strip('|').split('|')]


def tables_on_page(markdown):
    """Yield (caption, header, rows) for each markdown table with a nearby "Table N:" caption."""
    lines = markdown.splitlines()
    i = 0
    while i < len(lines):
        if not lines[i].strip().startswith('|'):
            i += 1
            continue
        start = i
        while i < len(lines) and lines[i].strip().startswith('|'):
            i += 1
        block = [l for l in lines[start:i]]
        if len(block) < 3 or not SEPARATOR.match(block[1].strip()):
            continue
        # Captions sit just above or just below the table.
        nearby = lines[max(0, start - 4):start][::-1] + lines[i:i + 4]
        caption = next((CAPTION.match(clean_inline(l)) for l in nearby if CAPTION.match(clean_inline(l))), None)
        if caption:
            yield f'{caption.group(1)}: {caption.group(2)}', cells(block[0]), [cells(r) for r in block[2:]]


def metric_phrase(caption):
    # "Table 2: In-domain and zero-shot performances on BEIR. All scores denote nDCG@10." -> first sentence
    text = caption.split(':', 1)[1].strip()
    return re.split(r'(?<=[.;])\s', text)[0].rstrip('.;')[:160]


def main():
    rng = random.Random(0)
    by_doc = {}
    for path in sorted(glob.glob(os.path.join('eval', 'docs', '*.pdf'))):
        doc = os.path.basename(path)
        paper = PAPER_NAMES[doc.split('_')[0]]
        for page in parse_pdf(path, 'llamaparse'):
            for caption, header, rows in tables_on_page(page['markdown']):
                values = [c for r in rows for c in r[1:]]
                label = ''
                candidates = []
                for row in rows:
                    label = row[0] or label   # grouped rows leave the label cell empty
                    for column, value in zip(header[1:], row[1:]):
                        value = re.sub(r'[†‡*]', '', value).strip()
                        if re.search('[A-Za-z]', column) and re.search('[A-Za-z]', label) and NUMBER.match(value) and sum(re.sub(r'[†‡*]', '', v).strip() == value for v in values) == 1:
                            candidates.append((label, column, value))
                for label, column, value in rng.sample(candidates, min(PER_TABLE, len(candidates))):
                    by_doc.setdefault(doc, []).append({
                        'question': f'In the {paper} paper, what is reported for "{column}" on "{label}" in the table of {metric_phrase(caption)}?',
                        'answer': value, 'doc': doc, 'page': page['page_number'], 'caption': caption,
                        'row': label, 'column': column,
                    })

    # Spread the questions across the papers.
    questions = []
    while len(questions) < TOTAL and any(by_doc.values()):
        for doc in sorted(by_doc):
            if by_doc[doc] and len(questions) < TOTAL:
                questions.append(by_doc[doc].pop(rng.randrange(len(by_doc[doc]))))
    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(questions, f, indent=2, ensure_ascii=False)
    print(f'{len(questions)} questions -> {OUT_PATH}')
    for q in questions:
        print(f"  [{q['doc'][:6]} p{q['page']}] {q['question'][:150]}  => {q['answer']}")


if __name__ == '__main__':
    main()
