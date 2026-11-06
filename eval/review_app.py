"""Click-through review of LLM-drafted test questions.

Shows one unverified question at a time next to the PDF page it claims to
come from. Fix the question, pages or answer if needed, then mark it correct
or delete it. Every click saves eval/testset.json.

Run from the project root:
    python -m streamlit run eval/review_app.py
"""
import json
import os

import pymupdf
import streamlit as st

TESTSET_PATH = os.path.join('eval', 'testset.json')
DOCS_DIR = os.path.join('eval', 'docs')


def load_testset():
    with open(TESTSET_PATH, encoding='utf-8') as f:
        return json.load(f)


def save_testset(testset):
    with open(TESTSET_PATH, 'w', encoding='utf-8') as f:
        json.dump(testset, f, indent=2, ensure_ascii=False)


@st.cache_resource
def open_pdf(doc):
    return pymupdf.open(os.path.join(DOCS_DIR, doc))


@st.cache_data
def page_png(doc, page_number):
    return open_pdf(doc)[page_number - 1].get_pixmap(dpi=110).tobytes('png')


st.set_page_config(page_title='Review test questions', layout='wide')

with st.sidebar:
    st.header('How to review')
    st.markdown(
        '- **Delete** questions from reference lists or checklists, vague ones, '
        'and ones whose answer is wrong or not on the page.\n'
        '- If other pages also answer it, **add them** to the pages box.\n'
        '- Fix a wrong reference answer or clumsy wording, then click **Correct**.\n'
        '- Use **Skip** to come back to a question later.'
    )

testset = load_testset()
pending = [i for i, t in enumerate(testset) if not t.get('verified')]
done = len(testset) - len(pending)

st.progress(done / len(testset) if testset else 1.0, text=f'{done} of {len(testset)} verified')
if not pending:
    st.success('All questions are reviewed. Run evaluate.py to get trustworthy numbers.')
    st.stop()

i = pending[0]
t = testset[i]
key = f"{i}-{abs(hash(t['question']))}"  # fresh widgets for every question
page_count = open_pdf(t['doc']).page_count

left, right = st.columns([2, 3])

with left:
    st.caption(f"{t['doc']} · {len(pending)} left")
    question = st.text_area('Question', t['question'], key=f'q-{key}', height=100)
    pages_text = st.text_input('Pages that answer it (comma-separated)',
                               ', '.join(map(str, t['expected_pages'])), key=f'p-{key}')
    answer = st.text_area('Reference answer', t.get('reference_answer', ''), key=f'a-{key}', height=100)
    keywords = st.text_input('Keywords a correct answer contains (comma-separated)',
                             ', '.join(t.get('expected_keywords', [])), key=f'k-{key}')

    try:
        pages = sorted({int(p) for p in pages_text.replace(' ', '').split(',') if p})
        pages_ok = bool(pages) and all(1 <= p <= page_count for p in pages)
    except ValueError:
        pages, pages_ok = [], False
    if not pages_ok:
        st.error(f'Pages must be numbers between 1 and {page_count}.')

    save_col, delete_col, skip_col = st.columns(3)
    if save_col.button('Correct', type='primary', disabled=not pages_ok, width='stretch'):
        t.update({
            'question': question.strip(),
            'expected_pages': pages,
            'reference_answer': answer.strip(),
            'expected_keywords': [k.strip() for k in keywords.split(',') if k.strip()],
            'verified': True,
        })
        save_testset(testset)
        st.rerun()
    if delete_col.button('Delete', width='stretch'):
        testset.pop(i)
        save_testset(testset)
        st.rerun()
    if skip_col.button('Skip', width='stretch'):
        testset.append(testset.pop(i))  # move to the end of the queue
        save_testset(testset)
        st.rerun()

with right:
    shown = st.number_input('Showing page', min_value=1, max_value=page_count,
                            value=t['expected_pages'][0], key=f'view-{key}')
    st.image(page_png(t['doc'], int(shown)), width='stretch')
