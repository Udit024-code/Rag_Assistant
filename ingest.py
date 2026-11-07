"""Document ingestion: PDF -> per-page markdown -> section-aware chunks.

Parsers (all return one markdown string per page):
  llamaparse - LlamaCloud API. Tables, layout and OCR for scanned pages.
               Used automatically when LLAMA_CLOUD_API_KEY is set.
  pymupdf    - Local layout model (pymupdf4llm). Tables and headings, no OCR.
  pdfplumber - Plain text, no structure. Kept for comparison.

Chunking follows the document's own structure: pages are split at markdown
headings into sections, and long sections are split further with LlamaIndex's
SentenceSplitter. Every chunk records its paper title and section heading for
citations and the LLM prompt.

Prepending "paper: ... section: ..." to the *embedded* text (embed_context)
is optional and off by default: with bge-small it lowered Hit@5 in all five
parser/chunk-size pairings tested, as the shared title words make chunks of
one paper look alike. Worth re-testing with larger embedding models.
"""
import hashlib
import json
import logging
import os
import re

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import MetadataMode
from llama_index.core.utils import get_tokenizer

from config import get_secret

# pdfminer (under pdfplumber) logs a warning for every malformed font in a
# PDF; they are harmless and drown out useful output.
logging.getLogger('pdfminer').setLevel(logging.ERROR)

CACHE_DIR = os.path.join('.cache', 'parsed')
PARSERS = ('llamaparse', 'pymupdf', 'pdfplumber')

# Best of 128/192/256/384/512 on the eval set (LlamaParse + bge-small).
DEFAULT_CHUNK_TOKENS = 192
DEFAULT_CHUNK_OVERLAP = 24

# Tables are chunked separately: big enough to keep many rows with their
# header and caption (gemini-embedding-2 reads up to 8K tokens).
TABLE_CHUNK_TOKENS = 512
TABLE_SEPARATOR_RE = re.compile(r'^\|(\s*:?-+:?\s*\|)+\s*$')
TABLE_CAPTION_RE = re.compile(r'^\W*table\s+\d+\s*[:.]', re.IGNORECASE)

# Pages with less text than this are treated as scanned (image-only).
MIN_TEXT_CHARS = 50

HEADING_RE = re.compile(r'^(#{1,6})\s+(.*)$')
# Parsers sometimes mark figure captions or whole bold paragraphs as headings;
# real section headings are short and aren't captions.
MAX_HEADING_WORDS = 15
CAPTION_RE = re.compile(r'^(figure|fig\.|table)\s*\d', re.IGNORECASE)
# "4.1.3 Hardware" is three levels deep whatever '#' level the parser chose.
SECTION_NUMBER_RE = re.compile(r'^(\d+(?:\.\d+)*)\.?\s')


# ---------------------------------------------------------------- parsing

def _parse_llamaparse(pdf_path):
    from llama_cloud import LlamaCloud

    client = LlamaCloud(api_key=get_secret('LLAMA_CLOUD_API_KEY'))
    with open(pdf_path, 'rb') as f:
        # cost_effective runs OCR and layout analysis on every page; "agentic"
        # tiers add an LLM pass that costs several times more credits.
        result = client.parsing.parse(
            tier='cost_effective',
            version='latest',
            upload_file=(os.path.basename(pdf_path), f.read()),
            # Markdown pipe tables instead of the default HTML <table>s: about a
            # third of the tokens, which matters on Groq's 8K tokens/minute.
            output_options={'markdown': {'tables': {'output_tables_as_markdown': True,
                                                    'compact_markdown_tables': True}}},
            expand=['markdown'],
        )
    pages = []
    for page in result.markdown.pages:
        if not page.success:
            logging.warning('LlamaParse failed on page %s: %s', page.page_number, page.error)
        pages.append({'page_number': page.page_number, 'markdown': page.markdown if page.success else ''})
    return pages


def _parse_pymupdf(pdf_path):
    import pymupdf4llm

    # Layout mode: an ONNX layout model finds headings, tables and reading
    # order. ~4s/page on CPU, but parsed pages are cached.
    pymupdf4llm.use_layout(True)
    pages = pymupdf4llm.to_markdown(pdf_path, page_chunks=True, show_progress=False)
    return [{'page_number': p['metadata']['page_number'], 'markdown': p['text']} for p in pages]


def _parse_pdfplumber(pdf_path):
    from pipeline import extract_text_by_page

    return [{'page_number': p['page_number'], 'markdown': p['text']} for p in extract_text_by_page(pdf_path)]


def default_parser():
    return 'llamaparse' if get_secret('LLAMA_CLOUD_API_KEY') else 'pymupdf'


def file_digest(pdf_path):
    # Identifies a document by content, so re-uploading the same PDF under
    # another name is recognised as already indexed.
    with open(pdf_path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def parse_pdf(pdf_path, parser=None):
    """Return [{'page_number', 'markdown'}], cached on disk by file content + parser."""
    parser = parser or default_parser()
    if parser not in PARSERS:
        raise ValueError(f'unknown parser {parser!r}; choose from {PARSERS}')

    cache_path = os.path.join(CACHE_DIR, f'{file_digest(pdf_path)}-{parser}.json')
    if os.path.exists(cache_path):
        with open(cache_path, encoding='utf-8') as f:
            return json.load(f)

    pages = {'llamaparse': _parse_llamaparse, 'pymupdf': _parse_pymupdf, 'pdfplumber': _parse_pdfplumber}[parser](pdf_path)

    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(pages, f, ensure_ascii=False)
    return pages


def scanned_pages(pages):
    """Page numbers with (almost) no text - image-only pages that need OCR."""
    return [p['page_number'] for p in pages if len(p['markdown'].strip()) < MIN_TEXT_CHARS]


# ---------------------------------------------------------------- chunking

def clean_inline(text):
    """Strip markdown emphasis and HTML tags (e.g. <sup>) for headings and embeddings."""
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'(\*\*|__|\*|`)', '', text)
    text = re.sub(r'(?<!\w)_|_(?!\w)', '', text)  # _italic_ markers, not snake_case
    return re.sub(r'\s+', ' ', text).strip()


def split_sections(pages):
    """Split pages at headings, tracking the heading path across page breaks.

    Returns [{'page', 'title', 'section', 'text'}]. A page that starts in the
    middle of a section inherits that section's heading from the page before.
    """
    title = None
    stack = []  # [(level, heading)] for the current position in the document
    sections = []

    for page in pages:
        current = []

        def flush():
            text = '\n'.join(current).strip()
            if text:
                sections.append({'page': page['page_number'], 'title': title,
                                 'section': ' > '.join(h for _, h in stack), 'text': text})
            current.clear()

        for line in page['markdown'].splitlines():
            match = HEADING_RE.match(line)
            heading = clean_inline(match.group(2)) if match else ''
            if not match or len(heading.split()) > MAX_HEADING_WORDS or CAPTION_RE.match(heading):
                current.append(line)
                continue
            flush()
            number = SECTION_NUMBER_RE.match(heading)
            level = number.group(1).count('.') + 1 if number else len(match.group(1))
            if title is None:
                title = heading  # first heading in the document is the paper title
                continue
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, heading))
        flush()

    return sections


def split_tables(text):
    """Split section text into prose and tables.

    Returns [{'kind': 'text', 'text'}] and [{'kind': 'table', 'caption',
    'header', 'rows'}] blocks in order. header is [header row, separator], or
    [] for a table continued from the previous page. caption is the nearest
    "Table N:" line within four lines above or below.
    """
    lines = text.splitlines()
    blocks, prose, i = [], [], 0
    while i < len(lines):
        if not lines[i].strip().startswith('|'):
            prose.append(lines[i])
            i += 1
            continue
        start = i
        while i < len(lines) and lines[i].strip().startswith('|'):
            i += 1
        table = lines[start:i]
        has_header = len(table) >= 2 and TABLE_SEPARATOR_RE.match(table[1].strip())
        nearby = prose[-4:][::-1] + lines[i:i + 4]
        caption = next((l.strip() for l in nearby if TABLE_CAPTION_RE.match(clean_inline(l))), None)
        if prose:
            blocks.append({'kind': 'text', 'text': '\n'.join(prose)})
            prose = []
        blocks.append({'kind': 'table', 'caption': caption,
                       'header': table[:2] if has_header else [], 'rows': table[2:] if has_header else table})
    if prose:
        blocks.append({'kind': 'text', 'text': '\n'.join(prose)})
    return blocks


def table_chunk_texts(caption, header, rows, max_tokens=TABLE_CHUNK_TOKENS):
    """Split a table by rows, repeating the caption and header in every piece.

    Without them a piece is a grid of numbers: the LLM can't tell which
    column (model) or metric a value belongs to.
    """
    count = lambda lines: len(get_tokenizer()('\n'.join(lines)))
    prefix = ([caption] if caption else []) + header
    budget = max(max_tokens - count(prefix), 1)
    groups, current = [], []
    for row in rows:
        if current and count(current + [row]) > budget:
            groups.append(current)
            current = []
        current.append(row)
    if current:
        groups.append(current)
    return ['\n'.join(prefix + group) for group in groups]


def chunk_pages(pages, source, chunk_tokens=DEFAULT_CHUNK_TOKENS, chunk_overlap=DEFAULT_CHUNK_OVERLAP,
                first_chunk_id=0, embed_context=False, table_aware=True):
    """Section-aware chunks. table_aware=False chunks tables like prose (the old behaviour)."""
    sections = split_sections(pages)
    title = next((s['title'] for s in sections if s['title']), None) or os.path.splitext(source)[0]
    # chunk_size counts metadata too, so the embedded text stays within budget.
    splitter = SentenceSplitter(chunk_size=chunk_tokens, chunk_overlap=chunk_overlap)
    embed_mode = MetadataMode.EMBED if embed_context else MetadataMode.NONE

    chunks = []

    def add(text, embed_text, section, page, is_table=False):
        chunks.append({
            'chunk_id': first_chunk_id + len(chunks),
            'text': text,
            'embed_text': embed_text,
            'source': source,
            'paper': title,
            'section': section['section'],
            'page': page,
            'char_count': len(text),
            'is_table': is_table,
        })

    last_table = None   # header + caption, for tables continued on the next page
    for s in sections:
        blocks = split_tables(s['text']) if table_aware else [{'kind': 'text', 'text': s['text']}]
        for block in blocks:
            if block['kind'] == 'table':
                if not block['header'] and last_table and last_table['section'] == s['section']:
                    block = {**block, 'header': last_table['header'], 'caption': block['caption'] or last_table['caption']}
                elif block['header']:
                    last_table = {'section': s['section'], 'header': block['header'], 'caption': block['caption']}
                for text in table_chunk_texts(block['caption'], block['header'], block['rows']):
                    add(text, clean_inline(text), s, s['page'], is_table=True)
                continue
            if not block['text'].strip():
                continue
            doc = Document(text=block['text'], metadata={'paper': title, 'section': s['section'], 'source': source, 'page': s['page']})
            # Paper title and section can go into the embedded text; filename and
            # page number are for citations only.
            doc.excluded_embed_metadata_keys = ['source', 'page']
            doc.excluded_llm_metadata_keys = ['source', 'page']
            for node in splitter.get_nodes_from_documents([doc]):
                add(node.get_content(metadata_mode=MetadataMode.NONE),
                    clean_inline(node.get_content(metadata_mode=embed_mode)), s, s['page'])
    return chunks


def table_as_records(text):
    """Render a table chunk as one labelled line per row, for the LLM.

    Wide markdown tables are where the answer model misread values (a score
    reported under the neighbouring column). Spelling each cell out as
    "column: value" removes the need to count pipes.
    """
    lines = text.splitlines()
    table = [l for l in lines if l.strip().startswith('|')]
    if len(table) < 2 or not TABLE_SEPARATOR_RE.match(table[1].strip()):
        return text
    split = lambda row: [clean_inline(c.replace('<br/>', ' ').replace('<br>', ' ')) for c in row.strip().strip('|').split('|')]
    header = split(table[0])
    out = [l for l in lines if not l.strip().startswith('|')]
    label = ''
    for row in table[2:]:
        cells = split(row)
        label = cells[0] or label   # grouped rows leave the label cell empty
        pairs = [f'{h or "column " + str(i)}: {v}' for i, (h, v) in enumerate(zip(header[1:], cells[1:]), start=2) if v]
        out.append(f"- {header[0] or 'row'} = {label}; " + '; '.join(pairs))
    return '\n'.join(out)


def passage_for_llm(chunk, table_records=True):
    """The text an LLM sees for a chunk: tables as labelled records."""
    if table_records and chunk.get('is_table'):
        return table_as_records(chunk['text'])
    return chunk['text']


# ---------------------------------------------------------------- entry points

def ingest_pdf(pdf_path, source_name=None, parser=None, chunk_tokens=DEFAULT_CHUNK_TOKENS,
               chunk_overlap=DEFAULT_CHUNK_OVERLAP, first_chunk_id=0, embed_context=False, table_aware=True):
    """PDF -> chunks. Returns (chunks, scanned_page_numbers).

    scanned_page_numbers lists pages that came out empty; with the local
    parser these are image-only pages that need LlamaParse's OCR.
    """
    source_name = source_name or os.path.basename(pdf_path)
    pages = parse_pdf(pdf_path, parser)
    chunks = chunk_pages(pages, source_name, chunk_tokens, chunk_overlap, first_chunk_id, embed_context, table_aware)
    doc_id = file_digest(pdf_path)
    for chunk in chunks:
        chunk['doc_id'] = doc_id
    return chunks, scanned_pages(pages)


def ingest_corpus(pdf_paths, parser=None, chunk_tokens=DEFAULT_CHUNK_TOKENS, chunk_overlap=DEFAULT_CHUNK_OVERLAP,
                  embed_context=False, table_aware=True):
    # Chunks from several PDFs in one list, with chunk_ids unique across all of them.
    corpus = []
    for path in pdf_paths:
        chunks, _ = ingest_pdf(path, parser=parser, chunk_tokens=chunk_tokens, chunk_overlap=chunk_overlap,
                               first_chunk_id=len(corpus), embed_context=embed_context, table_aware=table_aware)
        corpus += chunks
    return corpus
