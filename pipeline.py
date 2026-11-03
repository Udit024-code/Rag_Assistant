import logging
import pdfplumber
import os
import nltk
from nltk.tokenize import sent_tokenize

# Download the sentence tokenizer only if it isn't installed yet, instead of
# hitting the network on every import.
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

# pdfminer (under pdfplumber) logs a warning for every malformed font in a
# PDF; they are harmless and drown out useful output.
logging.getLogger('pdfminer').setLevel(logging.ERROR)

# pdfplumber's default x_tolerance (3) treats small gaps between letters as
# "same word", which glues 10-20% of words together in LaTeX papers
# ("TheRAG-Sequencemodeluses..."). 1.5 keeps word boundaries intact and
# raised Hit@5 from 0.52 to 0.78 on the eval set (run evaluate.py with
# --x-tolerance 3 to reproduce the old behaviour).
DEFAULT_X_TOLERANCE = 1.5


def extract_text_by_page(pdf_path, x_tolerance=DEFAULT_X_TOLERANCE):
    pages_data = []

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text(x_tolerance=x_tolerance) or ""

            page_info = {
                "page_number": i,
                "text": text
            }

            pages_data.append(page_info)

    return pages_data



def chunk_text_sentences(text, max_sentences=5, overlap_sentences=1):
    # Split the raw text into a clean list of individual sentences
    sentences = sent_tokenize(text)
    chunks = []
    
    # If the text is shorter than the max window size, return it as a single chunk
    if len(sentences) <= max_sentences:
        if sentences:
            chunks.append(" ".join(sentences))
        return chunks

    # Calculate how many new sentences to advance after each window stride
    step = max_sentences - overlap_sentences
    
    # Slide the window across the sentence array
    for i in range(0, len(sentences), step):
        # Extract the segment slice of sentences
        window_sentences = sentences[i : i + max_sentences]
        
        # Ensure we don't save near-empty trailing overlaps at the absolute end
        if i > 0 and len(window_sentences) <= overlap_sentences:
            break
            
        # Join the chosen sentences back into a single text block string
        chunk_str = " ".join(window_sentences)
        chunks.append(chunk_str)
        
    return chunks




def build_pipeline(pdf_path, chunk_size, overlap, source_name=None,
                   x_tolerance=DEFAULT_X_TOLERANCE, first_chunk_id=0):
    # source_name lets callers record the original filename when pdf_path is
    # a temporary copy (e.g. a Streamlit upload).
    source_name = source_name or os.path.basename(pdf_path)
    final_chunks = []
    chunk_id = first_chunk_id

    pages = extract_text_by_page(pdf_path, x_tolerance=x_tolerance)

    for page in pages:
        page_number = page["page_number"]
        text = page["text"]

        chunks = chunk_text_sentences(text, max_sentences=chunk_size, overlap_sentences=overlap)

        for chunk in chunks:
            chunk_data = {
                "chunk_id": chunk_id,
                "text": chunk,
                "source": source_name,
                "page": page_number,
                "char_count": len(chunk)
            }

            final_chunks.append(chunk_data)
            chunk_id += 1

    return final_chunks


def build_corpus(pdf_paths, chunk_size, overlap, x_tolerance=DEFAULT_X_TOLERANCE):
    # Chunks from several PDFs in one list, with chunk_ids unique across all of them.
    corpus = []
    for path in pdf_paths:
        corpus += build_pipeline(path, chunk_size, overlap, x_tolerance=x_tolerance,
                                 first_chunk_id=len(corpus))
    return corpus


