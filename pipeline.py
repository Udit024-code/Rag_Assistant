import pdfplumber
import os
import nltk
nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize

def extract_text_by_page(pdf_path):
    pages_data = []

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""

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




def build_pipeline(pdf_path, chunk_size, overlap):
    final_chunks = []
    chunk_id = 0

    pages = extract_text_by_page(pdf_path)

    for page in pages:
        page_number = page["page_number"]
        text = page["text"]

        chunks = chunk_text_sentences(text, max_sentences=chunk_size, overlap_sentences=overlap)

        for chunk in chunks:
            chunk_data = {
                "chunk_id": chunk_id,
                "text": chunk,
                "source": os.path.basename(pdf_path),
                "page": page_number,
                "char_count": len(chunk)
            }

            final_chunks.append(chunk_data)
            chunk_id += 1

    return final_chunks


