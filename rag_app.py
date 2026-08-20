import os
import pickle
import faiss
from groq import Groq
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Import retrieval function from your existing retrieval.py file
from retrieval import retrieve

# Load environment variables from .env file (used when running locally)
load_dotenv()


def _get_groq_api_key():
    """Read the Groq API key from the environment (.env locally) or,
    when deployed on Streamlit Cloud, from st.secrets."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        try:
            import streamlit as st
            api_key = st.secrets["GROQ_API_KEY"]
        except Exception:
            pass
    return api_key


# Initialize the Groq client with a key from either .env or Streamlit secrets
client = Groq(api_key=_get_groq_api_key())

def build_prompt(question, retrieved_chunks):

  system_instruction = (
        "SYSTEM: You are a research assistant. Answer the user's question using ONLY the context provided below.\n"
        "If the answer is not in the context, say 'I cannot find this information in the provided document.'\n"
        "Always cite the page number your answer comes from."
    )
  
  context_parts = []
  for chunk in retrieved_chunks:
        page_num = chunk["page"]  # Matches pipeline.py schema
        chunk_text = chunk["text"] # Matches pipeline.py schema
        context_parts.append(f"[Page {page_num}]: {chunk_text}")

  context_str = "\n".join(context_parts)

  prompt = (
        f"{system_instruction}\n\n"
        f"CONTEXT:\n{context_str}\n\n"
        f"QUESTION:\n{question}"
    )
    
  return prompt

def generate_answer(prompt):
    """
    Step 3: Sends the prompt to Groq and extracts the text response.
    """
    try:
        response = client.chat.completions.create(
            # Change this exact line below:
            model="llama-3.1-8b-instant",  
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_completion_tokens=1024
        )
        # Extract the clean response text from Groq's structure
        return response.choices[0].message.content
        
    except Exception as e:
        return f"API Error: Failed to generate answer due to: {e}"
    

# Load the embedding model once at import time and reuse it for every query
# (loading it on each call is slow, especially on limited cloud hardware).
_EMBED_MODEL = None


def _get_embed_model():
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        _EMBED_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
    return _EMBED_MODEL


def rag_pipeline(question, index, chunks):
    """
    Step 4: Orchestrates the entire end-to-end RAG system in memory.
    """
    # 1. Load Embedding Model (cached after the first call)
    model = _get_embed_model()

    # 2. Retrieve top-3 relevant chunks using your retrieval.py logic
    retrieved_chunks = retrieve(question, index, chunks, model, top_k=3)

    # 3. Build the augmented prompt
    formatted_prompt = build_prompt(question, retrieved_chunks)

    # 4. Generate grounded answer from the LLM
    answer = generate_answer(formatted_prompt)

    # Return the retrieved chunks too, so the UI can show the sources the
    # answer was grounded in (page numbers + snippets).
    return answer, retrieved_chunks



