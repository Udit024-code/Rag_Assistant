from groq import Groq

from config import get_secret
# Import retrieval function from your existing retrieval.py file
from retrieval import retrieve


# Initialize the Groq client with a key from either .env or Streamlit secrets
client = Groq(api_key=get_secret("GROQ_API_KEY"))

def build_prompt(question, retrieved_chunks):

  system_instruction = (
        "SYSTEM: You are a research assistant. Answer the user's question using ONLY the context provided below.\n"
        "If the answer is not in the context, say 'I cannot find this information in the provided document.'\n"
        "Always cite the page number your answer comes from."
    )
  
  context_parts = []
  for chunk in retrieved_chunks:
        page_num = chunk["page"]  # Matches ingest.py schema
        chunk_text = chunk["text"] # Matches ingest.py schema
        section = f" | {chunk['section']}" if chunk.get("section") else ""
        context_parts.append(f"[Page {page_num}{section}]:\n{chunk_text}")

  # Chunks span several lines (markdown tables), so separate them clearly.
  context_str = "\n\n".join(context_parts)

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
            # llama-3.1-8b-instant was shut down by Groq on 2026-08-16;
            # gpt-oss-20b is Groq's recommended replacement.
            model="openai/gpt-oss-20b",
            messages=[
                {"role": "user", "content": prompt}
            ],
            # gpt-oss is a reasoning model: its hidden reasoning tokens count
            # toward max_completion_tokens, so keep reasoning short and leave
            # headroom for the visible answer.
            reasoning_effort="low",
            max_completion_tokens=2048
        )
        # Extract the clean response text from Groq's structure
        return response.choices[0].message.content
        
    except Exception as e:
        return f"API Error: Failed to generate answer due to: {e}"
    

def rag_pipeline(question, index, chunks):
    """
    Step 4: Orchestrates the entire end-to-end RAG system in memory.
    """
    # 1-2. Retrieve top-3 relevant chunks. retrieval.py caches the embedding
    # model, so indexing and querying share one loaded copy.
    retrieved_chunks = retrieve(question, index, chunks, top_k=3)

    # 3. Build the augmented prompt
    formatted_prompt = build_prompt(question, retrieved_chunks)

    # 4. Generate grounded answer from the LLM
    answer = generate_answer(formatted_prompt)

    # Return the retrieved chunks too, so the UI can show the sources the
    # answer was grounded in (page numbers + snippets).
    return answer, retrieved_chunks



