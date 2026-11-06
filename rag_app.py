from groq import Groq

from config import get_secret


# Initialize the Groq client with a key from either .env or Streamlit secrets
client = Groq(api_key=get_secret("GROQ_API_KEY"))

def build_prompt(question, retrieved_chunks):

  system_instruction = (
        "SYSTEM: You are a research assistant. Answer the user's question using ONLY the context provided below.\n"
        "If the answer is not in the context, say 'I cannot find this information in the provided documents.'\n"
        "Always cite the paper and page number your answer comes from."
    )

  context_parts = []
  for chunk in retrieved_chunks:
        page_num = chunk["page"]  # Matches ingest.py schema
        chunk_text = chunk["text"] # Matches ingest.py schema
        paper = chunk.get("paper") or chunk["source"]
        section = f" | {chunk['section']}" if chunk.get("section") else ""
        context_parts.append(f"[{paper}, page {page_num}{section}]:\n{chunk_text}")

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
    

def rag_pipeline(question, retriever, k=5, **filters):
    """
    Step 4: Orchestrates the entire end-to-end RAG system.

    retriever is a retrieval.Retriever (hybrid search + reranking); filters
    such as owners/doc_ids limit which documents are searched.
    """
    # 1-2. Retrieve the top-k chunks. The reranker keeps these precise, so 5
    # fit comfortably in Groq's free 8K tokens/minute.
    retrieved_chunks = retriever(question, k, **filters)

    # 3. Build the augmented prompt
    formatted_prompt = build_prompt(question, retrieved_chunks)

    # 4. Generate grounded answer from the LLM
    answer = generate_answer(formatted_prompt)

    # Return the retrieved chunks too, so the UI can show the sources the
    # answer was grounded in (page numbers + snippets).
    return answer, retrieved_chunks



