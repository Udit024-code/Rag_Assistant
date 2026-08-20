import numpy as np
import faiss
import pickle
import os
from sentence_transformers import SentenceTransformer
from pipeline import build_pipeline



def embed_chunks(chunks) -> np.ndarray:
  model = SentenceTransformer('all-MiniLM-L6-v2')

  text_list = [chunk["text"] for chunk in chunks]

  # normalize_embeddings=True gives unit-length vectors, so inner-product
  # search (IndexFlatIP) behaves as cosine similarity — the standard choice
  # for text retrieval.
  embeddings = model.encode(text_list, normalize_embeddings=True)

  return embeddings



def build_index(embeddings, chunks) -> list:
  dimension = embeddings.shape[1]
  # Inner product on normalized vectors == cosine similarity.
  index = faiss.IndexFlatIP(dimension)

  embeddings_float32 = embeddings.astype('float32')

  index.add(embeddings_float32)

  faiss.write_index(index, "chunks.index")

  with open("chunks_metadata.pkl", "wb") as f:
    pickle.dump(chunks, f)

  return index


def retrieve(query, index, chunks, model, top_k=3):
  # Normalize the query the same way as the chunks so the inner-product
  # search measures cosine similarity.
  query_embedding = model.encode(query, normalize_embeddings=True).astype('float32')

  query_vector = query_embedding.reshape(1, -1)

  distances, indices = index.search(query_vector, top_k)

  relevant_chunks = []
  for idx in indices[0]:

    if idx != -1 and idx < len(chunks):
      relevant_chunks.append(chunks[idx])

  return relevant_chunks

if __name__ == "__main__":
    # Everything below this line ONLY runs if you execute retrieval.py directly.
    # It will NOT run when Phase 4 imports this file!
    
    print("Building pipeline and embedding chunks...")
    chunks = build_pipeline('pdf.pdf', chunk_size=500, overlap=50)
    embeddings = embed_chunks(chunks)
    
    print("Building and saving FAISS index...")
    build_index(embeddings, chunks)
    
    print(f'Index file exists: {os.path.exists("chunks.index")}')
    print(f'Metadata file exists: {os.path.exists("chunks_metadata.pkl")}')
    
    # Verify the load
    index = faiss.read_index('chunks.index')
    print(f'Vectors in index: {index.ntotal}')