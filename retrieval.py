import numpy as np
import faiss
import pickle
import os
from functools import lru_cache
from sentence_transformers import SentenceTransformer
from pipeline import build_pipeline


DEFAULT_EMBED_MODEL = 'all-MiniLM-L6-v2'

# Some embedding models are trained with an instruction prepended to the
# *query* (never the documents). Leaving it off quietly costs retrieval quality.
QUERY_PREFIXES = {
  'BAAI/bge-small-en-v1.5': 'Represent this sentence for searching relevant passages: ',
  'BAAI/bge-base-en-v1.5': 'Represent this sentence for searching relevant passages: ',
}


@lru_cache(maxsize=None)
def get_embed_model(name=DEFAULT_EMBED_MODEL):
  # Loading a model takes seconds, so load each one once and reuse it.
  return SentenceTransformer(name)


def embed_chunks(chunks, model_name=DEFAULT_EMBED_MODEL) -> np.ndarray:
  model = get_embed_model(model_name)

  text_list = [chunk["text"] for chunk in chunks]

  # normalize_embeddings=True gives unit-length vectors, so inner-product
  # search (IndexFlatIP) behaves as cosine similarity — the standard choice
  # for text retrieval.
  embeddings = model.encode(text_list, normalize_embeddings=True)

  return embeddings



def build_index(embeddings):
  dimension = embeddings.shape[1]
  # Inner product on normalized vectors == cosine similarity.
  index = faiss.IndexFlatIP(dimension)

  index.add(embeddings.astype('float32'))

  return index


def save_index(index, chunks, index_path="chunks.index", metadata_path="chunks_metadata.pkl"):
  # Kept separate from build_index: the Streamlit app keeps the index in
  # memory per session and must not write shared files on the server.
  faiss.write_index(index, index_path)

  with open(metadata_path, "wb") as f:
    pickle.dump(chunks, f)


def retrieve(query, index, chunks, model_name=DEFAULT_EMBED_MODEL, top_k=3):
  model = get_embed_model(model_name)
  query = QUERY_PREFIXES.get(model_name, '') + query

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
    # chunk_size/overlap are counted in sentences, matching the app.
    chunks = build_pipeline('pdf.pdf', chunk_size=5, overlap=1)
    embeddings = embed_chunks(chunks)

    print("Building and saving FAISS index...")
    index = build_index(embeddings)
    save_index(index, chunks)

    print(f'Index file exists: {os.path.exists("chunks.index")}')
    print(f'Metadata file exists: {os.path.exists("chunks_metadata.pkl")}')

    # Verify the load
    index = faiss.read_index('chunks.index')
    print(f'Vectors in index: {index.ntotal}')
