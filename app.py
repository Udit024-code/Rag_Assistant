import streamlit as st
import tempfile
import os

# Import your production pipeline functions
from rag_app import rag_pipeline
from ingest import default_parser, ingest_pdf
from retrieval import embed_chunks, build_index

# App title and description
st.title("RAG Research Assistant")
st.caption("Upload a PDF and interact with your grounded research document index in real-time.")

# --- SESSION STATE SETUP ---
# We use session_state so the app doesn't re-embed the PDF every time you click a button
if "index" not in st.session_state:
    st.session_state.index = None
if "chunks" not in st.session_state:
    st.session_state.chunks = None
if "indexed_file" not in st.session_state:
    st.session_state.indexed_file = None

# 1. File Uploader
uploaded_file = st.file_uploader("Upload your research PDF", type=["pdf"])

if uploaded_file:
    # 2. Process the file only if we haven't already indexed it. Comparing
    # name + size means uploading a different PDF triggers a fresh index
    # instead of silently answering from the previous document.
    file_key = (uploaded_file.name, uploaded_file.size)
    if st.session_state.indexed_file != file_key:
        parser = default_parser()
        # The local layout parser takes a few seconds per page on CPU.
        wait_hint = "" if parser == "llamaparse" else " Local parsing takes a few seconds per page."
        with st.spinner(f"Parsing the PDF ({parser}) and building the vector index...{wait_hint}"):

            # Save the uploaded file to a temporary path for the parser to read
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name

            try:
                # Run the ingestion pipeline: parse -> section-aware chunks
                new_chunks, scanned = ingest_pdf(tmp_file_path, source_name=uploaded_file.name, parser=parser)
                if not new_chunks:
                    st.error("No text could be extracted from this PDF. If it is a scanned document, "
                             "set LLAMA_CLOUD_API_KEY so it can be read with OCR.")
                    st.stop()

                # Embed and build index
                new_embeddings = embed_chunks(new_chunks)
                new_index = build_index(new_embeddings)

                # 3. Store results in st.session_state
                st.session_state.index = new_index
                st.session_state.chunks = new_chunks
                st.session_state.indexed_file = file_key

                st.success(f"Document indexed: {len(new_chunks)} chunks.")
                if scanned:
                    # Image-only pages have no text layer; only LlamaParse runs OCR.
                    st.warning(f"Pages {', '.join(map(str, scanned))} look scanned (no text found) and were skipped. "
                               "Set LLAMA_CLOUD_API_KEY to read them with OCR.")
            finally:
                # Clean up the temporary file so we don't leak storage
                os.remove(tmp_file_path)

    # Only show the search bar IF the document is processed
    if st.session_state.index is not None:
        # Text input for the user's question
        question = st.text_input("Enter your research question:", placeholder="e.g., What are the three structural blind spots?")
        
        # Search button to trigger execution
        if st.button("Run RAG Pipeline"):
            if question.strip() == "":
                st.warning("Please enter a valid question.")
            else:
                with st.spinner("Searching index and generating answer..."):
                    try:
                        # 4. Run the pipeline using the memory-loaded index and chunks
                        answer, sources = rag_pipeline(
                            question=question,
                            index=st.session_state.index,
                            chunks=st.session_state.chunks
                        )

                        # Display the response cleanly using Markdown
                        st.markdown("### Answer:")
                        st.write(answer)

                        # Show the retrieved source chunks the answer is grounded in,
                        # so the user can verify it against the original document.
                        with st.expander(f"📄 Sources ({len(sources)} chunks retrieved)"):
                            for i, chunk in enumerate(sources, start=1):
                                section = f" · {chunk['section']}" if chunk.get("section") else ""
                                st.markdown(f"**{i}. Page {chunk['page']}**{section}")
                                # Chunks are markdown, so tables render as tables.
                                st.markdown(chunk["text"])
                    except Exception as e:
                        st.error(f"An error occurred during execution: {e}")