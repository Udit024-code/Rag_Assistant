import streamlit as st
import tempfile
import os

# Import your production pipeline functions
from rag_app import rag_pipeline 
from pipeline import build_pipeline
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

# 1. File Uploader
uploaded_file = st.file_uploader("Upload your research PDF", type=["pdf"])

if uploaded_file:
    # 2. Process the file only if we haven't already indexed it
    if st.session_state.index is None:
        with st.spinner("Extracting text and building vector index... This may take a moment."):
            
            # Save the uploaded file to a temporary path for pdfplumber to read
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            try:
                # Run your data ingestion pipeline
                # Note: Adjust chunk_size and overlap parameters to match your Phase 5 logic
                new_chunks = build_pipeline(tmp_file_path, chunk_size=5, overlap=1)
                
                # Embed and build index
                new_embeddings = embed_chunks(new_chunks)
                new_index = build_index(new_embeddings, new_chunks)
                
                # 3. Store results in st.session_state
                st.session_state.index = new_index
                st.session_state.chunks = new_chunks
                
                st.success("Document successfully processed and indexed!")
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
                                st.markdown(f"**{i}. Page {chunk['page']}**")
                                st.caption(chunk["text"])
                    except Exception as e:
                        st.error(f"An error occurred during execution: {e}")