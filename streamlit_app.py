# app/streamlit_app.py

import streamlit as st

from retrieval.vector_store import VectorStore
from retrieval.retriever import Retriever
from retrieval.hybrid_retriever import HybridRetriever

from embeddings.text_embedder import TextEmbedder
from generation.llm_generator import LLMGenerator


# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Multi-Modal RAG QA System",
    layout="wide"
)

st.title("📊 Multi-Modal Document Intelligence RAG System")
st.write("Ask questions over Excel, PDF, and ChartQA datasets")

# -----------------------------
# SESSION STATE INIT
# -----------------------------
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "generator" not in st.session_state:
    st.session_state.generator = LLMGenerator()

# -----------------------------
# QUERY INPUT
# -----------------------------
query = st.text_input("🔎 Ask your question:")

# -----------------------------
# MAIN PIPELINE
# -----------------------------
if query:

    if st.session_state.retriever is None:
        st.error("Vector store not loaded. Run ingestion first (main.py).")

    else:
        with st.spinner("Retrieving relevant context..."):

            retrieved_chunks = st.session_state.retriever.retrieve(query)

        # -----------------------------
        # SHOW RETRIEVED CONTEXT
        # -----------------------------
        st.subheader("📚 Retrieved Context")

        for i, chunk in enumerate(retrieved_chunks):
            with st.expander(f"Source {i+1} | Type: {chunk.get('type')}"):
                st.write("**Page:**", chunk.get("page", "N/A"))
                st.write(chunk.get("content"))

        # -----------------------------
        # GENERATE ANSWER
        # -----------------------------
        with st.spinner("Generating answer..."):

            answer = st.session_state.generator.generate(
                query,
                retrieved_chunks
            )

        # -----------------------------
        # FINAL ANSWER
        # -----------------------------
        st.subheader("💡 Answer")
        st.success(answer)

# -----------------------------
# SIDEBAR INFO
# -----------------------------
st.sidebar.title("System Info")

st.sidebar.write("""
This system supports:

✔ Excel financial tables  
✔ IMF PDF reports  
✔ ChartQA images
✔ Multi-modal retrieval (FAISS)  
✔ LLM grounded generation  
✔ Citation-based answers  
""")