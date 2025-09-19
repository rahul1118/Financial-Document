# streamlit_app.py
import streamlit as st
import io
import pandas as pd
from document_processor import (
    extract_text_and_tables_from_pdf,
    extract_text_and_tables_from_excel,
    chunk_text_blocks, tables_to_text_snippets
)
from qa import Retriever, answer_question
from dotenv import load_dotenv
import os

load_dotenv()

st.set_page_config(page_title="Financial Document Q&A", layout="wide")

st.title("Financial Document Q&A Assistant")
st.markdown("Upload financial PDFs / Excel and ask questions about revenue, expenses, profits, etc.")

# Sidebar settings
with st.sidebar:
    st.header("Settings")
    model = st.text_input("Local Ollama model name", value=os.getenv("OLLAMA_MODEL", "llama2"))
    top_k = st.slider("Retrieval top K", min_value=1, max_value=5, value=3)
    max_chunk_chars = st.number_input("Chunk size (chars)", value=800, step=100)
    overlap_chars = st.number_input("Chunk overlap (chars)", value=200, step=50)
    st.markdown("Make sure Ollama CLI is installed and model is pulled locally (e.g., `ollama pull <model>`).")

uploaded_files = st.file_uploader("Upload PDF or Excel files", accept_multiple_files=True, type=['pdf','xls','xlsx'])
process_button = st.button("Process uploaded files")

# Session state for extracted docs and retriever
if "docs" not in st.session_state:
    st.session_state.docs = []  # list of text chunks
if "orig_tables" not in st.session_state:
    st.session_state.orig_tables = []  # list of DataFrames (for display)

if process_button:
    if not uploaded_files:
        st.warning("Please upload at least one PDF or Excel file.")
    else:
        st.info("Processing files...")
        all_text_blocks = []
        all_tables = []
        for f in uploaded_files:
            fname = f.name
            bytes_io = io.BytesIO(f.read())
            if fname.lower().endswith(".pdf"):
                out = extract_text_and_tables_from_pdf(bytes_io)
            else:
                out = extract_text_and_tables_from_excel(bytes_io)
            text_blocks = out.get("text_blocks", [])
            tables = out.get("tables", [])
            all_text_blocks.extend(text_blocks)
            all_tables.extend(tables)
        # convert tables to text snippets and create chunks
        table_snips = tables_to_text_snippets(all_tables)
        chunks = chunk_text_blocks(all_text_blocks + table_snips, max_chars=int(max_chunk_chars), overlap=int(overlap_chars))
        st.session_state.docs = chunks
        st.session_state.orig_tables = all_tables
        st.success(f"Processed {len(uploaded_files)} files — extracted {len(chunks)} text chunks and {len(all_tables)} tables.")

# Show extracted tables (collapsible)
if st.session_state.orig_tables:
    with st.expander("Show extracted tables"):
        for i, df in enumerate(st.session_state.orig_tables):
            st.write(f"Table {i+1} preview:")
            try:
                st.dataframe(df.head(50))
            except Exception:
                st.write(str(df))

# Build retriever if docs present
if st.session_state.docs:
    retriever = Retriever(st.session_state.docs)
    st.success("Retriever ready.")
else:
    retriever = None

# Chat UI
st.markdown("---")
st.header("Ask a question")
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

question = st.text_input("Type your question about the uploaded documents (e.g., 'What was total revenue in 2024?')")

if st.button("Ask"):
    if not question:
        st.warning("Type a question first.")
    elif retriever is None:
        st.warning("No documents processed yet. Upload and process files first.")
    else:
        st.info("Retrieving context and asking local model...")
        answer = answer_question(retriever, question, st.session_state.docs, model=model, top_k=int(top_k))
        st.session_state.chat_history.append({"q": question, "a": answer})
        st.success("Answer received — see below.")

# Display chat history
if st.session_state.chat_history:
    st.markdown("### Conversation")
    for turn in reversed(st.session_state.chat_history):
        st.markdown(f"**Q:** {turn['q']}")
        st.markdown(f"**A:** {turn['a']}")

st.markdown("---")
st.caption("This app extracts text & tables, retrieves top contexts using TF-IDF, then calls a local Ollama model to generate the final answer. Keep numerical context as close as possible (e.g., 'Total Revenue: 1,234,567').")
