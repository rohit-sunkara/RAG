# app.py

import streamlit as st
import os
from openai import OpenAI
from Rag import (
    read_pdfs, embed_documents, build_faiss_index, query_gpt,
    load_index, save_index,
    compute_folder_hash, load_saved_hash, save_hash
)

st.set_page_config(page_title="RAG Q&A", page_icon="📄")

st.title("📄 Ask Questions from Your PDFs using GPT-4.1")
st.markdown("Upload a folder of PDFs and ask questions. The assistant will use GPT-4.1 + your content.")

# --- Step 1: Inputs ---
pdf_folder = st.text_input("📁 Path to PDF folder:", value=r"C:\Users\geeth\Desktop\rohit\RAG")
api_key = ""
client = OpenAI(api_key=api_key)

if st.button("📚 Load PDFs and Build Index"):
    if not os.path.exists(pdf_folder):
        st.error("❌ Folder path does not exist.")
    elif not api_key:
        st.error("❌ Please enter a valid API key.")
    else:
        with st.spinner("Checking for changes in PDFs..."):
            current_hash = compute_folder_hash(pdf_folder)
            saved_hash = load_saved_hash()

            if saved_hash != current_hash:
                st.info("📄 PDFs have changed. Rebuilding index...")
                chunks = read_pdfs(pdf_folder)
                vectors = embed_documents(chunks)
                index = build_faiss_index(vectors)

                # Save everything
                save_index(index, chunks)
                save_hash(current_hash)
            else:
                st.success("✅ PDFs unchanged. Loading cached index...")
                index, chunks = load_index()

            client = OpenAI(api_key=api_key)

        st.session_state.chunks = chunks
        st.session_state.index = index
        st.session_state.client = client


# --- Step 2: Ask questions ---
if "chunks" in st.session_state and "index" in st.session_state:
    question = st.text_input("💬 Ask your question:")

    if st.button("🤖 Get Answer"):
        with st.spinner("Thinking..."):
            answer = query_gpt(
                question,
                st.session_state.chunks,
                st.session_state.index,
                st.session_state.client
            )
        st.markdown("### 🧠 Answer:")
        st.write(answer)
