# rag_engine.py

import os
import faiss
import numpy as np
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import tiktoken
from openai import OpenAI
import pickle

# Load embedding model (free & local)
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# Read and chunk all PDFs in a folder
def read_pdfs(folder_path, max_tokens=500):
    text_chunks = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            reader = PdfReader(os.path.join(folder_path, filename))
            full_text = ""
            for page in reader.pages:
                full_text += page.extract_text() or ""
            text_chunks.extend(chunk_text(full_text, max_tokens))
    return text_chunks

# Token-based chunking
def chunk_text(text, max_tokens=500):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    chunks = [tokenizer.decode(tokens[i:i+max_tokens]) for i in range(0, len(tokens), max_tokens)]
    return chunks

# Embed text chunks
def embed_documents(chunks):
    vectors = embed_model.encode(chunks, convert_to_tensor=False)
    return np.array(vectors).astype("float32")

# Build FAISS index
def build_faiss_index(vectors):
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)
    return index


# Save index and chunks
def save_index(index, chunks, folder="data"):
    os.makedirs(folder, exist_ok=True)
    faiss.write_index(index, os.path.join(folder, "index.faiss"))
    with open(os.path.join(folder, "chunks.pkl"), "wb") as f:
        pickle.dump(chunks, f)


def load_index(folder="data"):
    try:
        index = faiss.read_index(os.path.join(folder, "index.faiss"))
        with open(os.path.join(folder, "chunks.pkl"), "rb") as f:
            chunks = pickle.load(f)
        return index, chunks
    except Exception as e:
        print("⚠️ Failed to load saved index:", e)
        return None, None


import hashlib
import json

# Compute hash of all PDFs in folder
def compute_folder_hash(folder):
    files = sorted([
        (f, os.path.getmtime(os.path.join(folder, f)))
        for f in os.listdir(folder)
        if f.endswith(".pdf")
    ])
    file_info = json.dumps(files).encode('utf-8')
    return hashlib.md5(file_info).hexdigest()

# Save hash
def save_hash(hash_val, folder="data"):
    with open(os.path.join(folder, "hash.txt"), "w") as f:
        f.write(hash_val)

# Load saved hash
def load_saved_hash(folder="data"):
    try:
        with open(os.path.join(folder, "hash.txt"), "r") as f:
            return f.read()
    except FileNotFoundError:
        return None



# Query GPT-4.1 with retrieved context
def query_gpt(question, chunks, index, client, top_k=5):
    question_vector = embed_model.encode([question])[0].astype("float32").reshape(1, -1)
    D, I = index.search(question_vector, top_k)
    retrieved_chunks = [chunks[i] for i in I[0]]

    context = "\n\n".join(retrieved_chunks)
    prompt = f"""
You are a helpful assistant. Use the context below to answer the user's question.
If the context is not relevant or does not contain the answer, you may use your own knowledge.

Context:
{context}

Question:
{question}

Answer:"""

    response = client.chat.completions.create(
        model="gpt-4.1-nano-2025-04-14",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    return response.choices[0].message.content
