# 📦 Required Libraries
import os
import faiss
import numpy as np
import openai
import getpass
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from typing import List
import tiktoken

# 🔐 Ask for OpenAI API Key securely
from openai import OpenAI

# 🔑 Manually input API key
api_key = ""
client = OpenAI(api_key=api_key)

# 📁 Set your PDF folder path here
PDF_FOLDER = r"C:\Users\geeth\Desktop\rohit\RAG"  # ⬅ Replace with your actual path

# 🧠 Load the embedding model (runs locally)
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# 🧩 Step 1: Read and chunk PDFs
def read_pdfs(folder_path: str) -> List[str]:
    text_chunks = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            reader = PdfReader(os.path.join(folder_path, filename))
            full_text = ""
            for page in reader.pages:
                full_text += page.extract_text() or ""
            chunks = chunk_text(full_text)
            text_chunks.extend(chunks)
    return text_chunks

# ✂️ Step 2: Token-based chunking (300 tokens per chunk)
def chunk_text(text: str, max_tokens=300) -> List[str]:
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk = tokenizer.decode(tokens[i:i+max_tokens])
        chunks.append(chunk)
    return chunks

# 🔢 Step 3: Embed all text chunks
def embed_documents(chunks: List[str]):
    vectors = embed_model.encode(chunks, convert_to_tensor=False)
    return vectors

# 🧠 Step 4: Create FAISS vector index
def build_faiss_index(vectors, chunks):
    dim = len(vectors[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(vectors).astype("float32"))
    return index

# 🤖 Step 5: Ask GPT-4.1 using top-k similar chunks
def query_rag(question: str, index, chunks, top_k=5):
    import numpy as np

    # Embed the question
    question_vector = embed_model.encode([question])[0]
    question_vector = np.array([question_vector]).astype("float32")

    # Search for relevant chunks using FAISS
    D, I = index.search(question_vector, top_k)
    retrieved_chunks = [chunks[i] for i in I[0]]

    # Construct context
    context = "\n\n".join(retrieved_chunks)
    prompt = f"""
You are a helpful assistant. Use the context below to answer the user's question
If the context is not relevant or does not contain the answer, you may use your own knowledge.
. Answer within 100 words.

Context:
{context}

Question:
{question}

Answer:"""

    # Send to GPT-4.1 using OpenAI SDK v1+
    response = client.chat.completions.create(
        model="gpt-4.1-nano-2025-04-14",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    return response.choices[0].message.content


# 🚀 Main app flow
def main():
    print("📚 Reading and chunking PDFs...")
    chunks = read_pdfs(PDF_FOLDER)

    print("🔍 Embedding text chunks...")
    vectors = embed_documents(chunks)

    print("🧠 Building FAISS index...")
    index = build_faiss_index(vectors, chunks)

    print("\n✅ Ready! Ask me anything from your PDFs.")
    while True:
        question = input("\n💬 Your question (or type 'exit'): ")
        if question.lower() == 'exit':
            break
        answer = query_rag(question, index, chunks)
        print(f"\n🧠 Answer: {answer}")

# 🏁 Run the app
if __name__ == "__main__":
    main()
