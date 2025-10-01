import json

import faiss
import numpy as np
import ollama
import streamlit as st
from sentence_transformers import SentenceTransformer

# Paths
INDEX_PATH = "./data/vector_store/faiss.index"
METADATA_PATH = "./data/vector_store/metadata.json"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ---------- Load FAISS + metadata ----------
@st.cache_resource
def load_index_and_metadata():
    index = faiss.read_index(INDEX_PATH)
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return index, metadata


# ---------- Embed query ----------
@st.cache_resource
def load_embedder(model_name=EMBED_MODEL):
    return SentenceTransformer(model_name)


def embed_query(query: str, embedder) -> np.ndarray:
    return embedder.encode([query], convert_to_numpy=True)


# ---------- Retrieve top-k chunks ----------
def retrieve(query_vec: np.ndarray, index, metadata, k: int = 3):
    distances, indices = index.search(query_vec, k)
    results = [metadata[str(idx)] for idx in indices[0] if str(idx) in metadata]
    return results


# ---------- Build augmented prompt ----------
def build_prompt(query: str, retrieved_chunks):
    context = "\n\n".join([f"- {c['text']}" for c in retrieved_chunks])
    prompt = f"""
You are a knowledgeable assistant on world mythology and fantasy texts.

Question:
{query}

Context:
{context}

Answer the question using only the context above. If the context is insufficient, say so.
"""
    return prompt


# ---------- Generate answer ----------
def generate_answer(prompt: str, model: str) -> str:
    response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]


# ---------- RAG pipeline ----------
def rag_query(query: str, model: str, k: int = 3):
    index, metadata = load_index_and_metadata()
    embedder = load_embedder()
    query_vec = embed_query(query, embedder)
    retrieved_chunks = retrieve(query_vec, index, metadata, k=k)
    prompt = build_prompt(query, retrieved_chunks)
    answer = generate_answer(prompt, model=model)
    return answer, retrieved_chunks


# ---------- Streamlit UI ----------
st.set_page_config(page_title="Mythical Worlds", page_icon="📜", layout="wide")

st.title("📜 Mythical Worlds")
st.markdown(
    "Ask questions about myths and legends, powered by Retrieval-Augmented Generation (RAG)."
)

# User input
query = st.text_input(
    "Enter your question:", placeholder="e.g. Who was Odin in Norse mythology?"
)

col1, col2 = st.columns(2)
with col1:
    model = st.selectbox("Choose a model:", ["mistral", "llama3", "phi3"], index=0)
with col2:
    k = st.slider("Number of chunks to retrieve", min_value=1, max_value=10, value=3)

if st.button("Ask"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Searching myths..."):
            answer, sources = rag_query(query, model=model, k=k)

        st.subheader("Answer")
        st.write(answer)

        st.subheader("Sources")
        for s in sources:
            with st.expander(f"{s['title']}"):
                st.write(s["text"])
