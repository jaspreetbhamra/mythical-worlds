import argparse
import json

import faiss
import numpy as np
import ollama
from sentence_transformers import SentenceTransformer

# Paths
INDEX_PATH = "../data/vector_store/faiss.index"
METADATA_PATH = "../data/vector_store/metadata.json"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ---------- Load FAISS + metadata ----------
def load_index_and_metadata(index_path=INDEX_PATH, metadata_path=METADATA_PATH):
    index = faiss.read_index(index_path)
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return index, metadata


# ---------- Embed query ----------
def embed_query(query: str, model_name: str = EMBED_MODEL) -> np.ndarray:
    model = SentenceTransformer(model_name)
    vec = model.encode([query], convert_to_numpy=True)
    return vec


# ---------- Retrieve top-k chunks ----------
def retrieve(query_vec: np.ndarray, index, metadata, k: int = 3):
    distances, indices = index.search(query_vec, k)
    results = []
    for rank, idx in enumerate(indices[0]):
        idx = str(idx)
        if idx in metadata:
            results.append(
                {
                    "rank": rank,
                    "score": float(
                        distances[0][rank]
                    ),  # cosine similarity (since we used normalized vectors)
                    "text": metadata[idx]["text"],
                    "title": metadata[idx].get("title", "N/A"),
                }
            )
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


# ---------- Full pipeline ----------
def rag_query(query: str, model: str, k: int = 3):
    index, metadata = load_index_and_metadata()
    query_vec = embed_query(query)
    retrieved_chunks = retrieve(query_vec, index, metadata, k=k)
    prompt = build_prompt(query, retrieved_chunks)
    answer = generate_answer(prompt, model=model)
    return answer, retrieved_chunks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RAG pipeline for Mythical Worlds Explorer"
    )
    parser.add_argument("query", type=str, help="User query (in quotes)")
    parser.add_argument(
        "--model",
        type=str,
        default="mistral",
        help="Ollama model to use (e.g., mistral, llama3, phi3)",
    )
    parser.add_argument("--k", type=int, default=3, help="Number of chunks to retrieve")
    args = parser.parse_args()

    answer, chunks = rag_query(args.query, model=args.model, k=args.k)

    print("\n=== Answer ===\n")
    print(answer)

    print("\n=== Sources ===\n")
    for c in chunks:
        print(f"[{c['title']}] {c['text'][:200]}...")
