import os
import json
import glob
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict

# Paths
PROCESSED_DIR = "data/processed/"
INDEX_DIR = "data/vectorstore/"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # small & fast


def load_chunks(processed_dir: str) -> List[Dict]:
    """Load all JSONL chunks from processed/ directory."""
    chunks = []
    for filepath in glob.glob(os.path.join(processed_dir, "*.jsonl")):
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                chunks.append(json.loads(line.strip()))
    print(f"✅ Loaded {len(chunks)} chunks from {processed_dir}")
    return chunks


def embed_chunks(chunks: List[Dict], model_name: str = MODEL_NAME) -> np.ndarray:
    """Generate embeddings for all chunks."""
    model = SentenceTransformer(model_name)
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    print(f"✅ Generated embeddings with shape {embeddings.shape}")
    return embeddings


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    """Build a FAISS index (L2 similarity)."""
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    print(f"✅ Built FAISS index with {index.ntotal} vectors")
    return index


def save_index(
    index: faiss.IndexFlatL2, chunks: List[Dict], index_dir: str = INDEX_DIR
):
    """Save FAISS index + metadata mapping."""
    os.makedirs(index_dir, exist_ok=True)

    # Save FAISS index
    faiss.write_index(index, os.path.join(index_dir, "faiss.index"))

    # Save metadata (id → title, text)
    metadata = {
        i: {"id": c["id"], "title": c["title"], "text": c["text"]}
        for i, c in enumerate(chunks)
    }
    with open(os.path.join(index_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved FAISS index + metadata to {index_dir}")


if __name__ == "__main__":
    # 1. Load
    chunks = load_chunks(PROCESSED_DIR)
    # 2. Embed
    embeddings = embed_chunks(chunks)
    # 3. Build index
    index = build_faiss_index(embeddings)
    # 4. Save
    save_index(index, chunks)
