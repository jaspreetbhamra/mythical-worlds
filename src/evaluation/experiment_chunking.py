import argparse
import csv  # moved to top
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List

import faiss
import numpy as np
import yaml
from sentence_transformers import SentenceTransformer


# -----------------------------
# Helpers: I/O and cleaning
# -----------------------------
def load_raw_text(file_path: str) -> str:
    """Load raw Gutenberg .txt file as string."""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def clean_gutenberg(text: str) -> str:
    start_pattern = r"\*\*\* START OF.* \*\*\*"
    end_pattern = r"\*\*\* END OF.* \*\*\*"
    start = re.search(start_pattern, text, re.IGNORECASE)
    end = re.search(end_pattern, text, re.IGNORECASE)
    if start:
        text = text[start.end() :]
    if end:
        text = text[: end.start()]
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


# -----------------------------
# Chunking
# -----------------------------
def chunk_fixed_words(text: str, max_words: int, overlap_words: int) -> List[str]:
    words = text.split()
    step = max_words - overlap_words
    if step <= 0:
        raise ValueError("overlap must be smaller than chunk size")
    chunks = []
    for i in range(0, len(words), step):
        chunk_words = words[i : i + max_words]
        if not chunk_words:
            break
        chunks.append(" ".join(chunk_words))
        if i + max_words >= len(words):
            break
    return chunks


# -----------------------------
# Embeddings + FAISS (cosine)
# -----------------------------
def build_faiss_index_cosine(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
    emb_norm = embeddings / norms
    dim = emb_norm.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb_norm.astype(np.float32))
    return index, emb_norm


# -----------------------------
# Evaluation
# -----------------------------
def hit_positions(
    retrieved_texts: List[str], must_have_keywords: List[str]
) -> List[int]:
    positions = []
    kws = [k.lower() for k in must_have_keywords]
    for i, t in enumerate(retrieved_texts):
        low = t.lower()
        if any(k in low for k in kws):
            positions.append(i)
    return positions


def score_query(index, q_emb, emb_norm, chunk_texts, k, answer_keywords) -> Dict:
    q_emb = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-12)
    t0 = time.time()
    distances, indices = index.search(q_emb.astype(np.float32), k)
    latency = (time.time() - t0) * 1000.0

    top_texts = [chunk_texts[int(i)] for i in indices[0]]
    positions = hit_positions(top_texts, answer_keywords)

    hit_at_k = 1.0 if positions else 0.0
    mrr = 0.0
    if positions:
        first = min(positions)
        mrr = 1.0 / (first + 1)

    return {
        "hit_at_k": hit_at_k,
        "mrr_at_k": mrr,
        "latency_ms": latency,
        "scores": distances[0].tolist(),  # cosine similarities
    }


# -----------------------------
# Experiment runner
# -----------------------------
def run_experiment(
    file_paths: List[str],
    queries_yaml: str,
    out_dir: str,
    model_name: str,
    sizes: List[int],
    overlaps: List[int],
    ks: List[int],
):
    os.makedirs(out_dir, exist_ok=True)

    with open(queries_yaml, "r", encoding="utf-8") as f:
        qspec = yaml.safe_load(f)
    queries = qspec["queries"]

    embedder = SentenceTransformer(model_name)
    all_summary_rows = []

    for file_path in file_paths:
        book_name = Path(file_path).stem
        raw = load_raw_text(file_path)
        text = clean_gutenberg(raw)

        book_summary_rows = []
        for max_words in sizes:
            for ov in overlaps:
                if ov >= max_words:
                    continue
                chunks = chunk_fixed_words(text, max_words, ov)
                emb = embedder.encode(chunks, convert_to_numpy=True)
                index, emb_norm = build_faiss_index_cosine(emb)

                for k in ks:
                    hits, mrrs, latencies = [], [], []
                    detailed_scores = []  # NEW: richer logging

                    for q in queries:
                        qvec = embedder.encode([q["question"]], convert_to_numpy=True)
                        res = score_query(
                            index, qvec, emb_norm, chunks, k, q["keywords"]
                        )

                        hits.append(res["hit_at_k"])
                        mrrs.append(res["mrr_at_k"])
                        latencies.append(res["latency_ms"])

                        # NEW: keep query + domain in scores
                        detailed_scores.append(
                            {
                                "question": q["question"],
                                "domain": q.get("domain", "unknown"),
                                "hit_at_k": res["hit_at_k"],
                                "mrr_at_k": res["mrr_at_k"],
                                "latency_ms": res["latency_ms"],
                                "scores": res["scores"],
                            }
                        )

                    row = {
                        "book": book_name,
                        "chunk_words": max_words,
                        "overlap_words": ov,
                        "n_chunks": len(chunks),
                        "avg_chunk_len_words": (
                            int(np.mean([len(c.split()) for c in chunks]))
                            if chunks
                            else 0
                        ),
                        "hit_at_k_mean": np.mean(hits),
                        "mrr_at_k_mean": np.mean(mrrs),
                        "avg_query_latency_ms": np.mean(latencies),
                        "k": k,
                        "embed_model": model_name,
                    }
                    book_summary_rows.append(row)
                    all_summary_rows.append(row)

                    # Save detailed per-query scores (with domains)
                    detail_path = (
                        Path(out_dir)
                        / f"scores_{book_name}_{max_words}w_{ov}ov_k{k}.json"
                    )
                    with open(detail_path, "w", encoding="utf-8") as f:
                        json.dump({"scores": detailed_scores}, f, indent=2)

                print(
                    f"[{book_name}] {max_words}w/{ov}ov → Hit@{ks} ~ {np.mean(hits):.2f}"
                )

        # Per-book summary
        book_csv = Path(out_dir) / f"summary_{book_name}.csv"
        with open(book_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(book_summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(book_summary_rows)

    # Aggregate summary
    agg_csv = Path(out_dir) / "summary_all_books.csv"
    with open(agg_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_summary_rows)

    print(f"\n✅ Saved summaries to {out_dir}")


# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Chunking experiment for RAG retrieval quality."
    )
    parser.add_argument(
        "--input", required=True, help="Path to a .txt file or folder of .txt files"
    )
    parser.add_argument(
        "--queries", required=True, help="YAML file with queries and answer keywords."
    )
    parser.add_argument("--out", default="data/experiments/chunking/")
    parser.add_argument(
        "--embed_model", default="sentence-transformers/all-MiniLM-L6-v2"
    )
    parser.add_argument(
        "--sizes", default="200,300,500,800", help="Comma-separated chunk sizes (words)"
    )
    parser.add_argument(
        "--overlaps", default="0,50,100", help="Comma-separated overlaps (words)"
    )
    parser.add_argument(
        "--ks", default="3", help="Comma-separated k values for retrieval evaluation"
    )
    args = parser.parse_args()

    p = Path(args.input)
    if p.is_dir():
        file_paths = sorted(p.glob("*.txt"))
    else:
        file_paths = [p]

    sizes = [int(x) for x in args.sizes.split(",")]
    overlaps = [int(x) for x in args.overlaps.split(",")]
    ks = [int(x) for x in args.ks.split(",")]

    run_experiment(
        file_paths=file_paths,
        queries_yaml=args.queries,
        out_dir=args.out,
        model_name=args.embed_model,
        sizes=sizes,
        overlaps=overlaps,
        ks=ks,
    )
