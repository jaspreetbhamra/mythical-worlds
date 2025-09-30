import argparse
import re
from pathlib import Path

import pandas as pd
import yaml
from rag_pipeline import (
    rag_query,
)  # assumes your rag_query(query, model, k) returns (answer, retrieved_chunks)


# -----------------------------
# Simple hallucination checks
# -----------------------------
def check_hallucination(
    answer: str, retrieved_chunks: list, query: str, threshold: float = 0.6
) -> bool:
    """
    Returns True if hallucination is detected.
    Heuristics:
      1. If no retrieved chunks → hallucination.
      2. If answer mentions named entities not present in retrieved text → hallucination.
      3. If answer length > 0 but top retrieval similarity < threshold (to add later when logging scores).
    """
    # Combine context text
    context = " ".join([c["text"] for c in retrieved_chunks]).lower()

    # Entities from answer (simple heuristic: capitalized words)
    entities = re.findall(r"\b[A-Z][a-z]+\b", answer)

    unsupported = [e for e in entities if e.lower() not in context]
    if not retrieved_chunks:
        return True
    if unsupported:
        return True

    return False


# -----------------------------
# Main eval function
# -----------------------------
def evaluate_queries(queries_yaml: str, model: str, k: int, out_path: str):
    with open(queries_yaml, "r", encoding="utf-8") as f:
        queries = yaml.safe_load(f)["queries"]

    results = []
    for q in queries:
        question = q["question"]
        answer, retrieved = rag_query(question, model=model, k=k)

        halluc = check_hallucination(answer, retrieved, question)

        results.append(
            {
                "question": question,
                "answer": answer,
                "retrieved_titles": [c.get("title", "N/A") for c in retrieved],
                "retrieved_texts": [c["text"] for c in retrieved],
                "hallucinated": halluc,
            }
        )

        print(f"Q: {question}\nHallucinated: {halluc}\n---")

    df = pd.DataFrame(results)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\n✅ Saved hallucination eval results to {out_path}")


# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", required=True, help="Path to queries YAML")
    parser.add_argument("--model", default="mistral", help="Ollama model name")
    parser.add_argument("--k", type=int, default=3, help="Top-k for retrieval")
    parser.add_argument("--out", default="data/experiments/hallucination/results.csv")
    args = parser.parse_args()

    evaluate_queries(args.queries, args.model, args.k, args.out)
