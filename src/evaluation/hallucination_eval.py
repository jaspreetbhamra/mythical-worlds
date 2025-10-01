import argparse
from pathlib import Path

import pandas as pd
import spacy
import yaml
from sentence_transformers import SentenceTransformer, util

from retrieval.rag_pipeline import rag_query

# Load spaCy for NER
# python -m spacy download en_core_web_sm
nlp = spacy.load("en_core_web_sm")

# Load a sentence-transformer for semantic similarity
embedder = SentenceTransformer("all-MiniLM-L6-v2")  # small & fast


def extract_entities_spacy(text: str) -> list:
    """Extract named entities using spaCy NER."""
    doc = nlp(text)
    return [ent.text for ent in doc.ents]  # keep original casing for embeddings


def entity_in_context(entity: str, context: str, sim_threshold: float = 0.75) -> bool:
    """Check if entity is present in context (string or semantic match)."""
    entity_lower = entity.lower()
    if entity_lower in context.lower():
        return True

    # Semantic similarity fallback
    ent_vec = embedder.encode(entity, convert_to_tensor=True)
    ctx_vec = embedder.encode(context, convert_to_tensor=True)

    sim = util.cos_sim(ent_vec, ctx_vec).item()
    return sim >= sim_threshold


def check_hallucination(
    answer: str, retrieved_chunks: list, threshold: float = 0.6
) -> bool:
    """
    Returns True if hallucination is detected.
    Heuristics:
      1. If no retrieved chunks → hallucination.
      2. If answer entities not supported by context → hallucination.
      3. If top retrieval similarity < threshold → hallucination.
    """
    if not retrieved_chunks:
        return True

    # Combine context
    context = " ".join([c["text"] for c in retrieved_chunks])

    # Extract entities with spaCy
    entities = extract_entities_spacy(answer)

    unsupported = []
    for e in entities:
        if not entity_in_context(e, context):
            unsupported.append(e)

    # Retrieval similarity threshold check
    top_score = max([c["score"] for c in retrieved_chunks]) if retrieved_chunks else 0.0
    below_thresh = top_score < threshold

    return bool(unsupported or below_thresh)


# -----------------------------
# Main eval function
# -----------------------------
def evaluate_queries(
    queries_yaml: str, model: str, k: int, out_path: str, threshold: float
):
    with open(queries_yaml, "r", encoding="utf-8") as f:
        queries = yaml.safe_load(f)["queries"]

    results = []
    for q in queries:
        question = q["question"]
        answer, retrieved = rag_query(question, model=model, k=k)

        halluc = check_hallucination(answer, retrieved, threshold=threshold)

        results.append(
            {
                "question": question,
                "answer": answer,
                "retrieved_titles": [c.get("title", "N/A") for c in retrieved],
                "retrieved_texts": [c["text"] for c in retrieved],
                "retrieved_scores": [c["score"] for c in retrieved],
                "top_score": max([c["score"] for c in retrieved]) if retrieved else 0.0,
                "hallucinated": halluc,
            }
        )

        print(
            f"Q: {question}\nTop score: {results[-1]['top_score']:.2f} | Hallucinated: {halluc}\n---"
        )

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
    parser.add_argument(
        "--threshold", type=float, default=0.6, help="Cosine similarity threshold"
    )
    args = parser.parse_args()

    evaluate_queries(args.queries, args.model, args.k, args.out, args.threshold)
