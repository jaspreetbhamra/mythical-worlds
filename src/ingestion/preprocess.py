import os
from pathlib import Path
import re
import json
from typing import List

DATA_DIR = Path("data/raw/")
OUTPUT_DIR = Path("data/processed/")


# ----------- STEP 1: Load raw text -----------
def load_text(file_path: str) -> str:
    """Load raw text file as string."""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


# ----------- STEP 2: Clean Gutenberg headers/footers -----------
def clean_gutenberg_text(text: str) -> str:
    """
    Remove Project Gutenberg boilerplate headers and footers.
    Looks for START/END markers.
    """
    start_pattern = r"\*\*\* START OF.* \*\*\*"
    end_pattern = r"\*\*\* END OF.* \*\*\*"

    # Find start
    start_match = re.search(start_pattern, text, re.IGNORECASE)
    if start_match:
        text = text[start_match.end() :]

    # Find end
    end_match = re.search(end_pattern, text, re.IGNORECASE)
    if end_match:
        text = text[: end_match.start()]

    return text.strip()


# ----------- STEP 3: Chunking -----------
def chunk_text(text: str, max_words: int = 200) -> List[str]:
    """
    Split text into chunks of ~max_words words.
    Adjust max_words depending on downstream embedding/tokenizer.
    """
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i : i + max_words])
        chunks.append(chunk)
    return chunks


# ----------- STEP 4: Save processed chunks -----------
def save_chunks(chunks: List[str], title: str, output_dir: Path = OUTPUT_DIR):
    """Save chunks into JSONL (one record per line)."""
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, f"{title}.jsonl")

    with open(out_file, "w", encoding="utf-8") as f:
        for idx, chunk in enumerate(chunks):
            record = {"id": f"{title}_{idx}", "title": title, "text": chunk}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"✅ Saved {len(chunks)} chunks to {out_file}")


# ----------- STEP 5: Full pipeline -----------
def preprocess_file(file_path: str, max_words: int = 200, out_dir: Path = OUTPUT_DIR):
    """Run full pipeline: load → clean → chunk → save."""
    title = os.path.splitext(os.path.basename(file_path))[0]
    raw_text = load_text(file_path)
    clean_text = clean_gutenberg_text(raw_text)
    chunks = chunk_text(clean_text, max_words=max_words)
    save_chunks(chunks, title, out_dir)


if __name__ == "__main__":
    # Process all files in data/raw/
    for fname in os.listdir(DATA_DIR):
        if fname.endswith(".txt"):
            preprocess_file(os.path.join(DATA_DIR, fname), max_words=200)
