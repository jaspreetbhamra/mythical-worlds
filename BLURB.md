Been digging into RAGs lately, but instead of treating it as a black-box recipe, went end-to-end on a “fantasy knowledge base” project (think Iliad, Beowulf, Norse sagas from Project Gutenberg). The idea was to actually measure how different design choices affect retrieval quality and hallucination. Some highlights:

* **Pipeline:**

  * Preprocess raw texts (strip Gutenberg headers/footers, normalize, clean OCR).
  * Chunk into fixed windows (200–800 words, variable overlaps).
  * Encode with `sentence-transformers` (MiniLM, later MPNet).
  * Store in FAISS (FlatIP, cosine).
  * Retrieval → top-k chunks → LLM generation (via Ollama).
  * Wrapped with Streamlit for dashboards + experiment visualizations.

* **Experiments & Learnings:**

  * *Chunking:* sweet spot around 300–500 words; too small = fragmented context, too big = muddied retrieval.
  * *Overlap:* minor improvements but ballooned index size — not worth it most of the time.
  * *k retrieval:* best at k=2–3; precision dropped at higher k (noise creeps in).
  * *Out-of-distribution queries:* similarity histograms made it obvious when system was guessing → cosine thresholds gave an “I don’t know” mode.
  * *Cross-book queries:* without metadata, retrieval blurred sources (Beowulf vs Achilles). Reinforces why filters matter.
  * *Embeddings:* MiniLM blurred fine distinctions, MPNet separated characters/events better. Clear retrieval gains from embedding choice.
  * *Reranking (conceptual stage):* cross-encoders can clean up semantically-close-but-wrong hits, especially in multi-book corpora.
  * *Hallucination logging:* built a lightweight eval loop → flagged hallucinations if answer used unsupported entities or low-similarity chunks. Dashboard plots hallucination rate, pie charts, similarity histograms.

* **Tech stack:** Python, FAISS, `sentence-transformers`, Ollama (Mistral and LLAMA3), Streamlit dashboards, YAML query configs, custom eval scripts.

* **Main takeaway:** in RAG, the “plumbing” (chunking, retrieval, reranking, filtering, eval) matters at least as much as the LLM. Even small changes in chunk size or retrieval strategy shifted retrieval precision and hallucination rates by 20–30%.

Next up: scaling experiments (more books, HNSW indexes), adaptive chunking, and trying out formal eval frameworks like RAGAS.

---

