# Effects of Chunking on RAG Performance

## ✅ What’s been tested so far

* **Chunking matters (a lot)**

  * Ran controlled experiments on Iliad & Beowulf.
  * Tiny chunks (~200 words) = too fragmented → low hit rates (~0.2).
  * Mid-sized (300–500) = decent sweet spot.
  * Large (~800) sometimes best for long, flowy epics (Iliad loved them).
  * Overlap gave marginal gains but ballooned index size → not worth it most of the time.

* **Out-of-distribution queries**

  * Fed in LOTR / Harry Potter style queries to an Iliad-only index.
  * Retrieval similarity scores dropped noticeably.
  * Clear signal that a cosine similarity threshold could be used for an “I don’t know” mode.

* **Effect of k (top-k retrieval)**

  * Precision@k curve showed the big jump is from k=1 → 2.
  * Plateau (or even degrade) after k>3.
  * More context ≠ better; noise creeps in.

* **Cross-book queries**

  * When multiple books in the index (Iliad, Beowulf, Odyssey, etc.), the retriever sometimes confused sources if chunks were too small (keywords spread out) or too large (topics blended).
  * Reinforces why metadata filtering is a must in multi-book corpora.

* **Hallucination checks (end-to-end)**

  * Wrapped RAG pipeline with a logger.
  * Marked hallucinations if answer used entities not in retrieved context or if top similarity < threshold.
  * Streamlit dashboard now plots hallucination rate, pie chart, and histograms of similarity scores for “hallucinated vs not.”
  * Simple, but already gives a decent readout of when the system is “making stuff up.”

* **Dashboarding**

  * Built a Streamlit visualizer to load experiment results.
  * Can toggle between aggregate vs per-book plots, Precision@k curves, similarity histograms, and hallucination breakdowns.
  * Basically a one-stop shop for chunking experiments.

---

# Effects of Embedding Quality

* **Embedding Showdown (set up)**

  * Experiment plan for MiniLM vs MPNet (later E5).
  * Expect MPNet/E5 to improve disambiguation, raise in/out score gap, and lower hallucination.

---

# ✅ Reranking Experiment — Summary

* **Problem we’re solving:**
  Dense retrieval (FAISS + embeddings) sometimes pulls *semantically close but wrong* chunks → noisy answers, cross-book confusion, and higher hallucination.

* **Solution introduced:**
  Add a **cross-encoder reranker** (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2`):

  1. Retrieve more (top-10/20) with dense vectors.
  2. Rerank candidates with query–chunk pair scoring.
  3. Keep top-k (e.g. 3) after reranking.

* **Expected benefits:**

  * Cleaner top results (esp. in cross-book queries).
  * Lower hallucination rate (wrong-book context gets pushed down).
  * Better separation between in-corpus vs OOD.

* **Tradeoffs:**

  * Adds latency (cross-encoder runs a forward pass per candidate).
  * More compute, but can be batched.
  * Works best when paired with a decent retriever (MiniLM/MPNet/E5).

* **Experiments to run (conceptually):**

  * **Iliad-only:** check baseline vs reranked precision.
  * **Cross-book:** expect major accuracy gains.
  * **OOD:** reranker should down-rank spurious matches.

* **Integration idea:**

  * New flag in pipeline (`use_reranker`).
  * Dashboard plots comparing baseline vs reranked retrieval (Hit@k, MRR, hallucination rates).

---

The next natural step after chunking → embeddings → metadata filtering (conceptually) → reranking is:

**Hybrid Retrieval (Dense + Sparse).**

Dense is great for semantic similarity, but it sometimes misses exact keywords (e.g. “Hector” vs “Hektor”). Sparse methods (BM25, keyword search) handle that well. Hybrid = best of both worlds.

