import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from scipy.stats import ks_2samp, ttest_ind
from sklearn.metrics import auc, roc_curve

sns.set_theme(style="whitegrid")


# -----------------------------
# Load data
# -----------------------------
def load_results(results_dir: str):
    results_path = Path(results_dir)
    agg = pd.read_csv(results_path / "summary_all_books.csv")
    book_summaries = {}
    for csv_file in results_path.glob("summary_*.csv"):
        if csv_file.name == "summary_all_books.csv":
            continue
        book = csv_file.stem.replace("summary_", "")
        book_summaries[book] = pd.read_csv(csv_file)
    return agg, book_summaries


def load_scores(results_dir: str, in_distribution_domains=None, reduce="top1", kth=1):
    """
    Returns two 1-D lists of floats: scores_in, scores_out.

    reduce:
      - "top1" : use the highest-sim score for each query
      - "mean" : mean of the top-k scores per query file
      - "kth"  : use the kth score (1-indexed)
      - "all"  : flatten all top-k scores across queries
    """
    if in_distribution_domains is None:
        in_distribution_domains = ["iliad"]  # customize in your UI

    scores_in, scores_out = [], []

    for f in Path(results_dir).glob("scores_*.json"):
        with open(f, "r", encoding="utf-8") as j:
            data = json.load(j)

        for entry in data.get("scores", []):
            sims = entry.get("scores", [])  # list of top-k sims for this query
            if not sims:
                continue

            # reduce per-query list to a scalar or flattened list
            if reduce == "top1":
                vals = [float(sims[0])]
            elif reduce == "mean":
                vals = [float(np.mean(sims))]
            elif reduce == "kth":
                # 1-indexed kth
                if 1 <= kth <= len(sims):
                    vals = [float(sims[kth - 1])]
                else:
                    continue
            elif reduce == "all":
                vals = [float(x) for x in sims]
            else:
                # default to top1
                vals = [float(sims[0])]

            domain = str(entry.get("domain", "unknown")).lower()
            target = scores_in if domain in in_distribution_domains else scores_out
            target.extend(vals)

    return scores_in, scores_out


# -----------------------------
# Visualization functions
# -----------------------------
def plot_hit_vs_chunk(df, title="Hit@k vs Chunk Size"):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(
        data=df,
        x="chunk_words",
        y="hit_at_k_mean",
        hue="overlap_words",
        marker="o",
        ax=ax,
    )
    ax.set_title(title)
    ax.set_ylabel("Hit@k mean")
    ax.set_xlabel("Chunk size (words)")
    return fig


def plot_mrr_vs_chunk(df, title="MRR vs Chunk Size"):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(
        data=df,
        x="chunk_words",
        y="mrr_at_k_mean",
        hue="overlap_words",
        marker="o",
        ax=ax,
    )
    ax.set_title(title)
    ax.set_ylabel("MRR mean")
    ax.set_xlabel("Chunk size (words)")
    return fig


def plot_precision_k(df, title="Precision@k Curve"):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(
        data=df,
        x="k",
        y="hit_at_k_mean",
        hue="chunk_words",
        style="overlap_words",
        marker="o",
        ax=ax,
    )
    ax.set_title(title)
    ax.set_ylabel("Hit@k mean")
    ax.set_xlabel("k")
    return fig


def plot_similarity_hist(scores_in, scores_out):
    fig, ax = plt.subplots(figsize=(8, 5))

    # Histogram as density
    sns.histplot(
        scores_in,
        bins=30,
        kde=True,
        stat="density",
        color="green",
        label="In-corpus",
        alpha=0.5,
        ax=ax,
    )
    sns.histplot(
        scores_out,
        bins=30,
        kde=True,
        stat="density",
        color="red",
        label="Out-of-corpus",
        alpha=0.5,
        ax=ax,
    )

    # Mean lines
    if scores_in:
        ax.axvline(pd.Series(scores_in).mean(), color="green", linestyle="--")
    if scores_out:
        ax.axvline(pd.Series(scores_out).mean(), color="red", linestyle="--")

    ax.set_title("Cosine Similarity Distributions")
    ax.set_xlabel("Cosine similarity")
    ax.set_ylabel("Density")
    ax.legend()
    return fig


def plot_similarity_hist_with_stats(scores_in, scores_out, show_kde=True):
    # Ensure 1-D float arrays, clip to cosine range
    in_arr = pd.Series(scores_in, dtype=float).dropna().clip(-1, 1)
    out_arr = pd.Series(scores_out, dtype=float).dropna().clip(-1, 1)

    fig, ax = plt.subplots(figsize=(8, 5))

    # Draw histograms (density normalized, no KDE here)
    sns.histplot(
        in_arr,
        bins=30,
        stat="density",
        alpha=0.45,
        label=f"In (n={len(in_arr)})",
        ax=ax,
    )
    sns.histplot(
        out_arr,
        bins=30,
        stat="density",
        alpha=0.45,
        label=f"Out (n={len(out_arr)})",
        ax=ax,
    )

    # Optional KDE overlays (once per group)
    if show_kde and len(in_arr) > 1:
        sns.kdeplot(in_arr, ax=ax)
    if show_kde and len(out_arr) > 1:
        sns.kdeplot(out_arr, ax=ax)

    # Means
    if len(in_arr) > 0:
        ax.axvline(in_arr.mean(), linestyle="--")
    if len(out_arr) > 0:
        ax.axvline(out_arr.mean(), linestyle="--")

    ax.set_title("Cosine Similarity Distributions (In vs Out)")
    ax.set_xlabel("Cosine similarity")
    ax.set_ylabel("Density")
    ax.legend()

    # Statistical tests
    ks_p, t_p, d = None, None, None
    if len(in_arr) > 1 and len(out_arr) > 1:
        ks_stat, ks_p = ks_2samp(in_arr, out_arr)
        t_stat, t_p = ttest_ind(in_arr, out_arr, equal_var=False)

        # Cohen's d (effect size)
        def cohens_d(a, b):
            a, b = np.asarray(a), np.asarray(b)
            na, nb = len(a), len(b)
            if na < 2 or nb < 2:
                return np.nan
            sa, sb = a.std(ddof=1), b.std(ddof=1)
            # pooled SD
            s = np.sqrt(((na - 1) * sa**2 + (nb - 1) * sb**2) / (na + nb - 2))
            return (a.mean() - b.mean()) / s if s > 0 else np.nan

        d = float(cohens_d(in_arr, out_arr))

    return fig, ks_p, t_p, d


def compute_optimal_threshold(scores_in, scores_out, plot=True):
    """
    Compute the best cosine similarity threshold to separate in vs out.
    Returns: threshold, fpr, tpr, roc_auc
    """

    # Build labels: 1 = in-distribution, 0 = out-of-distribution
    y_true = np.array([1] * len(scores_in) + [0] * len(scores_out))
    y_scores = np.array(scores_in + scores_out)

    # ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    # Youden’s J statistic = max(tpr - fpr)
    j_scores = tpr - fpr
    j_best_idx = np.argmax(j_scores)
    best_threshold = thresholds[j_best_idx]

    if plot:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
        ax.plot([0, 1], [0, 1], "k--")
        ax.scatter(
            fpr[j_best_idx],
            tpr[j_best_idx],
            color="red",
            label=f"Best thr={best_threshold:.2f}",
        )
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve for In vs Out")
        ax.legend()
        plt.show()

    return best_threshold, fpr, tpr, roc_auc


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Chunking Experiment Visualizer", layout="wide")

st.title("📊 RAG Chunking Experiment Visualizer")

results_dir = st.text_input(
    "Path to results folder", "data/experiments/chunking/iliad_beowulf_minilm/"
)

if Path(results_dir).exists():
    agg, book_summaries = load_results(results_dir)

    # in sidebar
    st.sidebar.header("Settings")
    in_books = st.sidebar.multiselect(
        "Choose in-distribution books",
        options=["iliad", "odyssey", "beowulf", "aeneid", "metamorphoses"],
        default=["iliad", "beowulf"],
    )

    st.header("Aggregate Results")
    st.dataframe(agg.head())

    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(plot_hit_vs_chunk(agg, "Aggregate Hit@k vs Chunk Size"))
    with col2:
        st.pyplot(plot_mrr_vs_chunk(agg, "Aggregate MRR vs Chunk Size"))

    st.pyplot(plot_precision_k(agg, "Aggregate Precision@k Curve"))

    st.header("Per-Book Results")
    for book, df in book_summaries.items():
        st.subheader(book)
        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(plot_hit_vs_chunk(df, f"{book}: Hit@k vs Chunk Size"))
        with col2:
            st.pyplot(plot_mrr_vs_chunk(df, f"{book}: MRR vs Chunk Size"))

    st.sidebar.header("Similarity Settings")
    reduce = st.sidebar.selectbox(
        "Similarity reduction per query", ["top1", "mean", "kth", "all"], index=0
    )
    kth = st.sidebar.number_input(
        "k for 'kth' reduction (1-indexed)", min_value=1, value=3
    )

    in_domains = st.sidebar.multiselect(
        "In-distribution domains",
        options=["iliad", "beowulf", "odyssey", "aeneid", "metamorphoses"],
        default=["iliad"],
    )

    scores_in, scores_out = load_scores(
        results_dir, in_distribution_domains=in_domains, reduce=reduce, kth=kth
    )

    if scores_in and scores_out:
        fig, ks_p, t_p, d = plot_similarity_hist_with_stats(
            scores_in, scores_out, show_kde=True
        )
        st.pyplot(fig)

        # Compute threshold
        thr, fpr, tpr, roc_auc = compute_optimal_threshold(
            scores_in, scores_out, plot=False
        )
        st.markdown("### Threshold Analysis")
        st.write(f"**Optimal threshold (Youden’s J):** {thr:.3f}")
        st.write(f"**ROC AUC:** {roc_auc:.3f}")

        # Optional: display ROC curve inline
        fig2, ax2 = plt.subplots(figsize=(5, 5))
        ax2.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
        ax2.plot([0, 1], [0, 1], "k--")
        ax2.axvline(x=fpr[np.argmax(tpr - fpr)], linestyle="--", color="red")
        ax2.axhline(y=tpr[np.argmax(tpr - fpr)], linestyle="--", color="red")
        ax2.set_xlabel("False Positive Rate")
        ax2.set_ylabel("True Positive Rate")
        ax2.set_title("ROC Curve")
        ax2.legend()
        st.pyplot(fig2)

        st.markdown("### Statistical Tests")
        if ks_p is not None:
            st.write(f"**KS test p-value:** {ks_p:.4f}")
        if t_p is not None:
            st.write(f"**Welch’s t-test p-value:** {t_p:.4f}")
        if d is not None:
            st.write(f"**Cohen’s d (effect size):** {d:.3f}")
    else:
        st.info("No similarity scores available yet.")

else:
    st.warning("Results folder not found. Run experiment_chunking.py first.")
