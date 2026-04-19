"""
Retriever evaluation script for the CKD RAG system.

Evaluates retrieval effectiveness by running queries against the vectorstore
and comparing retrieved document sources to ground-truth relevant sources.

Produces:
- Per-query confusion matrix (TP, FP, FN, TN)
- Aggregate confusion matrix
- Precision, Recall, F1, Accuracy per query and overall
- Breakdown by topic
- Visual confusion matrix plot (saved to tests/retriever_confusion_matrix.png)

Usage:
    python tests/eval_retriever.py                    # default: CKDRetriever, k=5
    python tests/eval_retriever.py --k 10             # increase k
    python tests/eval_retriever.py --retriever tree   # tree retriever
    python tests/eval_retriever.py --retriever hybrid # hybrid retriever
    python tests/eval_retriever.py --no-plot          # skip matplotlib plot
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

# Project root
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import EMBEDDING_DIMENSION

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

DATASET_PATH = Path(__file__).parent / "retriever_eval_dataset.json"
OUTPUT_DIR = Path(__file__).parent


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class ConfusionCounts:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def accuracy(self) -> float:
        total = self.tp + self.fp + self.fn + self.tn
        return (self.tp + self.tn) / total if total else 0.0


@dataclass
class QueryResult:
    query_id: str
    query: str
    topic: str
    relevant_sources: list[str]
    retrieved_sources: list[str]
    confusion: ConfusionCounts = field(default_factory=ConfusionCounts)


# ── Core evaluation ──────────────────────────────────────────────────────────

def load_dataset(path: Path = DATASET_PATH) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    return data["entries"]


def build_retriever(retriever_type: str, k: int):
    """Build the retriever and return (retriever, all_sources)."""
    from simple_rag.embeddings import EmbeddingGemmaWrapper
    from simple_rag.vectorstore import CKDVectorStore
    from simple_rag.retriever import create_retriever

    embeddings = EmbeddingGemmaWrapper(dimension=EMBEDDING_DIMENSION)
    vectorstore = CKDVectorStore(embedding_function=embeddings)

    # Collect all unique source names from the vectorstore
    stats = vectorstore.get_collection_stats()
    # Get all sources from the collection for TN calculation
    collection = vectorstore._client.get_collection(vectorstore.collection_name)
    all_meta = collection.get(include=["metadatas"])
    all_sources = set()
    for meta in all_meta.get("metadatas", []):
        if meta and "source" in meta:
            all_sources.add(meta["source"])

    retriever = create_retriever(
        vectorstore=vectorstore,
        k=k,
        use_hybrid=(retriever_type == "hybrid"),
        use_tree=(retriever_type == "tree"),
        embedding_function=embeddings if retriever_type == "tree" else None,
    )
    return retriever, all_sources


def evaluate_query(
    entry: dict,
    retriever,
    all_sources: set[str],
) -> QueryResult:
    """Run a single query and compute confusion counts at the source level."""
    query = entry["query"]
    relevant_set = set(entry["relevant_sources"])

    # Retrieve
    docs = retriever.invoke(query)
    retrieved_set = {doc.metadata.get("source", "") for doc in docs}

    # Confusion matrix (source-level, binary: relevant or not)
    tp = len(retrieved_set & relevant_set)
    fp = len(retrieved_set - relevant_set)
    fn = len(relevant_set - retrieved_set)
    tn = len(all_sources - retrieved_set - relevant_set)

    return QueryResult(
        query_id=entry["id"],
        query=query,
        topic=entry.get("topic", "unknown"),
        relevant_sources=list(relevant_set),
        retrieved_sources=list(retrieved_set),
        confusion=ConfusionCounts(tp=tp, fp=fp, fn=fn, tn=tn),
    )


def aggregate(results: list[QueryResult]) -> ConfusionCounts:
    agg = ConfusionCounts()
    for r in results:
        agg.tp += r.confusion.tp
        agg.fp += r.confusion.fp
        agg.fn += r.confusion.fn
        agg.tn += r.confusion.tn
    return agg


def aggregate_by_topic(results: list[QueryResult]) -> dict[str, ConfusionCounts]:
    topics: dict[str, ConfusionCounts] = defaultdict(ConfusionCounts)
    for r in results:
        t = topics[r.topic]
        t.tp += r.confusion.tp
        t.fp += r.confusion.fp
        t.fn += r.confusion.fn
        t.tn += r.confusion.tn
    return dict(topics)


# ── Reporting ────────────────────────────────────────────────────────────────

def print_report(results: list[QueryResult], retriever_type: str, k: int):
    header = f"Retriever Evaluation Report — {retriever_type} (k={k})"
    print(f"\n{'=' * len(header)}")
    print(header)
    print(f"{'=' * len(header)}")

    # Per-query table
    print(f"\n{'ID':<35} {'TP':>4} {'FP':>4} {'FN':>4} {'TN':>4}  {'Prec':>6} {'Rec':>6} {'F1':>6}")
    print("-" * 85)
    for r in results:
        c = r.confusion
        print(
            f"{r.query_id:<35} {c.tp:4d} {c.fp:4d} {c.fn:4d} {c.tn:4d}"
            f"  {c.precision:6.3f} {c.recall:6.3f} {c.f1:6.3f}"
        )

    # Topic breakdown
    topic_agg = aggregate_by_topic(results)
    print(f"\n{'Topic':<20} {'TP':>4} {'FP':>4} {'FN':>4} {'TN':>4}  {'Prec':>6} {'Rec':>6} {'F1':>6}")
    print("-" * 70)
    for topic in sorted(topic_agg):
        c = topic_agg[topic]
        print(
            f"{topic:<20} {c.tp:4d} {c.fp:4d} {c.fn:4d} {c.tn:4d}"
            f"  {c.precision:6.3f} {c.recall:6.3f} {c.f1:6.3f}"
        )

    # Aggregate
    agg = aggregate(results)
    print(f"\n{'─' * 40}")
    print(f"AGGREGATE CONFUSION MATRIX")
    print(f"{'─' * 40}")
    print(f"               Predicted +   Predicted -")
    print(f"  Actual +     TP={agg.tp:<6d}   FN={agg.fn:<6d}")
    print(f"  Actual -     FP={agg.fp:<6d}   TN={agg.tn:<6d}")
    print(f"{'─' * 40}")
    print(f"  Precision : {agg.precision:.4f}")
    print(f"  Recall    : {agg.recall:.4f}")
    print(f"  F1 Score  : {agg.f1:.4f}")
    print(f"  Accuracy  : {agg.accuracy:.4f}")
    print(f"  Queries   : {len(results)}")
    print()

    # Missed sources detail
    print("MISSED SOURCES (FN detail):")
    for r in results:
        missed = set(r.relevant_sources) - set(r.retrieved_sources)
        if missed:
            print(f"  {r.query_id}:")
            for s in sorted(missed):
                print(f"    - {s}")
    print()


def save_results_json(results: list[QueryResult], retriever_type: str, k: int):
    out_path = OUTPUT_DIR / "retriever_eval_results.json"
    agg = aggregate(results)
    payload = {
        "config": {"retriever": retriever_type, "k": k},
        "aggregate": {
            "tp": agg.tp, "fp": agg.fp, "fn": agg.fn, "tn": agg.tn,
            "precision": round(agg.precision, 4),
            "recall": round(agg.recall, 4),
            "f1": round(agg.f1, 4),
            "accuracy": round(agg.accuracy, 4),
        },
        "per_query": [
            {
                "id": r.query_id,
                "query": r.query,
                "topic": r.topic,
                "relevant_sources": r.relevant_sources,
                "retrieved_sources": r.retrieved_sources,
                "tp": r.confusion.tp,
                "fp": r.confusion.fp,
                "fn": r.confusion.fn,
                "tn": r.confusion.tn,
                "precision": round(r.confusion.precision, 4),
                "recall": round(r.confusion.recall, 4),
                "f1": round(r.confusion.f1, 4),
            }
            for r in results
        ],
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Results saved to {out_path}")


def plot_confusion_matrix(results: list[QueryResult], retriever_type: str, k: int):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not installed — skipping plot.")
        return

    agg = aggregate(results)
    matrix = np.array([[agg.tp, agg.fn], [agg.fp, agg.tn]])
    labels = np.array([
        [f"TP\n{agg.tp}", f"FN\n{agg.fn}"],
        [f"FP\n{agg.fp}", f"TN\n{agg.tn}"],
    ])

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: aggregate confusion matrix
    ax = axes[0]
    im = ax.imshow(matrix, cmap="Blues", aspect="auto")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted Relevant", "Predicted Not Relevant"])
    ax.set_yticklabels(["Actually Relevant", "Actually Not Relevant"])
    ax.set_title(
        f"Aggregate Confusion Matrix\n{retriever_type} (k={k})\n"
        f"P={agg.precision:.3f}  R={agg.recall:.3f}  F1={agg.f1:.3f}"
    )
    for i in range(2):
        for j in range(2):
            ax.text(j, i, labels[i, j], ha="center", va="center",
                    fontsize=14, fontweight="bold",
                    color="white" if matrix[i, j] > matrix.max() / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)

    # Right: per-topic F1 bar chart
    ax2 = axes[1]
    topic_agg = aggregate_by_topic(results)
    topics = sorted(topic_agg.keys())
    f1s = [topic_agg[t].f1 for t in topics]
    precisions = [topic_agg[t].precision for t in topics]
    recalls = [topic_agg[t].recall for t in topics]

    x = np.arange(len(topics))
    width = 0.25
    ax2.bar(x - width, precisions, width, label="Precision", color="#2196F3")
    ax2.bar(x, recalls, width, label="Recall", color="#FF9800")
    ax2.bar(x + width, f1s, width, label="F1", color="#4CAF50")
    ax2.set_xticks(x)
    ax2.set_xticklabels(topics, rotation=45, ha="right")
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("Score")
    ax2.set_title("Per-Topic Retrieval Metrics")
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out_path = OUTPUT_DIR / "retriever_confusion_matrix.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix plot saved to {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate CKD RAG retriever")
    parser.add_argument("--k", type=int, default=5, help="Number of documents to retrieve (default 5)")
    parser.add_argument(
        "--retriever", choices=["basic", "hybrid", "tree"], default="basic",
        help="Retriever type (default: basic)",
    )
    parser.add_argument("--no-plot", action="store_true", help="Skip matplotlib plot")
    args = parser.parse_args()

    print(f"Loading evaluation dataset from {DATASET_PATH} ...")
    entries = load_dataset()
    print(f"  {len(entries)} queries loaded")

    print(f"Building {args.retriever} retriever (k={args.k}) ...")
    retriever, all_sources = build_retriever(args.retriever, args.k)
    print(f"  {len(all_sources)} unique sources in vectorstore")

    print(f"Running evaluation ...")
    results: list[QueryResult] = []
    for i, entry in enumerate(entries, 1):
        print(f"  [{i}/{len(entries)}] {entry['id']}", end=" ... ", flush=True)
        result = evaluate_query(entry, retriever, all_sources)
        results.append(result)
        c = result.confusion
        print(f"TP={c.tp} FP={c.fp} FN={c.fn} P={c.precision:.2f} R={c.recall:.2f}")

    print_report(results, args.retriever, args.k)
    save_results_json(results, args.retriever, args.k)

    if not args.no_plot:
        plot_confusion_matrix(results, args.retriever, args.k)


if __name__ == "__main__":
    main()
