"""
Run RAGAS evaluation comparing flat (CKDRetriever) vs tree-based (TreeRetriever).

Usage:
    uv run python eval/run_retriever_comparison.py
    uv run python eval/run_retriever_comparison.py --queries eval/test_queries.json
    uv run python eval/run_retriever_comparison.py --retriever flat
    uv run python eval/run_retriever_comparison.py --retriever tree
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("RAGAS_STREAM_JUDGE", "true")

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_llm, get_embeddings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent / "results"
DEFAULT_QUERIES = Path(__file__).parent / "test_queries.json"


def extract_contexts(docs: list) -> list[str]:
    """Extract page_content strings from LangChain Document objects."""
    return [doc.page_content for doc in docs]


def aggregate_scores(scores_list: list) -> dict[str, float]:
    """Compute mean of each RAGAS metric across a list of RAGASScores."""
    if not scores_list:
        return {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_precision": 0.0,
            "context_recall": 0.0,
            "average": 0.0,
        }

    n = len(scores_list)
    totals = {
        "faithfulness": sum(s.faithfulness for s in scores_list) / n,
        "answer_relevancy": sum(s.answer_relevancy for s in scores_list) / n,
        "context_precision": sum(s.context_precision for s in scores_list) / n,
        "context_recall": sum(s.context_recall for s in scores_list) / n,
    }
    totals["average"] = sum(totals.values()) / 4
    return totals


def load_queries(path: Path) -> list[dict]:
    """Load test queries from JSON file."""
    data = json.loads(path.read_text())
    return data["queries"]


def run_single_query(query_data: dict, retriever, llm, evaluator) -> dict:
    """Run retrieval + generation + evaluation for one query."""
    query = query_data["query"]
    reference = query_data.get("reference")

    docs = retriever.invoke(query)
    contexts = extract_contexts(docs)

    from config import RAG_PROMPT_TEMPLATE, build_system_prompt, load_patient_context

    system_prompt = build_system_prompt(load_patient_context())
    context_str = "\n\n---\n\n".join(
        f"[{i+1}] {doc.metadata.get('source', 'Unknown')}\n{doc.page_content}"
        for i, doc in enumerate(docs)
    )
    prompt = f"{system_prompt}\n\n{RAG_PROMPT_TEMPLATE.format(context=context_str, question=query)}"

    print(f"\n{'-'*65}")
    print(f"Query [{query_data['id']}]: {query}")
    print(f"{'-'*65}")
    print("Model Response: ", end="", flush=True)

    answer_parts = []
    try:
        for chunk in llm.as_langchain_llm().stream(prompt):
            text = getattr(chunk, "content", None) or str(chunk)
            if text:
                print(text, end="", flush=True)
                answer_parts.append(text)
        answer = "".join(answer_parts)
    except (AttributeError, NotImplementedError):
        answer = llm.generate(prompt)
        print(answer, end="", flush=True)

    print(f"\n{'-'*65}\n", flush=True)

    scores = evaluator.evaluate(
        query=query,
        response=answer,
        contexts=contexts,
        reference=reference,
    )

    return {
        "query_id": query_data["id"],
        "query": query,
        "category": query_data["category"],
        "reference": reference,
        "num_docs_retrieved": len(docs),
        "contexts": contexts,
        "answer": answer,
        "scores": scores.to_dict(),
    }


def run_evaluation(queries: list[dict], retriever, retriever_name: str, llm, evaluator) -> dict:
    """Run full evaluation for one retriever across all queries."""
    from importlib import import_module
    ragas_mod = import_module("agentic_rag.evaluation.ragas_eval")

    results = []
    all_scores = []

    for i, q in enumerate(queries, 1):
        logger.info(f"  [{i}/{len(queries)}] {retriever_name}: {q['id']}")
        t0 = time.time()

        result = run_single_query(q, retriever, llm, evaluator)
        result["elapsed_seconds"] = round(time.time() - t0, 2)
        results.append(result)

        scores = ragas_mod.RAGASScores(**result["scores"])
        all_scores.append(scores)
        logger.info(
            f"    faithfulness={scores.faithfulness:.2f} "
            f"relevancy={scores.answer_relevancy:.2f} "
            f"precision={scores.context_precision:.2f} "
            f"recall={scores.context_recall:.2f} "
            f"({result['elapsed_seconds']}s)"
        )

    agg = aggregate_scores(all_scores)

    return {
        "retriever": retriever_name,
        "num_queries": len(queries),
        "aggregate_scores": agg,
        "per_query": results,
    }


def print_comparison(flat_results: dict, tree_results: dict):
    """Print a side-by-side comparison table."""
    flat_agg = flat_results["aggregate_scores"]
    tree_agg = tree_results["aggregate_scores"]

    print(f"\n{'='*65}")
    print(f"  RETRIEVER COMPARISON: Flat (CKDRetriever) vs Tree (TreeRetriever)")
    print(f"{'='*65}")
    print(f"  {'Metric':<25} {'Flat':>8} {'Tree':>8} {'Delta':>8}  {'Winner'}")
    print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8}  {'-'*6}")

    for metric in ["faithfulness", "answer_relevancy", "context_precision", "context_recall", "average"]:
        f_val = flat_agg[metric]
        t_val = tree_agg[metric]
        delta = t_val - f_val
        winner = "Tree" if delta > 0.01 else ("Flat" if delta < -0.01 else "Tie")
        marker = "+" if delta > 0 else ""
        print(f"  {metric:<25} {f_val:>8.3f} {t_val:>8.3f} {marker}{delta:>7.3f}  {winner}")

    print(f"{'='*65}")

    categories = set()
    for pq in flat_results["per_query"]:
        categories.add(pq["category"])

    if len(categories) > 1:
        print(f"\n  Per-Category Average Scores:")
        print(f"  {'Category':<15} {'Flat Avg':>10} {'Tree Avg':>10} {'Delta':>8}")
        print(f"  {'-'*15} {'-'*10} {'-'*10} {'-'*8}")

        for cat in sorted(categories):
            flat_cat = [pq for pq in flat_results["per_query"] if pq["category"] == cat]
            tree_cat = [pq for pq in tree_results["per_query"] if pq["category"] == cat]

            flat_avg = sum(
                sum(pq["scores"].values()) / 4 for pq in flat_cat
            ) / len(flat_cat) if flat_cat else 0
            tree_avg = sum(
                sum(pq["scores"].values()) / 4 for pq in tree_cat
            ) / len(tree_cat) if tree_cat else 0

            delta = tree_avg - flat_avg
            marker = "+" if delta > 0 else ""
            print(f"  {cat:<15} {flat_avg:>10.3f} {tree_avg:>10.3f} {marker}{delta:>7.3f}")

        print()


def main():
    parser = argparse.ArgumentParser(description="RAGAS retriever comparison")
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES, help="Path to test queries JSON file")
    parser.add_argument("--retriever", choices=["flat", "tree", "both"], default="both", help="Which retriever(s) to evaluate")
    args = parser.parse_args()

    queries = load_queries(args.queries)
    logger.info(f"Loaded {len(queries)} test queries")

    logger.info("Loading embeddings...")
    embeddings = get_embeddings()

    logger.info("Loading vectorstore...")
    from importlib import import_module
    rag_mod = import_module("simple_rag")
    vectorstore = rag_mod.CKDVectorStore(embeddings)

    stats = vectorstore.get_collection_stats()
    logger.info(f"Vectorstore: {stats['document_count']} documents")
    if stats["document_count"] == 0:
        logger.error("Vectorstore is empty. Run the data pipeline first.")
        sys.exit(1)

    logger.info("Loading LLM...")
    llm = get_llm()

    logger.info("Creating RAGAS evaluator...")
    ragas_mod = import_module("agentic_rag.evaluation.ragas_eval")
    evaluator = ragas_mod.create_evaluator()

    retrievers = {}
    if args.retriever in ("flat", "both"):
        retrievers["flat"] = rag_mod.CKDRetriever(vectorstore=vectorstore)
    if args.retriever in ("tree", "both"):
        retrievers["tree"] = rag_mod.TreeRetriever(
            vectorstore=vectorstore,
            embedding_function=embeddings,
        )

    all_results = {}
    for name, retriever in retrievers.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating: {name} retriever")
        logger.info(f"{'='*60}")
        all_results[name] = run_evaluation(queries, retriever, name, llm, evaluator)

    if len(all_results) == 2:
        print_comparison(all_results["flat"], all_results["tree"])

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = RESULTS_DIR / f"comparison_{timestamp}.json"

    output = {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "num_queries": len(queries),
            "retrievers_evaluated": list(all_results.keys()),
        },
        "results": {
            name: {
                "aggregate_scores": r["aggregate_scores"],
                "per_query": [
                    {
                        "query_id": pq["query_id"],
                        "category": pq["category"],
                        "num_docs_retrieved": pq["num_docs_retrieved"],
                        "scores": pq["scores"],
                        "elapsed_seconds": pq["elapsed_seconds"],
                    }
                    for pq in r["per_query"]
                ],
            }
            for name, r in all_results.items()
        },
    }

    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    logger.info(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
