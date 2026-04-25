"""
Evaluation script for Agentic RAG and Multi-Agent RAG systems.

Evaluates both systems on:
  1. RAGAS metrics (faithfulness, relevancy, context precision, recall)
  2. CKD-specific metrics (citations, disclaimers, stage appropriateness)
  3. System-level metrics (routing accuracy, PII handling, agent selection)

Usage:
    # Evaluate both systems
    uv run python eval/run_agentic_eval.py

    # Evaluate only agentic RAG
    uv run python eval/run_agentic_eval.py --system agentic

    # Evaluate only multi-agent RAG
    uv run python eval/run_agentic_eval.py --system multi-agent

    # Use custom queries file
    uv run python eval/run_agentic_eval.py --queries eval/test_queries_agentic.json

    # Skip RAGAS (run only CKD + system metrics - no judge LLM needed)
    uv run python eval/run_agentic_eval.py --skip-ragas
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Stream RAGAS judge tokens by default
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
DEFAULT_QUERIES = Path(__file__).parent / "test_queries_agentic.json"


# ─── Helpers ─────────────────────────────────────────────────────────────────

def load_queries(path: Path) -> list[dict]:
    """Load test queries from JSON file."""
    data = json.loads(path.read_text())
    return data["queries"]


def extract_contexts(docs: list) -> list[str]:
    """Extract page_content strings from LangChain Document objects."""
    return [doc.page_content for doc in docs]


def aggregate_scores(scores_dicts: list[dict]) -> dict[str, float]:
    """Compute mean of each metric across a list of score dicts."""
    if not scores_dicts:
        return {}
    keys = scores_dicts[0].keys()
    agg = {}
    for k in keys:
        vals = [s[k] for s in scores_dicts if isinstance(s.get(k), (int, float))]
        agg[k] = sum(vals) / len(vals) if vals else 0.0
    return agg


# ─── Agentic RAG Evaluation ─────────────────────────────────────────────────

def evaluate_agentic_rag(
    queries: list[dict],
    ragas_evaluator,
    ckd_metrics,
    skip_ragas: bool = False,
) -> dict:
    """Run evaluation on the Agentic RAG system (LangGraph pipeline)."""
    from importlib import import_module

    logger.info("Initializing Agentic RAG components...")

    embeddings = get_embeddings()

    rag_mod = import_module("simple_rag")
    vectorstore = rag_mod.CKDVectorStore(embeddings)
    stats = vectorstore.get_collection_stats()
    logger.info(f"Vectorstore: {stats['document_count']} documents")
    if stats["document_count"] == 0:
        logger.error("Vectorstore is empty. Run scripts/ingest.py first.")
        return {"error": "empty_vectorstore"}

    retriever = rag_mod.CKDRetriever(vectorstore=vectorstore)
    llm = get_llm()

    pii_mod = import_module("agentic_rag.pii_handler")
    pii_handler = pii_mod.PIIHandler()

    graph_mod = import_module("agentic_rag.graph")
    graph = graph_mod.AgenticRAGGraph(
        pii_handler=pii_handler,
        retriever=retriever,
        llm=llm,
        evaluator=None,  # We evaluate externally
    )

    results = []
    ragas_scores_list = []
    ckd_scores_list = []

    # Filter to queries relevant for agentic evaluation
    agentic_queries = [q for q in queries if q["category"] != "multi_domain"]

    for i, q_data in enumerate(agentic_queries, 1):
        query_id = q_data["id"]
        query = q_data["query"]
        reference = q_data.get("reference")
        ckd_stage = q_data.get("ckd_stage")
        expected_intent = q_data.get("expected_intent", "RETRIEVAL")
        contains_pii = q_data.get("contains_pii", False)

        logger.info(f"  [{i}/{len(agentic_queries)}] {query_id}: {query[:60]}...")

        t0 = time.time()
        try:
            state = graph.invoke(query=query, ckd_stage=ckd_stage)
        except Exception as e:
            logger.error(f"    Graph invocation failed: {e}")
            results.append({
                "query_id": query_id,
                "error": str(e),
            })
            continue
        elapsed = round(time.time() - t0, 2)

        response = state.get("final_response", "")
        actual_intent = state.get("query_intent", "UNKNOWN")
        pii_detected = state.get("pii_detected", False)
        contexts = extract_contexts(state.get("retrieved_documents", []))
        steps = state.get("processing_steps", [])

        print(f"\n{'─'*65}")
        print(f"Query [{query_id}]: {query}")
        print(f"Intent: expected={expected_intent}, actual={actual_intent}")
        print(f"PII detected: {pii_detected}")
        print(f"Steps: {' → '.join(steps)}")
        print(f"{'─'*65}")
        print(f"Response: {response[:300]}...")
        print(f"{'─'*65}\n")

        # System-level metrics
        # Compare case-insensitively: graph stores intents as lowercase enum values
        # ("retrieval"), test queries spell them uppercase ("RETRIEVAL").
        intent_correct = str(actual_intent).lower() == str(expected_intent).lower()
        pii_correct = (pii_detected == contains_pii) if contains_pii else True

        result = {
            "query_id": query_id,
            "query": query,
            "category": q_data["category"],
            "expected_intent": expected_intent,
            "actual_intent": actual_intent,
            "intent_correct": intent_correct,
            "pii_expected": contains_pii,
            "pii_detected": pii_detected,
            "pii_correct": pii_correct,
            "num_contexts": len(contexts),
            "processing_steps": steps,
            "elapsed_seconds": elapsed,
            "response_length": len(response),
        }

        # RAGAS metrics (only for retrieval queries with contexts)
        if not skip_ragas and ragas_evaluator and contexts and response:
            try:
                ragas_scores = ragas_evaluator.evaluate(
                    query=query,
                    response=response,
                    contexts=contexts,
                    reference=reference,
                )
                result["ragas_scores"] = ragas_scores.to_dict()
                ragas_scores_list.append(ragas_scores.to_dict())
                logger.info(
                    f"    RAGAS: faith={ragas_scores.faithfulness:.2f} "
                    f"rel={ragas_scores.answer_relevancy:.2f} "
                    f"prec={ragas_scores.context_precision:.2f} "
                    f"rec={ragas_scores.context_recall:.2f}"
                )
            except Exception as e:
                logger.warning(f"    RAGAS evaluation failed: {e}")
                result["ragas_scores"] = {"error": str(e)}

        # CKD-specific metrics
        if ckd_metrics and response:
            ckd_scores = ckd_metrics.evaluate(
                query=query,
                response=response,
                ckd_stage=ckd_stage,
                response_time_ms=elapsed * 1000,
            )
            result["ckd_scores"] = ckd_scores.to_dict()
            ckd_scores_list.append(ckd_scores.to_dict())

        results.append(result)

    # Aggregate
    intent_accuracy = (
        sum(1 for r in results if r.get("intent_correct")) / len(results)
        if results else 0.0
    )
    pii_accuracy = (
        sum(1 for r in results if r.get("pii_correct")) / len(results)
        if results else 0.0
    )

    return {
        "system": "agentic_rag",
        "num_queries": len(agentic_queries),
        "aggregate": {
            "intent_accuracy": round(intent_accuracy, 3),
            "pii_accuracy": round(pii_accuracy, 3),
            "ragas": aggregate_scores(ragas_scores_list) if ragas_scores_list else {},
            "ckd_metrics": aggregate_scores(
                [{k: v for k, v in s.items() if isinstance(v, (int, float, bool))}
                 for s in ckd_scores_list]
            ) if ckd_scores_list else {},
        },
        "per_query": results,
    }


# ─── Multi-Agent RAG Evaluation ─────────────────────────────────────────────

def evaluate_multi_agent_rag(
    queries: list[dict],
    ragas_evaluator,
    ckd_metrics,
    skip_ragas: bool = False,
) -> dict:
    """Run evaluation on the Multi-Agent RAG system (orchestrator)."""
    from importlib import import_module

    logger.info("Initializing Multi-Agent RAG components...")

    embeddings = get_embeddings()

    rag_mod = import_module("simple_rag")
    vectorstore = rag_mod.CKDVectorStore(embeddings)
    stats = vectorstore.get_collection_stats()
    if stats["document_count"] == 0:
        logger.error("Vectorstore is empty. Run scripts/ingest.py first.")
        return {"error": "empty_vectorstore"}

    retriever = rag_mod.CKDRetriever(vectorstore=vectorstore)
    llm = get_llm()

    pii_mod = import_module("agentic_rag.pii_handler")
    pii_handler = pii_mod.PIIHandler()

    orch_mod = import_module("multi_agent_rag.orchestrator")
    orchestrator = orch_mod.MultiAgentOrchestrator(
        retriever=retriever,
        llm=llm,
        pii_handler=pii_handler,
    )

    results = []
    ragas_scores_list = []
    ckd_scores_list = []

    # Use all queries (multi-agent handles all types)
    for i, q_data in enumerate(queries, 1):
        query_id = q_data["id"]
        query = q_data["query"]
        reference = q_data.get("reference")
        ckd_stage = q_data.get("ckd_stage")
        expected_agents = q_data.get("expected_agents", [])
        contains_pii = q_data.get("contains_pii", False)

        # Skip out-of-scope and clarification queries for multi-agent
        # (multi-agent doesn't have OOS/clarification routing)
        if q_data["category"] in ("out_of_scope", "clarification"):
            continue

        logger.info(f"  [{i}/{len(queries)}] {query_id}: {query[:60]}...")

        t0 = time.time()
        try:
            response_obj = orchestrator.process(
                query=query,
                ckd_stage=ckd_stage,
                weight_kg=70.0,  # Default weight for protein calculations
            )
        except Exception as e:
            logger.error(f"    Orchestrator failed: {e}")
            results.append({
                "query_id": query_id,
                "error": str(e),
            })
            continue
        elapsed = round(time.time() - t0, 2)

        response = response_obj.answer
        agents_used = response_obj.agents_used
        routing = response_obj.routing_decision
        confidence = response_obj.confidence

        print(f"\n{'─'*65}")
        print(f"Query [{query_id}]: {query}")
        print(f"Routing: {routing.primary_agent.value} (confidence: {routing.confidence:.2f})")
        print(f"Agents used: {agents_used}")
        print(f"Expected agents: {expected_agents}")
        print(f"{'─'*65}")
        print(f"Response: {response[:300]}...")
        print(f"{'─'*65}\n")

        # Agent routing accuracy
        agent_overlap = set(agents_used) & set(expected_agents) if expected_agents else set()
        agent_precision = len(agent_overlap) / len(agents_used) if agents_used else 0.0
        agent_recall = len(agent_overlap) / len(expected_agents) if expected_agents else 1.0

        result = {
            "query_id": query_id,
            "query": query,
            "category": q_data["category"],
            "expected_agents": expected_agents,
            "actual_agents": agents_used,
            "routing_primary": routing.primary_agent.value,
            "routing_confidence": round(routing.confidence, 3),
            "agent_precision": round(agent_precision, 3),
            "agent_recall": round(agent_recall, 3),
            "elapsed_seconds": elapsed,
            "response_length": len(response),
            "system_confidence": round(confidence, 3),
        }

        # Collect contexts from RAG agent if used
        contexts = []
        for agent_name, agent_resp in response_obj.individual_responses.items():
            if hasattr(agent_resp, "contexts"):
                contexts.extend(agent_resp.contexts)
            elif hasattr(agent_resp, "retrieved_contexts"):
                contexts.extend(agent_resp.retrieved_contexts)

        # RAGAS metrics (only when we have contexts)
        if not skip_ragas and ragas_evaluator and contexts and response:
            try:
                ragas_scores = ragas_evaluator.evaluate(
                    query=query,
                    response=response,
                    contexts=contexts,
                    reference=reference,
                )
                result["ragas_scores"] = ragas_scores.to_dict()
                ragas_scores_list.append(ragas_scores.to_dict())
            except Exception as e:
                logger.warning(f"    RAGAS evaluation failed: {e}")
                result["ragas_scores"] = {"error": str(e)}

        # CKD-specific metrics
        if ckd_metrics and response:
            ckd_scores = ckd_metrics.evaluate(
                query=query,
                response=response,
                ckd_stage=ckd_stage,
                response_time_ms=elapsed * 1000,
            )
            result["ckd_scores"] = ckd_scores.to_dict()
            ckd_scores_list.append(ckd_scores.to_dict())

        results.append(result)

    # Aggregate
    valid_results = [r for r in results if "error" not in r]
    avg_agent_precision = (
        sum(r["agent_precision"] for r in valid_results) / len(valid_results)
        if valid_results else 0.0
    )
    avg_agent_recall = (
        sum(r["agent_recall"] for r in valid_results) / len(valid_results)
        if valid_results else 0.0
    )

    return {
        "system": "multi_agent_rag",
        "num_queries": len([q for q in queries if q["category"] not in ("out_of_scope", "clarification")]),
        "aggregate": {
            "agent_routing_precision": round(avg_agent_precision, 3),
            "agent_routing_recall": round(avg_agent_recall, 3),
            "ragas": aggregate_scores(ragas_scores_list) if ragas_scores_list else {},
            "ckd_metrics": aggregate_scores(
                [{k: v for k, v in s.items() if isinstance(v, (int, float, bool))}
                 for s in ckd_scores_list]
            ) if ckd_scores_list else {},
        },
        "per_query": results,
    }


# ─── Comparison & Reporting ──────────────────────────────────────────────────

def print_report(agentic_results: dict | None, multi_results: dict | None):
    """Print a formatted comparison report."""
    print(f"\n{'='*70}")
    print(f"  EVALUATION REPORT — CKD RAG Systems")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"{'='*70}")

    if agentic_results and "error" not in agentic_results:
        agg = agentic_results["aggregate"]
        print(f"\n  AGENTIC RAG ({agentic_results['num_queries']} queries)")
        print(f"  {'─'*50}")
        print(f"  Intent Routing Accuracy:  {agg['intent_accuracy']:.1%}")
        print(f"  PII Detection Accuracy:   {agg['pii_accuracy']:.1%}")

        if agg.get("ragas"):
            print(f"\n  RAGAS Metrics:")
            for metric, val in agg["ragas"].items():
                bar = "█" * int(val * 20) + "░" * (20 - int(val * 20))
                print(f"    {metric:<30} {bar} {val:.3f}")

        if agg.get("ckd_metrics"):
            print(f"\n  CKD Domain Metrics:")
            for metric, val in agg["ckd_metrics"].items():
                if isinstance(val, float):
                    print(f"    {metric:<30} {val:.3f}")

    if multi_results and "error" not in multi_results:
        agg = multi_results["aggregate"]
        print(f"\n  MULTI-AGENT RAG ({multi_results['num_queries']} queries)")
        print(f"  {'─'*50}")
        print(f"  Agent Routing Precision:  {agg['agent_routing_precision']:.1%}")
        print(f"  Agent Routing Recall:     {agg['agent_routing_recall']:.1%}")

        if agg.get("ragas"):
            print(f"\n  RAGAS Metrics:")
            for metric, val in agg["ragas"].items():
                bar = "█" * int(val * 20) + "░" * (20 - int(val * 20))
                print(f"    {metric:<30} {bar} {val:.3f}")

        if agg.get("ckd_metrics"):
            print(f"\n  CKD Domain Metrics:")
            for metric, val in agg["ckd_metrics"].items():
                if isinstance(val, float):
                    print(f"    {metric:<30} {val:.3f}")

    # Side-by-side comparison if both available
    if (agentic_results and "error" not in agentic_results and
            multi_results and "error" not in multi_results):
        a_ragas = agentic_results["aggregate"].get("ragas", {})
        m_ragas = multi_results["aggregate"].get("ragas", {})

        if a_ragas and m_ragas:
            print(f"\n  COMPARISON: Agentic vs Multi-Agent")
            print(f"  {'─'*50}")
            print(f"  {'Metric':<30} {'Agentic':>8} {'Multi':>8} {'Delta':>8}")
            print(f"  {'─'*30} {'─'*8} {'─'*8} {'─'*8}")

            all_metrics = set(list(a_ragas.keys()) + list(m_ragas.keys()))
            for metric in sorted(all_metrics):
                a_val = a_ragas.get(metric, 0.0)
                m_val = m_ragas.get(metric, 0.0)
                delta = m_val - a_val
                sign = "+" if delta > 0 else ""
                print(f"  {metric:<30} {a_val:>8.3f} {m_val:>8.3f} {sign}{delta:>7.3f}")

    print(f"\n{'='*70}\n")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Agentic RAG and Multi-Agent RAG systems"
    )
    parser.add_argument(
        "--queries", type=Path, default=DEFAULT_QUERIES,
        help="Path to test queries JSON file",
    )
    parser.add_argument(
        "--system", choices=["agentic", "multi-agent", "both"], default="both",
        help="Which system(s) to evaluate",
    )
    parser.add_argument(
        "--skip-ragas", action="store_true",
        help="Skip RAGAS evaluation (no judge LLM needed)",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output JSON file path (default: auto-generated in eval/results/)",
    )
    args = parser.parse_args()

    queries = load_queries(args.queries)
    logger.info(f"Loaded {len(queries)} test queries from {args.queries}")

    # Initialize evaluators
    ragas_evaluator = None
    if not args.skip_ragas:
        try:
            from agentic_rag.evaluation.ragas_eval import create_evaluator
            ragas_evaluator = create_evaluator()
            logger.info("RAGAS evaluator initialized")
        except Exception as e:
            logger.warning(f"RAGAS evaluator init failed: {e}. Continuing without RAGAS.")

    from agentic_rag.evaluation.custom_metrics import create_ckd_metrics
    ckd_metrics = create_ckd_metrics()

    # Run evaluations
    agentic_results = None
    multi_results = None

    if args.system in ("agentic", "both"):
        logger.info("\n" + "=" * 60)
        logger.info("EVALUATING: Agentic RAG")
        logger.info("=" * 60)
        agentic_results = evaluate_agentic_rag(
            queries, ragas_evaluator, ckd_metrics, args.skip_ragas
        )

    if args.system in ("multi-agent", "both"):
        logger.info("\n" + "=" * 60)
        logger.info("EVALUATING: Multi-Agent RAG")
        logger.info("=" * 60)
        multi_results = evaluate_multi_agent_rag(
            queries, ragas_evaluator, ckd_metrics, args.skip_ragas
        )

    # Print report
    print_report(agentic_results, multi_results)

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = args.output or RESULTS_DIR / f"agentic_eval_{timestamp}.json"

    output = {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "num_queries": len(queries),
            "systems_evaluated": [
                s for s, r in [("agentic_rag", agentic_results), ("multi_agent_rag", multi_results)]
                if r is not None
            ],
            "ragas_enabled": not args.skip_ragas,
        },
    }
    if agentic_results:
        output["agentic_rag"] = agentic_results
    if multi_results:
        output["multi_agent_rag"] = multi_results

    output_path.write_text(json.dumps(output, indent=2, default=str, ensure_ascii=False))
    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
