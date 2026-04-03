# RAGAS Retriever Comparison: Flat vs Tree-Based

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run RAGAS evaluation on the Simple RAG pipeline with both CKDRetriever (flat) and TreeRetriever (tree-based), compare results side-by-side, and produce a JSON report.

**Architecture:** A standalone evaluation script (`eval/run_retriever_comparison.py`) loads the vectorstore once, creates both retrievers, runs each query through a shared LLM, collects retrieval contexts and generated answers, then batch-evaluates via RAGAS. A small CKD-specific test dataset (`eval/test_queries.json`) provides the queries and optional ground truth references. Results are saved to `eval/results/` as JSON with per-query and aggregate scores.

**Tech Stack:** RAGAS 0.4.x, LangChain, ChromaDB, OpenRouter/Gemini judge LLM, pytest

---

## File Structure

| File | Responsibility |
|------|---------------|
| `eval/__init__.py` | Package marker |
| `eval/test_queries.json` | 10 CKD test queries with optional ground truth |
| `eval/run_retriever_comparison.py` | Main evaluation script — runs both retrievers, generates answers, calls RAGAS, saves results |
| `eval/results/` | Output directory for JSON result files (gitignored) |
| `tests/test_eval_comparison.py` | Unit tests for evaluation helpers (no LLM needed) |
| `.gitignore` | Add `eval/results/` entry |

---

### Task 1: Create test query dataset

**Files:**
- Create: `eval/__init__.py`
- Create: `eval/test_queries.json`

- [ ] **Step 1: Create the eval package**

```python
# eval/__init__.py
```

(Empty file — just a package marker.)

- [ ] **Step 2: Write the test queries JSON**

Create `eval/test_queries.json` with 10 CKD queries spanning different topics. Each query has an optional `reference` field (ground truth answer) for context_recall scoring.

```json
{
  "description": "CKD test queries for RAGAS retriever comparison evaluation",
  "queries": [
    {
      "id": "diet_potassium",
      "query": "What are the potassium restrictions for CKD stage 3 patients?",
      "reference": "CKD stage 3 patients should limit potassium intake to 2000-3000mg per day. High potassium foods to avoid include bananas, oranges, potatoes, and tomatoes.",
      "category": "diet"
    },
    {
      "id": "diet_phosphorus",
      "query": "What foods are high in phosphorus and should be avoided in CKD?",
      "reference": "High phosphorus foods to avoid include dairy products, nuts, seeds, cola drinks, processed meats, and chocolate. Phosphorus intake should be limited to 800-1000mg per day in CKD stages 3-5.",
      "category": "diet"
    },
    {
      "id": "diet_sodium",
      "query": "How much sodium should a CKD patient consume daily?",
      "reference": "CKD patients in stages 3-5 should limit sodium intake to less than 2000mg per day. This involves avoiding processed foods, canned soups, and adding salt to meals.",
      "category": "diet"
    },
    {
      "id": "medication_nsaids",
      "query": "Why should CKD patients avoid NSAIDs like ibuprofen?",
      "reference": "NSAIDs such as ibuprofen and naproxen are nephrotoxic and can worsen kidney function. They reduce blood flow to the kidneys and should be avoided in CKD patients. Paracetamol is a safer alternative for pain relief.",
      "category": "medication"
    },
    {
      "id": "medication_ace",
      "query": "When are ACE inhibitors recommended for kidney disease?",
      "reference": "ACE inhibitors and ARBs are recommended as first-line treatment for CKD patients with proteinuria. They reduce proteinuria and slow progression of kidney disease.",
      "category": "medication"
    },
    {
      "id": "lifestyle_exercise",
      "query": "What exercise recommendations exist for CKD patients?",
      "reference": "CKD patients are encouraged to engage in regular moderate-intensity exercise such as walking, cycling, or swimming for at least 150 minutes per week. Exercise improves cardiovascular health and quality of life.",
      "category": "lifestyle"
    },
    {
      "id": "stages_egfr",
      "query": "What eGFR level indicates CKD stage 4?",
      "reference": "CKD stage 4 is defined by an eGFR of 15-29 mL/min/1.73m2, indicating severely reduced kidney function. At this stage, preparation for renal replacement therapy should begin.",
      "category": "stages"
    },
    {
      "id": "dialysis_preparation",
      "query": "When should dialysis preparation begin for CKD patients?",
      "reference": "Dialysis preparation should begin in CKD stage 4 (eGFR 15-29) and typically includes creating vascular access, patient education about dialysis modalities, and referral to a nephrologist.",
      "category": "dialysis"
    },
    {
      "id": "anemia_esa",
      "query": "How are erythropoiesis-stimulating agents used in CKD anemia?",
      "reference": null,
      "category": "anemia"
    },
    {
      "id": "protein_intake",
      "query": "What is the recommended protein intake for pre-dialysis CKD patients?",
      "reference": "Pre-dialysis CKD patients in stages 3-5 should consume 0.6-0.8 g protein per kg body weight per day. Protein intake may increase once dialysis begins.",
      "category": "diet"
    }
  ]
}
```

- [ ] **Step 3: Commit**

```bash
git add eval/__init__.py eval/test_queries.json
git commit -m "feat(eval): add CKD test query dataset for retriever comparison"
```

---

### Task 2: Add eval/results/ to .gitignore

**Files:**
- Modify: `.gitignore`

- [ ] **Step 1: Append eval results directory to .gitignore**

Add to the end of `.gitignore`:

```
# Evaluation results (generated, not committed)
eval/results/
```

- [ ] **Step 2: Commit**

```bash
git add .gitignore
git commit -m "chore: gitignore eval results directory"
```

---

### Task 3: Write unit tests for evaluation helpers

**Files:**
- Create: `tests/test_eval_comparison.py`

These tests verify the query loading and result aggregation logic — no LLM or vectorstore needed.

- [ ] **Step 1: Write the test file**

```python
"""Tests for retriever comparison evaluation helpers."""

import json
import pytest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestLoadQueries:
    """Test query dataset loading."""

    def test_load_queries_file_exists(self):
        path = Path(__file__).parent.parent / "eval" / "test_queries.json"
        assert path.exists(), f"test_queries.json not found at {path}"

    def test_load_queries_valid_json(self):
        path = Path(__file__).parent.parent / "eval" / "test_queries.json"
        data = json.loads(path.read_text())
        assert "queries" in data
        assert len(data["queries"]) >= 5

    def test_query_schema(self):
        path = Path(__file__).parent.parent / "eval" / "test_queries.json"
        data = json.loads(path.read_text())
        for q in data["queries"]:
            assert "id" in q, f"Query missing 'id': {q}"
            assert "query" in q, f"Query missing 'query': {q}"
            assert "category" in q, f"Query missing 'category': {q}"
            # reference is optional (can be null)
            assert "reference" in q, f"Query missing 'reference' key: {q}"


class TestAggregateScores:
    """Test score aggregation logic."""

    def test_aggregate_empty(self):
        from eval.run_retriever_comparison import aggregate_scores
        result = aggregate_scores([])
        assert result == {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_precision": 0.0,
            "context_recall": 0.0,
            "average": 0.0,
        }

    def test_aggregate_single(self):
        from eval.run_retriever_comparison import aggregate_scores
        from importlib import import_module
        ragas_mod = import_module("agentic_rag.evaluation.ragas_eval")
        scores = [ragas_mod.RAGASScores(0.8, 0.9, 0.7, 0.6)]
        result = aggregate_scores(scores)
        assert result["faithfulness"] == pytest.approx(0.8)
        assert result["answer_relevancy"] == pytest.approx(0.9)
        assert result["context_precision"] == pytest.approx(0.7)
        assert result["context_recall"] == pytest.approx(0.6)
        assert result["average"] == pytest.approx(0.75)

    def test_aggregate_multiple(self):
        from eval.run_retriever_comparison import aggregate_scores
        from importlib import import_module
        ragas_mod = import_module("agentic_rag.evaluation.ragas_eval")
        scores = [
            ragas_mod.RAGASScores(1.0, 1.0, 1.0, 1.0),
            ragas_mod.RAGASScores(0.0, 0.0, 0.0, 0.0),
        ]
        result = aggregate_scores(scores)
        assert result["faithfulness"] == pytest.approx(0.5)
        assert result["average"] == pytest.approx(0.5)


class TestFormatContext:
    """Test context formatting from retrieved documents."""

    def test_format_contexts_extracts_page_content(self):
        from eval.run_retriever_comparison import extract_contexts
        from unittest.mock import MagicMock

        doc1 = MagicMock()
        doc1.page_content = "Context about potassium limits."
        doc2 = MagicMock()
        doc2.page_content = "Context about sodium intake."

        result = extract_contexts([doc1, doc2])
        assert result == [
            "Context about potassium limits.",
            "Context about sodium intake.",
        ]

    def test_format_contexts_empty(self):
        from eval.run_retriever_comparison import extract_contexts
        assert extract_contexts([]) == []
```

- [ ] **Step 2: Run the tests (expect partial failure — module not yet created)**

```bash
uv run pytest tests/test_eval_comparison.py -v
```

Expected: `TestLoadQueries` tests pass (queries file exists). `TestAggregateScores` and `TestFormatContext` fail with `ModuleNotFoundError` because `eval.run_retriever_comparison` doesn't exist yet.

- [ ] **Step 3: Commit the test file**

```bash
git add tests/test_eval_comparison.py
git commit -m "test(eval): add unit tests for retriever comparison helpers"
```

---

### Task 4: Implement the comparison script

**Files:**
- Create: `eval/run_retriever_comparison.py`

- [ ] **Step 1: Write the comparison script**

```python
"""
Run RAGAS evaluation comparing flat (CKDRetriever) vs tree-based (TreeRetriever).

Usage:
    uv run python eval/run_retriever_comparison.py
    uv run python eval/run_retriever_comparison.py --queries eval/test_queries.json
    uv run python eval/run_retriever_comparison.py --retriever flat   # only flat
    uv run python eval/run_retriever_comparison.py --retriever tree   # only tree
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

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


# ---------------------------------------------------------------------------
# Helpers (tested in tests/test_eval_comparison.py)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def run_single_query(
    query_data: dict,
    retriever,
    llm,
    evaluator,
) -> dict:
    """Run retrieval + generation + evaluation for one query.

    Args:
        query_data: Dict with 'id', 'query', 'reference', 'category'.
        retriever: LangChain BaseRetriever instance.
        llm: LLM with generate() method.
        evaluator: RAGASEvaluator instance.

    Returns:
        Dict with query info, contexts, answer, and RAGAS scores.
    """
    query = query_data["query"]
    reference = query_data.get("reference")

    # Retrieve
    docs = retriever.invoke(query)
    contexts = extract_contexts(docs)

    # Generate answer using the LLM with context
    from config import RAG_PROMPT_TEMPLATE, build_system_prompt, load_patient_context

    system_prompt = build_system_prompt(load_patient_context())
    context_str = "\n\n---\n\n".join(
        f"[{i+1}] {doc.metadata.get('source', 'Unknown')}\n{doc.page_content}"
        for i, doc in enumerate(docs)
    )
    prompt = f"{system_prompt}\n\n{RAG_PROMPT_TEMPLATE.format(context=context_str, question=query)}"
    answer = llm.generate(prompt)

    # Evaluate
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


def run_evaluation(
    queries: list[dict],
    retriever,
    retriever_name: str,
    llm,
    evaluator,
) -> dict:
    """Run full evaluation for one retriever across all queries.

    Args:
        queries: List of query dicts.
        retriever: Retriever instance.
        retriever_name: 'flat' or 'tree'.
        llm: LLM instance.
        evaluator: RAGASEvaluator instance.

    Returns:
        Dict with per-query results and aggregate scores.
    """
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="RAGAS retriever comparison")
    parser.add_argument(
        "--queries",
        type=Path,
        default=DEFAULT_QUERIES,
        help="Path to test queries JSON file",
    )
    parser.add_argument(
        "--retriever",
        choices=["flat", "tree", "both"],
        default="both",
        help="Which retriever(s) to evaluate",
    )
    args = parser.parse_args()

    # Load queries
    queries = load_queries(args.queries)
    logger.info(f"Loaded {len(queries)} test queries")

    # Initialize shared components
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

    # Build retrievers
    retrievers = {}
    if args.retriever in ("flat", "both"):
        retrievers["flat"] = rag_mod.CKDRetriever(vectorstore=vectorstore)
    if args.retriever in ("tree", "both"):
        retrievers["tree"] = rag_mod.TreeRetriever(
            vectorstore=vectorstore,
            embedding_function=embeddings,
        )

    # Run evaluation
    all_results = {}
    for name, retriever in retrievers.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating: {name} retriever")
        logger.info(f"{'='*60}")
        all_results[name] = run_evaluation(queries, retriever, name, llm, evaluator)

    # Comparison summary
    if len(all_results) == 2:
        print_comparison(all_results["flat"], all_results["tree"])

    # Save results
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

    # Per-category breakdown
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


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the unit tests to verify helpers pass**

```bash
uv run pytest tests/test_eval_comparison.py -v
```

Expected: All 8 tests pass.

- [ ] **Step 3: Commit**

```bash
git add eval/run_retriever_comparison.py
git commit -m "feat(eval): add RAGAS retriever comparison script (flat vs tree)"
```

---

### Task 5: Smoke test the full pipeline (dry run)

This task verifies the script loads and parses args without actually running evaluation (which requires GPU + API key).

**Files:**
- None created; validation only.

- [ ] **Step 1: Verify script parses arguments**

```bash
uv run python eval/run_retriever_comparison.py --help
```

Expected output shows usage with `--queries` and `--retriever` options.

- [ ] **Step 2: Verify query loading works**

```bash
uv run python -c "
from eval.run_retriever_comparison import load_queries
from pathlib import Path
qs = load_queries(Path('eval/test_queries.json'))
print(f'Loaded {len(qs)} queries')
for q in qs:
    ref_status = 'has reference' if q.get('reference') else 'no reference'
    print(f'  {q[\"id\"]}: {q[\"category\"]} ({ref_status})')
"
```

Expected: Lists 10 queries with their categories.

- [ ] **Step 3: Run full test suite to ensure nothing broke**

```bash
uv run pytest -v
```

Expected: All existing tests still pass + new tests pass.

- [ ] **Step 4: Commit (if any fixups were needed)**

```bash
git add -A
git commit -m "fix(eval): fixups from smoke testing"
```

---

### Task 6: Run the actual evaluation (on GPU machine)

This task runs on the target machine with GPU, vectorstore data, and `RAGAS_JUDGE_API_KEY` configured.

**Files:**
- No new files; execution only.

- [ ] **Step 1: Verify environment variables are set**

```bash
echo "RAGAS_JUDGE_API_KEY: ${RAGAS_JUDGE_API_KEY:+set}"
echo "USE_REMOTE_LLM: ${USE_REMOTE_LLM:-false}"
echo "MODEL_SERVER_URL: ${MODEL_SERVER_URL:-not set}"
```

All should show valid values. `RAGAS_JUDGE_API_KEY` must be set.

- [ ] **Step 2: Run both retrievers**

```bash
uv run python eval/run_retriever_comparison.py --retriever both 2>&1 | tee eval/results/run_log.txt
```

Expected: Runs 10 queries x 2 retrievers = 20 evaluations. Prints comparison table at the end. Saves JSON to `eval/results/comparison_YYYYMMDD_HHMMSS.json`.

- [ ] **Step 3: Run flat only (if debugging)**

```bash
uv run python eval/run_retriever_comparison.py --retriever flat
```

- [ ] **Step 4: Run tree only (if debugging)**

```bash
uv run python eval/run_retriever_comparison.py --retriever tree
```

- [ ] **Step 5: Review results**

Open the JSON file in `eval/results/` and verify:
- Each query has scores between 0 and 1
- Aggregate scores are computed correctly
- The comparison table shows meaningful differences

---

## Notes for the engineer

### Prerequisites on target machine
1. ChromaDB vectorstore must be populated (run `uv run python main.py simple` once, or run the data pipeline)
2. `RAGAS_JUDGE_API_KEY` must be set (OpenRouter free tier works: sign up at https://openrouter.ai)
3. LLM must be available (local MedGemma or remote vLLM via `USE_REMOTE_LLM=true`)

### Expected runtime
- ~2-5 minutes per query (depends on LLM speed and RAGAS judge API latency)
- Full run (10 queries x 2 retrievers): ~40-100 minutes
- Run with `--retriever flat` or `--retriever tree` to test one at a time

### Interpreting results
- **Faithfulness**: Higher = answer is more grounded in retrieved context. Tree should win if it retrieves more relevant sections.
- **Context Precision**: Higher = retrieved docs are more relevant. Tree should win by routing to correct sections.
- **Context Recall**: Higher = all needed info was retrieved. Requires ground truth references. Tree may lose if it's too narrow.
- **Answer Relevancy**: Higher = answer directly addresses the question. Should be similar for both if LLM is the same.
