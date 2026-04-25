# Level 2 — Agentic RAG (LangGraph)

Stateful workflow that wraps the simple retriever + MedGemma in a LangGraph
state machine. Adds PII detection, intent-based routing, and optional RAGAS
evaluation. Code lives in `agentic_rag/`.

For the system-wide architecture diagrams see
[`docs/architecture.md`](../architecture.md).

---

## Pipeline at a glance

```
User query
    │
    ▼
[pii_check]          ← Presidio scans for NHS numbers, names, DOBs, etc.
    │
    ▼
[analyze_query]      ← Keyword-based intent classification
    │
    ├── RETRIEVAL ──► [retrieve_documents] ──► [generate_response] ──► [evaluate] ──► END
    ├── DIRECT ──────► [generate_direct] ──────────────────────────────────────────► END
    ├── CLARIFICATION► [generate_clarification] ──────────────────────────────────► END
    └── OUT_OF_SCOPE ► [generate_out_of_scope] ──────────────────────────────────► END
```

---

## Key files

| File | What it contains |
|------|-----------------|
| `agentic_rag/graph.py` | `AgenticRAGGraph` — builds and compiles the LangGraph; `.invoke()` / `.stream()` |
| `agentic_rag/nodes.py` | `RAGNodes` — all node functions (`pii_check`, `analyze_query`, `retrieve_documents`, `generate_*`, `evaluate_response`) |
| `agentic_rag/pii_handler.py` | `PIIHandler` — Presidio wrapper with NHS / medical custom recognizers |
| `agentic_rag/evaluation/ragas_eval.py` | `RAGASEvaluator` — optional RAGAS scoring on every response |
| `agentic_rag/evaluation/custom_metrics.py` | CKD-specific metrics (citation, disclaimer, stage appropriateness, actionability) |
| `agentic_rag/evaluation/langsmith_setup.py` | LangSmith tracing integration |

---

## LangGraph state — `AgenticGraphState`

The state is a `TypedDict` passed between nodes. Each node receives the full
state dict and returns a partial update dict (only changed keys).

```python
{
    # Input
    "original_query": str,
    "ckd_stage": int | None,

    # After pii_check
    "anonymized_query": str,
    "pii_detected": bool,
    "pii_map": dict[str, str],       # placeholder → original value

    # After analyze_query
    "query_intent": str,             # "retrieval" | "direct" | "clarification" | "out_of_scope"
    "query_keywords": list[str],

    # After retrieve_documents
    "retrieved_documents": list[Document],
    "context": str,                  # formatted string for the LLM prompt

    # After generate_*
    "raw_response": str,
    "final_response": str,           # PII restored if needed

    # After evaluate
    "evaluation_scores": dict[str, float],

    # Metadata
    "error": str | None,
    "processing_steps": list[str],   # append-only, tracks which nodes ran
}
```

---

## Node-by-node walkthrough

### 1. `pii_check`
- Calls `PIIHandler.anonymize(original_query)`.
- Detects: names (`PERSON`), NHS numbers (`UK_NHS`), phone numbers, dates, emails, locations.
- Replaces them with `<PERSON_1>`, `<UK_NHS_1>`, etc.
- Stores the reverse map in `pii_map` so PII can be restored in the final response.
- **Safety.** If this node raises, LangGraph's `RetryPolicy` retries 3× before the graph stops. The pipeline never proceeds with unredacted text.

### 2. `analyze_query`
Keyword-based routing (no LLM call — fast and deterministic):

| Check | Intent |
|-------|--------|
| `"what do you mean"`, `"can you explain"`, etc. | `CLARIFICATION` |
| `"what is ckd"`, `"what is egfr"`, etc. | `DIRECT` |
| No CKD keywords (`kidney`, `egfr`, `dialysis`, …) at all | `OUT_OF_SCOPE` |
| Anything else | `RETRIEVAL` |

> **Known misses (see [`docs/TODO.md`](../TODO.md)):**
> `eGFR`-as-definition currently routes to RETRIEVAL; "what should I do
> about my levels?" routes to OUT_OF_SCOPE rather than CLARIFICATION.

### 3. `retrieve_documents` (RETRIEVAL only)
- Calls `retriever.invoke(anonymized_query)` — uses whichever retriever was passed in (flat / tree / raptor / contextual).
- Returns `retrieved_documents` + a formatted `context` string.

### 4. `generate_response` (RETRIEVAL only)
- Builds prompt: `system_prompt + RAG_PROMPT_TEMPLATE(context, question)`.
- Calls `llm.generate(prompt)`.
- If PII was detected, restores original values via `pii_handler.restore_in_response()`.

### 5. `generate_direct` (DIRECT)
- Skips retrieval entirely. Answers definitional questions (`"What is CKD?"`) from MedGemma's parametric knowledge.

### 6. `generate_clarification` / `generate_out_of_scope`
- Static templated responses — no LLM call, instant.
- Out-of-scope lists what topics the system *can* help with.

### 7. `evaluate_response` (RETRIEVAL only)
- Optional: only runs if an `evaluator` was passed to `AgenticRAGGraph`.
- Calls `RAGASEvaluator.evaluate(query, final_response, contexts)`.
- Scores stored in `evaluation_scores` but do not affect the response.
- In `main.py` agentic mode, no evaluator is passed → this node is a no-op.

---

## PII handler details (`agentic_rag/pii_handler.py`)

Custom recognizers on top of Presidio:

- **`NHSNumberRecognizer`**: matches `123 456 7890`, `123-456-7890`, `1234567890`.
- **Medical license numbers**: `GMC-1234567`, `NMC-12A3456`.

Flow:

1. `analyze()` returns positions and types of detected PII.
2. `anonymize()` replaces PII with numbered placeholders.
3. `restore_in_response()` substitutes placeholders back with original values in the answer.

```python
from agentic_rag.pii_handler import PIIHandler

handler = PIIHandler()
result = handler.anonymize("My name is John Smith, NHS: 123-456-7890")
print(result.anonymized_text)
# "My name is <PERSON_1>, NHS: <NHS_1>"
print(result.placeholder_map)
# {"<PERSON_1>": "John Smith", "<NHS_1>": "123-456-7890"}
```

---

## RetryPolicy configuration

| Node | `max_attempts` | `initial_interval` | `max_interval` |
|------|---------------|-------------------|---------------|
| `pii_check`, `retrieve_documents` (API nodes) | 3 | 2.0 s | 60 s |
| `generate_response`, `generate_direct` (LLM nodes) | 3 | 1.0 s | 30 s |
| `evaluate_response` | 2 | 1.0 s | — |

---

## Evaluation framework (in this module)

### `evaluation/ragas_eval.py` — RAGAS metrics

| Metric | Description |
|--------|-------------|
| Faithfulness | Is the answer grounded in context? |
| Answer Relevancy | Does it address the question? |
| Context Precision | Are retrieved docs relevant? |
| Context Recall | Are all relevant docs found? |

```python
from agentic_rag.evaluation.ragas_eval import RAGASEvaluator

evaluator = RAGASEvaluator()
scores = evaluator.evaluate(
    query="What are potassium restrictions?",
    response="Limit potassium to 2000-3000mg daily.",
    contexts=["CKD stage 3 patients should limit potassium..."],
)
print(scores.faithfulness, scores.answer_relevancy)
```

### `evaluation/custom_metrics.py` — CKD-specific metrics

| Metric | Description |
|--------|-------------|
| Citation Score | Are sources properly cited? |
| CKD Stage Appropriateness | Is advice correct for the patient's stage? |
| Disclaimer Present | Medical disclaimer included? |
| Actionability Score | Is advice actionable? |

For the full evaluation guide (run instructions, env-var setup, latest run
results) see [`docs/evaluation.md`](../evaluation.md).

### `evaluation/langsmith_setup.py` — Tracing

LangSmith integration for debugging and analytics.

```python
from agentic_rag.evaluation.langsmith_setup import setup_langsmith

setup_langsmith(api_key="...", project_name="ckd-rag")
```

---

## Usage

### From `main.py` (terminal demo)

```bash
uv run python main.py agentic                         # default tree retriever
uv run python main.py agentic --retriever flat        # flat retriever
uv run python main.py agentic --retriever contextual  # contextual retriever
```

**Special input prefix:**

```
stage:3 What foods should I avoid?
```

Sets `ckd_stage=3` on the initial state (passed through, not used for
metadata filtering currently).

### Programmatically

```python
from agentic_rag import AgenticRAGGraph, PIIHandler
from config import get_llm, get_embeddings
from simple_rag import CKDVectorStore, CKDRetriever

embeddings = get_embeddings()
vectorstore = CKDVectorStore(embeddings)
retriever = CKDRetriever(vectorstore=vectorstore)
llm = get_llm()

graph = AgenticRAGGraph(
    pii_handler=PIIHandler(),
    retriever=retriever,
    llm=llm,
    # evaluator=create_evaluator(),  # optional — adds RAGAS scoring
)

result = graph.invoke("What potassium limits apply to CKD stage 3?")
print(result["final_response"])
print(result["processing_steps"])
```

### Streaming

```python
for state_update in graph.stream("What is eGFR?"):
    for node_name, node_output in state_update.items():
        print(f"Node: {node_name}")
        # node_output is the partial state update from that node
```

---

## What `main.py` shows during agentic mode

```
12:34:01 🔧 pii_check          pii_check
12:34:01 ℹ️  info               ckd_stage=None
12:34:01 🔧 analyze_query      analyze_query
12:34:01 ℹ️  info               intent → retrieval
12:34:01 🔧 retrieve_documents retrieve_documents
12:34:01 ℹ️  info               retrieved 5 documents
12:34:01    📄 doc 1            kdigo_2024.pdf
...
12:34:05 🔧 generate_response  generate_response
12:34:06 🔧 evaluate           evaluate_response
```

---

## Configuration

Environment variables:

- `LANGSMITH_API_KEY` — for tracing
- `LANGSMITH_PROJECT` — project name
- `LANGSMITH_TRACING` — enable/disable

PII entity configuration lives in `config.py`. RAGAS evaluator configuration
is documented in [`docs/evaluation.md`](../evaluation.md).
