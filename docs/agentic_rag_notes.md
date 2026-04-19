# Agentic RAG (Level 2) — Architecture & How It Works

## Overview

Level 2 wraps the simple retriever + MedGemma in a **LangGraph state machine**. Instead of always doing retrieval, it first classifies the query and routes it to the most appropriate handler. It also runs PII detection before anything else.

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

## Key Files

| File | What it contains |
|------|-----------------|
| `agentic_rag/graph.py` | `AgenticRAGGraph` — builds and compiles the LangGraph; `.invoke()` / `.stream()` |
| `agentic_rag/nodes.py` | `RAGNodes` — all node functions (pii_check, analyze_query, retrieve_documents, generate_*, evaluate_response) |
| `agentic_rag/pii_handler.py` | `PIIHandler` — Presidio wrapper with NHS/medical custom recognizers |
| `agentic_rag/evaluation/ragas_eval.py` | `RAGASEvaluator` — optional RAGAS scoring on every response |

---

## LangGraph State: `AgenticGraphState`

The state is a `TypedDict` passed between nodes. Each node receives the full state dict and returns a partial update dict (only changed keys).

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

## Node-by-Node Walkthrough

### 1. `pii_check`
- Calls `PIIHandler.anonymize(original_query)`
- Detects: names (`PERSON`), NHS numbers (`UK_NHS`), phone numbers, dates, emails, locations
- Replaces them with `<PERSON_1>`, `<UK_NHS_1>`, etc.
- Stores the reverse map in `pii_map` so PII can be restored in the final response
- **Safety**: if this node raises, LangGraph's `RetryPolicy` retries 3× before the graph stops. The pipeline never proceeds with unredacted text.

### 2. `analyze_query`
Keyword-based routing (no LLM call — fast and deterministic):

| Check | Intent |
|-------|--------|
| `"what do you mean"`, `"can you explain"`, etc. | `CLARIFICATION` |
| `"what is ckd"`, `"what is egfr"`, etc. | `DIRECT` |
| No CKD keywords at all (`kidney`, `egfr`, `dialysis`, …) | `OUT_OF_SCOPE` |
| Anything else | `RETRIEVAL` |

### 3. `retrieve_documents` (RETRIEVAL path only)
- Calls `retriever.invoke(anonymized_query)` — uses whichever retriever was passed in (flat/tree/raptor/contextual)
- Returns `retrieved_documents` + formatted `context` string

### 4. `generate_response` (RETRIEVAL path only)
- Builds prompt: `system_prompt + RAG_PROMPT_TEMPLATE(context, question)`
- Calls `llm.generate(prompt)`
- If PII was detected, restores original values in the response via `pii_handler.restore_in_response()`

### 5. `generate_direct` (DIRECT path)
- Skips retrieval entirely
- Answers definitional questions (`"What is CKD?"`) directly from MedGemma's knowledge

### 6. `generate_clarification` / `generate_out_of_scope`
- Static templated responses — no LLM call, instant
- Out-of-scope lists what topics the system *can* help with

### 7. `evaluate` (after RETRIEVAL path only)
- Optional: only runs if an `evaluator` was passed to `AgenticRAGGraph`
- Calls `RAGASEvaluator.evaluate(query, final_response, contexts)`
- Scores are stored in `evaluation_scores` but do NOT affect the response
- In `main.py` agentic mode, no evaluator is passed → this node is a no-op

---

## PII Handler Details (`agentic_rag/pii_handler.py`)

Custom recognizers on top of Presidio:
- **`NHSNumberRecognizer`**: matches `123 456 7890`, `123-456-7890`, `1234567890`
- **Medical license numbers**: `GMC-1234567`, `NMC-12A3456`

Flow:
1. `analyze()` returns positions and types of detected PII
2. `anonymize()` replaces PII with numbered placeholders
3. `restore_in_response()` substitutes placeholders back with original values in the answer

---

## RetryPolicy Configuration

| Node | max_attempts | initial_interval | max_interval |
|------|-------------|-----------------|-------------|
| `pii_check`, `retrieve_documents` (API nodes) | 3 | 2.0s | 60s |
| `generate_response`, `generate_direct` (LLM nodes) | 3 | 1.0s | 30s |
| `evaluate` | 2 | 1.0s | — |

---

## How to Use

### In `main.py` terminal demo
```bash
uv run python main.py agentic                        # default tree retriever
uv run python main.py agentic --retriever flat        # flat retriever
uv run python main.py agentic --retriever contextual  # contextual retriever
```

**Special input prefix:**
```
stage:3 What foods should I avoid?
```
Sets `ckd_stage=3` on the initial state (passed through but not used for filtering currently).

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
print(result["processing_steps"])  # e.g. ["pii_check", "analyze_query", "retrieve_documents", ...]
```

### Streaming (as used in `main.py`)
```python
for state_update in graph.stream("What is eGFR?"):
    for node_name, node_output in state_update.items():
        print(f"Node: {node_name}")
        # node_output is the partial state update from that node
```

---

## What `main.py` Shows During Agentic Mode

For each query you'll see:
```
  12:34:01 🔧 pii_check         pii_check
  12:34:01 ℹ️  info              ckd_stage=None
  12:34:01 🔧 analyze_query     analyze_query
  12:34:01 ℹ️  info              intent → retrieval
  12:34:01 🔧 retrieve_documents retrieve_documents
  12:34:01 ℹ️  info              retrieved 5 documents
  12:34:01   📄 doc 1           kdigo_2024.pdf
  ...
  12:34:05 🔧 generate_response  generate_response
  12:34:06 🔧 evaluate          evaluate_response