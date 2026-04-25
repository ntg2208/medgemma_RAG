# Evaluation Guide

Covers everything about evaluating the RAG systems in this project:

- [Concepts](#concepts) — what RAGAS measures and what the CKD-specific metrics mean.
- [Quick Start](#quick-start) — minimal commands to get a result.
- [Environment Setup](#environment-setup) — judge LLM, embeddings, vLLM.
- [What Gets Evaluated](#what-gets-evaluated) — Agentic & Multi-Agent paths.
- [CLI Usage](#cli-usage) — `eval/run_agentic_eval.py` and friends.
- [Output Format](#output-format) — JSON schema.
- [Latest Run Summary](#latest-run-summary-2026-04-24-ship-snapshot) — actual numbers.
- [Interpreting Results](#interpreting-results) — thresholds and failure patterns.
- [Extending the Evaluation](#extending-the-evaluation) — new queries / metrics.

---

## Concepts

### What is RAGAS?

**RAGAS** (Retrieval Augmented Generation Assessment) is a framework for
evaluating RAG pipelines without human-labelled QA pairs. It uses an
**LLM-as-judge** (a cheap external model like Gemini Flash) to score
pipeline outputs.

Key idea: you don't need a perfect golden dataset. RAGAS infers quality by
comparing:

- what the LLM *said* vs what the context *supports* → **faithfulness**
- what the LLM *said* vs what the *question asked* → **answer relevancy**
- whether the *retrieved docs* were actually relevant → **context precision**
- whether the retrieved docs *covered all necessary info* → **context recall**

### The four core RAGAS metrics

#### 1. Faithfulness (0–1, higher is better)
> "Did the answer only say things that are supported by the retrieved context?"

- Breaks the answer into atomic claims, then checks each claim against context.
- **Low score = hallucination** — model invented facts not in retrieved chunks.
- Does **not** need a reference answer.
- Most important metric for a medical RAG: we never want unsupported claims.

#### 2. Answer Relevancy (0–1, higher is better)
> "Did the answer actually address the question that was asked?"

- Generates N paraphrased questions from the answer, embeds them, measures cosine similarity to the original question.
- Low score → off-topic or too generic.
- Does not need a reference answer. Uses embeddings (configured via `RAGAS_EMBEDDINGS_MODEL`).

#### 3. Context Precision (0–1, higher is better)
> "Were the retrieved chunks actually useful?"

- Asks the judge LLM whether each retrieved chunk was needed to answer the question.
- We use `LLMContextPrecisionWithoutReference` (no ground truth needed).
- Low score → retriever returning noise.
- Directly measures retriever quality independent of the answer.

#### 4. Context Recall (0–1, higher is better)
> "Did the retriever find all the information needed?"

- Compares the answer to a **reference (ground-truth)** answer.
- Checks whether every claim in the reference is attributable to retrieved context.
- **Requires a reference answer** — queries without `reference` get score 0.0.
- Low score → retriever missing important chunks.

### How they fit together

```
Query ──► Retriever ──► Context chunks ──► LLM ──► Answer
                │                                      │
                └── Context Precision                  │
                └── Context Recall (needs ref)         │
                                                       └── Faithfulness
                                                       └── Answer Relevancy
```

| Metric | Tells you | Needs reference? |
|--------|-----------|------------------|
| Faithfulness | LLM hallucination risk | No |
| Answer Relevancy | Answer quality / focus | No |
| Context Precision | Retriever noise | No |
| Context Recall | Retriever coverage | **Yes** |

### CKD-specific metrics (our own)

Defined in `agentic_rag/evaluation/custom_metrics.py`:

| Metric | Description |
|--------|-------------|
| Citation score | Regex check for source mentions (NICE, KDIGO, UKKA, …) |
| CKD stage appropriateness | Stage-specific keyword presence |
| Disclaimer present | Medical safety phrase included |
| Actionability | Concrete instructions vs vague advice |
| Medical accuracy indicators | NSAID avoidance, eGFR mentions, safe dosing |

### System-level metrics

Computed directly from pipeline state, not via judge LLM:

| Metric | Where it comes from |
|--------|---------------------|
| Intent routing accuracy | `actual_intent` vs `expected_intent` (case-insensitive after the 2026-04-25 fix) |
| PII detection accuracy | `pii_detected` vs `contains_pii` from query metadata |
| Agent routing precision/recall | Multi-agent: overlap of `actual_agents` and `expected_agents` |

### Cost note

Judge calls cost roughly **$0.001–0.005 per query**. A full eval (20 queries
× 2 systems × 4 RAGAS metrics) lands around **$0.10–0.40** on
`gemini-2.0-flash-001` via OpenRouter.

---

## Quick Start

```bash
# 1. Ensure the vectorstore is populated
uv run python scripts/ingest.py

# 2. Ensure vLLM server is running (or USE_REMOTE_MODELS=false for local)
#    vLLM should be serving MedGemma at MODEL_SERVER_URL

# 3. Run evaluation (both systems)
uv run python eval/run_agentic_eval.py

# 4. Results are saved to eval/results/agentic_eval_<timestamp>.json
```

For a one-off retriever-only comparison (flat vs tree, RAGAS only, 10
queries):

```bash
uv run python eval/run_retriever_comparison.py --retriever both
```

---

## Environment Setup

The following environment variables must be set (in `.env` or shell):

| Variable | Purpose | Example |
|----------|---------|---------|
| `GOOGLE_API_KEY` | Google AI Studio key (used as fallback for embeddings) | `AIzaSy...` |
| `RAGAS_JUDGE_MODEL` | Judge LLM for RAGAS metrics | `google/gemini-2.0-flash-001` |
| `RAGAS_JUDGE_API_KEY` | API key for judge LLM provider | OpenRouter or Google key |
| `RAGAS_JUDGE_BASE_URL` | OpenAI-compatible judge endpoint | `https://openrouter.ai/api/v1` |
| `RAGAS_JUDGE_TIMEOUT` | Per-call timeout in seconds (default 600) | `600` |
| `RAGAS_JUDGE_MAX_RETRIES` | Per-call retries (default 3) | `3` |
| `RAGAS_EMBEDDINGS_MODEL` | Embeddings for the answer-relevancy metric | `gemini-embedding-2` |
| `RAGAS_EMBEDDINGS_BASE_URL` | OpenAI-compatible embeddings endpoint | `https://generativelanguage.googleapis.com/v1beta/openai/` |
| `RAGAS_EMBEDDINGS_API_KEY` | API key for embeddings provider (falls back to `GOOGLE_API_KEY`) | `AIzaSy...` |
| `USE_REMOTE_MODELS` | Use remote vLLM server for MedGemma | `true` |
| `MODEL_SERVER_URL` | vLLM server address | `http://localhost:8000` |

### Why judge and embeddings are configured separately

The default judge provider (OpenRouter) does **not** serve embeddings, but the
RAGAS `answer_relevancy` metric needs one. The two are split:

```bash
# Judge → OpenRouter (cheap, supports gemini-2.0-flash)
export RAGAS_JUDGE_MODEL=google/gemini-2.0-flash-001
export RAGAS_JUDGE_API_KEY=sk-or-...
export RAGAS_JUDGE_BASE_URL=https://openrouter.ai/api/v1

# Embeddings → Google AI Studio (free tier covers eval workloads)
export RAGAS_EMBEDDINGS_MODEL=gemini-embedding-2
export RAGAS_EMBEDDINGS_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
export RAGAS_EMBEDDINGS_API_KEY=AIzaSy...   # or set GOOGLE_API_KEY
```

If you prefer Google for both, point `RAGAS_JUDGE_BASE_URL` at the same
Google endpoint and reuse the key.

---

## What Gets Evaluated

### 1. Agentic RAG

LangGraph state machine — see [`docs/modules/agentic-rag.md`](modules/agentic-rag.md):

```
START → PII Check → Query Analysis → [Router]
                                        ├─ RETRIEVAL → Retrieve → Generate → Evaluate → END
                                        ├─ DIRECT → Generate Direct → END
                                        ├─ CLARIFICATION → Ask for Details → END
                                        └─ OUT_OF_SCOPE → Decline → END
```

**Metrics evaluated:** all RAGAS, all CKD, plus intent-routing accuracy and
PII-detection accuracy.

### 2. Multi-Agent RAG

Orchestrator routes to Diet / Medication / Lifestyle / RAG agents — see
[`docs/modules/multi-agent-rag.md`](modules/multi-agent-rag.md):

```
Query → Orchestrator → [Keyword Scoring]
                          ├─ Diet Agent
                          ├─ Medication Agent
                          ├─ Lifestyle Agent
                          ├─ RAG Agent
                          └─ MULTI (2+ agents for cross-domain queries)
```

**Metrics evaluated:** all CKD, plus agent-routing precision/recall. RAGAS is
**not yet wired** for multi-agent — the orchestrator does not surface
per-agent contexts. See [`docs/TODO.md`](TODO.md).

---

## Test Queries

Test queries are defined in `eval/test_queries_agentic.json`. Each query
includes:

```json
{
  "id": "agentic_retrieval_01",
  "query": "What are the potassium restrictions for CKD stage 3 patients?",
  "reference": "CKD stage 3 patients should limit potassium...",
  "category": "diet",
  "expected_intent": "RETRIEVAL",
  "ckd_stage": 3,
  "expected_agents": ["diet"],
  "contains_pii": false
}
```

**Categories covered:**

| Category | Count | Tests |
|----------|-------|-------|
| Diet | 4 | Potassium, sodium, phosphorus, protein restrictions by stage |
| Medication | 3 | NSAID avoidance, ACE inhibitors, metformin safety |
| Lifestyle | 2 | Exercise, blood pressure management |
| Stages | 1 | eGFR level interpretation |
| Dialysis | 1 | Dialysis preparation timing |
| General | 2 | CKD definition, eGFR definition (DIRECT intent) |
| Out of Scope | 2 | Non-medical queries (OOS intent) |
| Clarification | 1 | Vague query needing more context |
| Multi-domain | 3 | Cross-agent queries (diet+exercise, medication+diet, lifestyle+diet+RAG) |
| PII Handling | 1 | Query containing name and NHS number |

---

## CLI Usage

```bash
# Evaluate both systems
uv run python eval/run_agentic_eval.py

# Evaluate only agentic RAG
uv run python eval/run_agentic_eval.py --system agentic

# Evaluate only multi-agent RAG
uv run python eval/run_agentic_eval.py --system multi-agent

# Skip RAGAS (fast — no judge LLM needed)
uv run python eval/run_agentic_eval.py --skip-ragas

# Custom queries file
uv run python eval/run_agentic_eval.py --queries eval/test_queries_agentic.json

# Custom output path
uv run python eval/run_agentic_eval.py --output eval/results/my_run.json
```

---

## Output Format

Results are saved as JSON in `eval/results/` (gitignored). Schema:

```json
{
  "metadata": {
    "timestamp": "<ISO-8601 UTC>",
    "num_queries": 20,
    "systems_evaluated": ["agentic_rag", "multi_agent_rag"],
    "ragas_enabled": true
  },
  "agentic_rag": {
    "system": "agentic_rag",
    "num_queries": 17,
    "aggregate": {
      "intent_accuracy": 0.0,
      "pii_accuracy": 0.0,
      "ragas":       { "faithfulness": 0.0, "answer_relevancy": 0.0, "context_precision": 0.0, "context_recall": 0.0 },
      "ckd_metrics": { "citation_score": 0.0, "ckd_stage_appropriateness": 0.0, "disclaimer_present": 0.0, "actionability_score": 0.0, "response_time_ms": 0.0 }
    },
    "per_query": [...]
  },
  "multi_agent_rag": {
    "system": "multi_agent_rag",
    "num_queries": 17,
    "aggregate": {
      "agent_routing_precision": 0.0,
      "agent_routing_recall": 0.0,
      "ragas":       {},
      "ckd_metrics": {...}
    },
    "per_query": [...]
  }
}
```

---

## Latest Run Summary (2026-04-24, ship snapshot)

File: `eval/results/agentic_eval_20260424_204001.json` (20 queries → 17 in
scope per system after filtering).

### Run parameters (baseline configuration as-run)

| Component | Value |
|-----------|-------|
| Generator LLM | MedGemma 1.5 4B (`google/medgemma-1.5-4b-it`) on EC2 vLLM |
| Generation `temperature` | 0.7 (lowered to 0.3 in `scripts/startup.sh` after this run; not yet re-evaluated) |
| Embeddings | EmbeddingGemma 300M, dim 768 |
| Retriever (Agentic RAG) | `CKDRetriever` (flat semantic + medical-term expansion), `TOP_K_RESULTS=5`, `SIMILARITY_THRESHOLD=0.3` |
| Retriever (Multi-Agent → RAG agent) | Same `CKDRetriever` |
| Prompts | Baseline `RAG_SYSTEM_PROMPT` + `RAG_PROMPT_TEMPLATE` from `config.py` (no per-agent prompt tuning) |
| Architecture | Default LangGraph pipeline (`agentic_rag/graph.py`) + default `MultiAgentOrchestrator` (no overrides) |
| Judge LLM | `google/gemini-2.0-flash-001` via OpenRouter |
| Judge embeddings | `gemini-embedding-2` via Google AI Studio |
| Test queries | `eval/test_queries_agentic.json` (20 total) |

> **Note.** The eval script currently hard-wires `CKDRetriever` (flat) for
> both systems (`eval/run_agentic_eval.py:101`, `:262`). Switch to
> `TreeRetriever` (or another retriever via the factory in
> `simple_rag/retriever.py`) before the next run if you want to publish
> numbers under a different retrieval strategy. See
> [`docs/TODO.md`](TODO.md).

### Aggregate metrics

| Metric | Agentic RAG | Multi-Agent RAG |
|--------|------------:|----------------:|
| Intent routing accuracy | 0.882 (15/17) | — |
| PII detection accuracy | 1.000 | — |
| Agent routing precision | — | 0.598 |
| Agent routing recall | — | 0.794 |
| RAGAS faithfulness | 0.448 | n/a* |
| RAGAS answer relevancy | 0.579 | n/a* |
| RAGAS context precision | 0.680 | n/a* |
| RAGAS context recall | 0.256 | n/a* |
| Citation score | 0.588 | 0.635 |
| Stage appropriateness | 0.537 | 0.584 |
| Disclaimer present | 0.765 | 0.529 |
| Actionability | 0.624 | 0.682 |
| Mean response time | 9.6 s | 10.5 s |

\* Multi-agent RAGAS is empty in this run — the orchestrator does not yet
propagate per-agent retrieved contexts into the eval harness.

### Known limitations in this run

- **Faithfulness 0.448** is below the 0.50 "Needs Work" threshold. MedGemma
  at `temperature=0.7` (used for this run) tends to elaborate beyond
  retrieved context. The vLLM serving config has since been lowered to
  `temperature=0.3`; a re-run is planned.
- **Context recall 0.256** is mostly an artefact of sparse `reference`
  answers in `test_queries_agentic.json` rather than retriever miss. Several
  queries have terse references (or none), which makes recall scoring
  unreliable.
- **Two intent-routing misses** (per-query `actual_intent` field):
  - `agentic_direct_02` ("What does eGFR stand for?") → routed RETRIEVAL
    instead of DIRECT. The keyword classifier in `agentic_rag/nodes.py`
    treats `eGFR` as a retrieval trigger.
  - `clarification_01` ("What should I do about my levels?") → routed
    OUT_OF_SCOPE instead of CLARIFICATION. The classifier has no
    "incomplete-context" detector.
- **Multi-agent over-fans-out**: 9 of 17 queries triggered the `multi`
  route with the RAG agent included, dragging routing precision to 0.60.
  The secondary-threshold logic in `multi_agent_rag/orchestrator.py`
  (currently 30 % of primary score) is too permissive.

### Historical note on `intent_accuracy`

In runs prior to 2026-04-25, `aggregate.intent_accuracy` was reported as
`0.0` because the graph stores intent values as lowercase enum strings
(`QueryIntent.RETRIEVAL.value == "retrieval"`) while `expected_intent` in
the queries file uses uppercase. The aggregate field was misleading; the
per-query `actual_intent` was always correct. Fixed in
`eval/run_agentic_eval.py` by comparing case-insensitively. Old per-query
data is still valid — only the rolled-up `intent_accuracy` was affected.

---

## Interpreting Results

### RAGAS score thresholds (medical RAG)

| Level | Score Range | Interpretation |
|-------|-------------|----------------|
| Excellent | > 0.85 | Production-ready quality |
| Good | 0.70 – 0.85 | Acceptable with monitoring |
| Needs Work | 0.50 – 0.70 | Investigate and improve |
| Poor | < 0.50 | Significant issues present |

### Common failure patterns

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| Low faithfulness | Model hallucinating beyond context | Tighter prompt, lower temperature |
| Low context precision | Retriever returning irrelevant docs | Improve embeddings, adjust `TOP_K` |
| Low context recall | Missing relevant documents | Better chunking, add data sources |
| Low answer relevancy | Answer doesn't address question | Improve prompt template |
| Low intent accuracy | Query classification errors | Expand keyword lists in `agentic_rag/nodes.py` |
| Low agent routing | Keyword overlap between agents | Refine routing keywords in `multi_agent_rag/orchestrator.py` |

### CKD-specific quality indicators

- **Citation score < 0.5** — responses aren't citing NICE / KidneyCareUK / KDIGO sources.
- **Disclaimer missing** — medical safety requirement not met.
- **Stage appropriateness < 0.5** — wrong dietary values or keywords for the CKD stage.
- **Actionability < 0.5** — responses too vague, not providing practical guidance.

---

## Pre-flight checklist

Before running, confirm:

- [ ] `.env` has `RAGAS_JUDGE_API_KEY` set (and `RAGAS_EMBEDDINGS_API_KEY` or `GOOGLE_API_KEY`).
- [ ] ChromaDB is populated: `Data/vectorstore/` has data.
- [ ] MedGemma is accessible (local: `HF_TOKEN` set; remote: `USE_REMOTE_MODELS=true` and the vLLM container is up).

---

## Extending the Evaluation

### Adding new test queries

Add entries to `eval/test_queries_agentic.json`:

```json
{
  "id": "unique_id",
  "query": "Your test question",
  "reference": "Expected ground truth answer (or null)",
  "category": "diet|medication|lifestyle|stages|multi_domain|...",
  "expected_intent": "RETRIEVAL|DIRECT|CLARIFICATION|OUT_OF_SCOPE",
  "ckd_stage": null,
  "expected_agents": ["diet", "medication"],
  "contains_pii": false
}
```

### Adding custom metrics

Extend `agentic_rag/evaluation/custom_metrics.py`:

```python
class CKDMetrics:
    def evaluate_my_metric(self, response: str) -> float:
        # custom evaluation logic
        return score
```

Then add it to the `evaluate()` method and `CKDMetricScores` dataclass.

---

## File map

```
eval/
├── test_queries.json              # 10 queries for retriever comparison
├── test_queries_agentic.json      # 20 queries for agentic + multi-agent
├── run_retriever_comparison.py    # Flat vs Tree retriever (RAGAS only)
├── run_agentic_eval.py            # Agentic + Multi-Agent (RAGAS + CKD + system)
└── results/                       # Timestamped JSON output (gitignored)

agentic_rag/evaluation/
├── ragas_eval.py                  # RAGAS v0.4.x evaluator (4 metrics)
├── custom_metrics.py              # CKD domain-specific metrics
└── langsmith_setup.py             # LangSmith tracing integration
```
