# RAGAS Evaluation Notes — CKD RAG System

## What is RAGAS?

**RAGAS** (Retrieval Augmented Generation Assessment) is a framework for evaluating RAG pipelines without human-labelled QA pairs. It uses an **LLM-as-judge** (a cheap external model like Gemini Flash) to score your pipeline's outputs.

Key idea: you don't need a perfect golden dataset. RAGAS infers quality by comparing:
- what the LLM *said* vs what the context *supports* (faithfulness)
- what the LLM *said* vs what the *question asked* (answer relevancy)
- whether the *retrieved docs* were actually relevant (context precision)
- whether the retrieved docs *covered all necessary info* (context recall)

---

## The 4 Core Metrics

### 1. Faithfulness (0–1, higher is better)
> "Did the answer only say things that are supported by the retrieved context?"

- Breaks the answer into atomic claims → checks each claim against context
- **Low score = hallucination** — model invented facts not in the retrieved chunks
- Does NOT need a reference/ground-truth answer
- Most important metric for a medical RAG: we never want unsupported claims

### 2. Answer Relevancy (0–1, higher is better)
> "Did the answer actually address the question that was asked?"

- Generates N paraphrased questions from the answer → embeds them → measures cosine similarity to original question
- Low score = the answer is off-topic or too generic
- Does NOT need a reference answer
- Uses embeddings (configured via `RAGAS_EMBEDDINGS_MODEL` in `.env`)

### 3. Context Precision (0–1, higher is better)
> "Were the retrieved chunks actually useful/relevant?"

- Asks the judge LLM whether each retrieved chunk was needed to answer the question
- We use `LLMContextPrecisionWithoutReference` (no ground truth needed)
- Low score = retriever is returning noise/irrelevant chunks
- Directly measures retriever quality independent of the answer

### 4. Context Recall (0–1, higher is better)
> "Did the retriever find all the information needed to answer the question?"

- Compares the answer to a **reference (ground truth)** answer
- Checks whether every claim in the reference is attributable to retrieved context
- **Requires a reference answer** — queries without `reference` in the JSON get score 0.0
- Low score = retriever is missing important chunks

---

## How They Fit Together

```
Query ──► Retriever ──► Context chunks ──► LLM ──► Answer
                │                                      │
                └── Context Precision                  │
                └── Context Recall (needs ref)         │
                                                       └── Faithfulness
                                                       └── Answer Relevancy
```

| Metric | What it tells you | Needs reference? |
|--------|-------------------|-----------------|
| Faithfulness | LLM hallucination risk | No |
| Answer Relevancy | Answer quality / focus | No |
| Context Precision | Retriever noise | No |
| Context Recall | Retriever coverage | **Yes** |

---

## The RAGAS Judge LLM

RAGAS calls an **external LLM API** for scoring — it does NOT use MedGemma.

In this project it's configured via `.env`:
```
RAGAS_JUDGE_MODEL=google/gemini-2.0-flash-001
RAGAS_JUDGE_API_KEY=your_openrouter_key
RAGAS_JUDGE_BASE_URL=https://openrouter.ai/api/v1
RAGAS_EMBEDDINGS_MODEL=text-embedding-3-small
```

Cost estimate: ~$0.001–0.005 per query evaluated. For 10 queries × 4 retrievers = 40 evaluations ≈ $0.05–0.20 total.

---

## Files in This Project

### Evaluation infrastructure

| File | Role |
|------|------|
| `agentic_rag/evaluation/ragas_eval.py` | Core RAGAS wrapper: `RAGASEvaluator`, `RAGASScores`, `create_evaluator()` |
| `eval/test_queries.json` | 10 CKD test queries with reference answers (9/10 have references) |
| `eval/run_retriever_comparison.py` | **Main eval script** — runs RAGAS on flat vs tree retriever |

### Retrievers available (in `simple_rag/`)

| Class | Description | Collection |
|-------|-------------|-----------|
| `CKDRetriever` | Flat cosine similarity retrieval | `ckd_guidelines` |
| `TreeRetriever` | 2-phase: section headings → chunks | `ckd_section_headings` + `ckd_guidelines` |
| `RaptorRetriever` | Hierarchical clustering summaries | `ckd_raptor` |
| `ContextualRetriever` | BM25 + semantic hybrid with context-enriched chunks | `ckd_contextual` |

---

## Running the Evaluation

### Flat retriever only (start here)
```bash
uv run python eval/run_retriever_comparison.py --retriever flat
```

### Results are saved to
```
eval/results/comparison_YYYYMMDD_HHMMSS.json
```

---

## What the Eval Script Does (step by step)

1. Loads `eval/test_queries.json` (10 queries)
2. Initialises `EmbeddingGemmaWrapper` + `CKDVectorStore`
3. Loads MedGemma via `get_llm()` (local or remote per `USE_REMOTE_MODELS`)
4. Creates `RAGASEvaluator` via `create_evaluator()`
5. For each query:
   - Retrieves docs with the retriever
   - Generates an answer with MedGemma
   - Calls `evaluator.evaluate(query, answer, contexts, reference)`
6. Prints side-by-side comparison table + saves JSON

---

## Pre-flight Checklist (flat eval only)

Before running, confirm:
- [ ] `.env` has `RAGAS_JUDGE_API_KEY` set
- [ ] ChromaDB is populated: `Data/vectorstore/` has data (main.py auto-loads on first run)
- [ ] MedGemma is accessible (local: `HF_TOKEN` set; remote: `USE_REMOTE_MODELS=true`)

---

## Interpreting Results

### Good scores to aim for in a medical RAG
| Metric | Acceptable | Good | Excellent |
|--------|-----------|------|-----------|
| Faithfulness | > 0.7 | > 0.85 | > 0.95 |
| Answer Relevancy | > 0.7 | > 0.85 | > 0.95 |
| Context Precision | > 0.6 | > 0.75 | > 0.90 |
| Context Recall | > 0.6 | > 0.75 | > 0.90 |

### Common failure patterns
- **Low faithfulness** → MedGemma adding domain knowledge not in retrieved chunks (common for medical LLMs)
- **Low context precision** → retriever returning loosely related chunks; tune `SIMILARITY_THRESHOLD` or `TOP_K_RESULTS` in `config.py`
- **Low context recall** → relevant info is in the index but retriever misses it
- **Low answer relevancy** → prompt template is too generic, or model is padding the answer
