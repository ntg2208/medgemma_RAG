# CKD Management RAG System

A 3-tier Retrieval-Augmented Generation (RAG) system for Chronic Kidney Disease (CKD) management, built for the **Kaggle MedGemma Impact Challenge**.

## Overview

This project provides an AI-powered assistant for CKD patients and healthcare providers, offering evidence-based information from:
- **NICE NG203 Guidelines** — CKD assessment and management
- **KidneyCareUK Resources** — Dietary and patient information
- **UK Kidney Association (UKKA)** — Clinical guidance
- **KDIGO Guidelines** — International CKD, anemia, and IgAN guidelines

22 clinical-guideline PDFs in total. The full document index is in
[`docs/data-pipeline.md`](docs/data-pipeline.md).

> 📚 **Looking for deeper docs?** Start with [`docs/README.md`](docs/README.md) — the documentation index.

## Features

### Three Levels of RAG

| Level | Description | Key Features |
|-------|-------------|--------------|
| **Level 1: Simple RAG** | Basic retrieval and generation | Tree-based retrieval, source citations, query expansion |
| **Level 2: Agentic RAG** | LangGraph workflow orchestration | PII detection, query routing, RAGAS evaluation |
| **Level 3: Multi-Agent** | Specialized domain agents | Diet, Medication, Lifestyle, Knowledge retrieval |

### Specialized Agents (Level 3)

- **RAG Agent**: General knowledge retrieval from guidelines
- **Diet Agent**: Personalized dietary recommendations (potassium, phosphorus, sodium, protein)
- **Medication Agent**: Kidney-safe medication guidance and drug interaction warnings
- **Lifestyle Agent**: Exercise, blood pressure, smoking, stress management guidance

## Tech Stack

| Component | Technology |
|-----------|------------|
| LLM | MedGemma 1.5 4B (`google/medgemma-1.5-4b-it`) |
| Embeddings | EmbeddingGemma 300M (`google/embeddinggemma-300m`) |
| Vector Store | ChromaDB (persistent) |
| Framework | LangChain + LangGraph |
| PII Detection | Microsoft Presidio |
| Evaluation | RAGAS 0.4.x + Custom CKD Metrics |
| Tracing | LangSmith |
| UI | Gradio |
| Remote Inference | vLLM + HuggingFace TEI |

## Project Structure

```
medgemma_RAG/
├── Data/
│   ├── documents/               # Source PDFs (22 clinical guidelines)
│   ├── processed_ocr/           # OCR output from Docling (markdown + JSON)
│   ├── cleaned_documents/       # LLM-cleaned markdown with metadata
│   ├── processed_with_sections/ # Section-split docs (main_text.md, references.md, metadata.json)
│   ├── processed/               # Exported chunk JSON files
│   ├── vectorstore/             # ChromaDB persistent storage
│   ├── preprocessing.py         # Block-aware chunking pipeline
│   ├── split_sections.py        # Document section splitter (LLM + heuristic)
│   ├── tree_builder.py          # Section tree construction from headings
│   ├── export_chunks.py         # Chunk export to JSON
│   └── test.py                  # OCR/cleaning test utilities
│
├── simple_rag/
│   ├── embeddings.py            # EmbeddingGemma wrapper (MRL support)
│   ├── vectorstore.py           # ChromaDB operations
│   ├── retriever.py             # CKDRetriever + HybridRetriever + factory
│   ├── tree_retriever.py        # Section-route-then-chunk retrieval
│   ├── raptor_builder.py        # RAPTOR tree builder (UMAP + GMM + LLM summary)
│   ├── raptor_retriever.py      # RAPTOR collapsed retrieval over all layers
│   ├── raptor_viz.py            # Pyvis interactive tree visualisation
│   ├── contextual_builder.py    # Anthropic Contextual Retrieval: per-chunk context + BM25
│   ├── contextual_retriever.py  # Hybrid semantic+BM25 retriever with RRF fusion
│   └── chain.py                 # Simple RAG chain with MedGemma
│
├── agentic_rag/
│   ├── pii_handler.py           # Presidio PII detection
│   ├── nodes.py                 # LangGraph node functions
│   ├── graph.py                 # Workflow definition with RetryPolicy
│   └── evaluation/
│       ├── ragas_eval.py        # RAGAS v0.4.x evaluation (Gemini/OpenRouter judge)
│       ├── custom_metrics.py    # CKD-specific metrics (citations, disclaimers, etc.)
│       └── langsmith_setup.py   # LangSmith tracing integration
│
├── multi_agent_rag/
│   ├── orchestrator.py          # Query routing
│   └── agents/                  # Specialized agents (BaseAgent interface)
│       ├── base.py              # Abstract BaseAgent + AgentResponse
│       ├── diet_agent.py
│       ├── medication_agent.py
│       ├── lifestyle_agent.py
│       └── rag_agent.py
│
├── eval/
│   ├── run_retriever_comparison.py   # RAGAS comparison (flat vs tree retriever)
│   ├── run_agentic_eval.py           # End-to-end RAGAS + CKD + system metrics
│   ├── test_queries.json             # 10 retriever-comparison queries
│   ├── test_queries_agentic.json     # 20 agentic / multi-agent queries
│   └── results/                      # Timestamped JSON output (gitignored)
│
├── scripts/
│   ├── build_raptor_index.py        # Build RAPTOR tree + index into ChromaDB
│   ├── build_contextual_index.py    # Contextualise chunks + build BM25 index
│   ├── visualize_raptor.py          # Render RAPTOR tree as interactive HTML
│   └── ...                          # OCR, EC2, S3, deployment scripts
│
├── tests/                       # Pytest test suite
│   ├── build_eval_ground_truth.py    # Keyword-based ground-truth builder
│   ├── eval_retriever.py             # Confusion-matrix retriever evaluator
│   ├── retriever_eval_dataset.json   # Generated ground-truth dataset
│   └── retriever_eval_results.json   # Latest evaluation run output
│
├── docs/                        # Documentation (see docs/README.md for index)
│   ├── README.md                # Documentation index / TOC
│   ├── architecture.md          # System architecture (3 levels, deployment)
│   ├── data-pipeline.md         # PDF → ChromaDB pipeline detail
│   ├── evaluation.md            # RAGAS concepts, run guide, latest results
│   ├── usage.md                 # Terminal demo + Gradio UI walkthrough
│   ├── TODO.md                  # Outstanding fix / update / upgrade items
│   ├── modules/                 # Per-tier deep dives
│   │   ├── simple-rag.md
│   │   ├── agentic-rag.md
│   │   └── multi-agent-rag.md
│   └── deployment/              # EC2 + vLLM operational docs
│
├── app.py                       # Gradio UI entry point
├── main.py                      # CLI terminal chat (all 3 levels)
├── config.py                    # Central configuration
└── requirements.txt             # Python dependencies
```

## Data Pipeline

The document processing pipeline has four stages:

```
PDF → Docling OCR → LLM Cleaning → Section Splitting → Block-Aware Chunking → ChromaDB
```

1. **OCR** (`scripts/ocr-process.sh`): Docling converts PDFs to markdown + JSON metadata
2. **Cleaning** (`Data/test.py`): LLM-assisted title generation, summary, artifact removal
3. **Section Splitting** (`Data/split_sections.py`): Separates front matter, main content, and references using LLM classification or heuristics
4. **Chunking** (`Data/preprocessing.py`): Block-aware algorithm that preserves paragraphs, lists, and tables as atomic units
5. **Tree Building** (`Data/tree_builder.py`): Constructs section hierarchy from heading numbering patterns
6. **Export** (`Data/export_chunks.py`): Exports chunks to JSON with metadata and statistics

## Retrieval Strategies

| Strategy | Description | Collection | Use Case |
|----------|-------------|------------|----------|
| **CKDRetriever** (`flat`) | Semantic search with medical term expansion | `ckd_guidelines` | Default for Agentic/Multi-Agent |
| **TreeRetriever** (`tree`) | Section-route-then-chunk (two-phase) | `ckd_guidelines` + `ckd_section_headings` | Default for Simple RAG |
| **HybridRetriever** | Reciprocal rank fusion of semantic results | `ckd_guidelines` | Experimental |
| **RaptorRetriever** (`raptor`) | Collapsed top-k over multi-layer summary tree | `ckd_raptor` | Hierarchical / thematic queries |
| **ContextualRetriever** (`contextual`) | Hybrid semantic + BM25 over LLM-contextualised chunks | `ckd_contextual` | Mixed lexical + semantic recall |

Select at runtime with `main.py --retriever {flat,tree,raptor,contextual}`.
RAPTOR and Contextual require a one-off index build (see below) before use.

### Building the RAPTOR and Contextual indices

Both build steps read from `Data/processed/*_chunks.json` and require the LLM
(`get_llm()`) plus embeddings (`get_embeddings()`), so you can point them at a
local model or a remote vLLM instance via `USE_REMOTE_MODELS`.

```bash
# RAPTOR: tree of LLM-summarised clusters (UMAP + GMM soft clustering)
uv run python scripts/build_raptor_index.py                # default max-depth=3
uv run python scripts/visualize_raptor.py --query "potassium"  # interactive HTML

# Contextual RAG: prepend per-chunk context + build BM25 index
uv run python scripts/build_contextual_index.py
```

> Both scripts currently call `delete_collection()` before re-indexing, so a
> rebuild fully replaces the previous index. Expect the Contextual build in
> particular to be slow because the context prompt includes the full source
> document for every chunk.

## Installation

### Prerequisites

- Python 3.12+
- CUDA-capable GPU (recommended) or Apple Silicon
- HuggingFace account with access to MedGemma
- `uv` package manager

### Setup

```bash
git clone <repository-url>
cd medgemma_RAG
uv sync
```

Configure environment:
```bash
cp .env.example .env
# Edit .env with:
#   HF_TOKEN=your_huggingface_token
#   RAGAS_JUDGE_API_KEY=your_openrouter_or_gemini_key  (for evaluation)
```

## Usage

### Terminal Chat (CLI)

```bash
uv run python main.py simple                        # Level 1, default tree retriever
uv run python main.py simple --retriever raptor      # Level 1 with RAPTOR
uv run python main.py agentic --retriever contextual # Level 2 with Contextual RAG
uv run python main.py multi --retriever flat         # Level 3 with CKDRetriever
uv run python main.py simple --show-context         # Print retrieved chunks before the answer
```

### Gradio Web UI

```bash
uv run python app.py  # Opens at localhost:7860
```

### Programmatic Usage

```python
from config import get_llm, get_embeddings

# Initialize components
embeddings = get_embeddings()
llm = get_llm()

# Level 1: Simple RAG
from simple_rag.vectorstore import CKDVectorStore
from simple_rag.tree_retriever import TreeRetriever
from simple_rag.chain import SimpleRAGChain

store = CKDVectorStore(embeddings)
retriever = TreeRetriever(vectorstore=store, embedding_function=embeddings)
chain = SimpleRAGChain(retriever=retriever, llm=llm)
response = chain.invoke("What are the dietary restrictions for CKD stage 3?")
```

## Configuration

Key settings in `config.py`:

| Setting | Description | Default |
|---------|-------------|---------|
| `EMBEDDING_DIMENSION` | Embedding vector size | 768 |
| `CHUNK_SIZE` | Document chunk size (tokens) | 2000 |
| `CHUNK_OVERLAP` | Trailing blocks to carry as overlap (capped at 150 tokens) | 200 |
| `TOP_K_RESULTS` | Documents to retrieve | 5 |
| `SIMILARITY_THRESHOLD` | Minimum similarity score | 0.3 |
| `SECTION_K` | Section headings to match (tree retrieval) | 8 |
| `CHUNKS_PER_SECTION` | Max chunks per matched section | 3 |

### RAGAS Evaluation Config

| Setting | Description | Default |
|---------|-------------|---------|
| `RAGAS_JUDGE_MODEL` | Judge LLM model ID | `google/gemini-2.0-flash-001` |
| `RAGAS_JUDGE_API_KEY` | API key for judge provider | (required) |
| `RAGAS_JUDGE_BASE_URL` | OpenAI-compatible endpoint | `https://openrouter.ai/api/v1` |
| `RAGAS_JUDGE_TIMEOUT` | Per-call timeout (seconds) | `600` |
| `RAGAS_JUDGE_MAX_RETRIES` | Per-call retries on failure | `3` |
| `RAGAS_EMBEDDINGS_MODEL` | Embeddings for relevancy metric | `gemini-embedding-2` |
| `RAGAS_EMBEDDINGS_BASE_URL` | Embeddings endpoint | `https://generativelanguage.googleapis.com/v1beta/openai/` |
| `RAGAS_EMBEDDINGS_API_KEY` | Falls back to `GOOGLE_API_KEY` | (required) |

### Remote Model Server Config

| Setting | Description | Default |
|---------|-------------|---------|
| `USE_REMOTE_MODELS` | Use remote vLLM server | `false` |
| `MODEL_SERVER_URL` | vLLM server URL | `http://localhost:8000` |
| `REMOTE_MODEL_ID` | Model ID on remote server | `google/medgemma-1.5-4b-it` |

## Evaluation

### Retriever confusion-matrix evaluation

The source-level retriever evaluator (`tests/eval_retriever.py`) runs a fixed
query set against a chosen retriever and reports TP/FP/FN/TN, precision,
recall, F1 and accuracy per-query, per-topic and in aggregate. Ground truth is
built from actual chunk content by keyword + density filtering (see
`tests/build_eval_ground_truth.py`), so retrieval effectiveness is measured
against sources that genuinely discuss each topic rather than filename guesses.

```bash
# (Re)generate ground truth from current chunks
uv run python tests/build_eval_ground_truth.py

# Evaluate a retriever (writes retriever_eval_results.json + confusion_matrix.png)
uv run python tests/eval_retriever.py --retriever basic  --k 5
uv run python tests/eval_retriever.py --retriever tree   --k 5
uv run python tests/eval_retriever.py --retriever hybrid --k 5
```

> **Caveat.** Ground truth is built by keyword matching; the retrievers score
> by semantic similarity. The metrics are a useful proxy but will systematically
> understate recall for semantically-adjacent but lexically-distinct sources.

### RAGAS retriever comparison

`eval/run_retriever_comparison.py` runs the configured retrievers through the
full chain (retrieval → MedGemma → RAGAS scoring) over the hand-authored queries
in `eval/test_queries.json` and prints a side-by-side metric table.

```bash
uv run python eval/run_retriever_comparison.py --retriever both
```

Requires `RAGAS_JUDGE_API_KEY` in `.env`.

### Agentic + Multi-Agent end-to-end evaluation

`eval/run_agentic_eval.py` runs the Agentic RAG (LangGraph) and Multi-Agent
orchestrator over `eval/test_queries_agentic.json` (20 queries spanning diet,
medication, lifestyle, stages, dialysis, OOS, clarification, multi-domain, and
PII handling). It scores RAGAS, CKD-specific, and system-level metrics
(intent routing, PII detection, agent routing) and writes a timestamped JSON to
`eval/results/`.

```bash
uv run python eval/run_agentic_eval.py                 # both systems
uv run python eval/run_agentic_eval.py --system agentic
uv run python eval/run_agentic_eval.py --skip-ragas    # CKD + system only (no judge)
```

See [`docs/evaluation.md`](docs/evaluation.md) for the full guide and the latest run summary.

### RAGAS Metrics (v0.4.x)
- **Faithfulness**: Is the answer grounded in context?
- **Answer Relevancy**: Does it address the question?
- **Context Precision**: Are retrieved docs relevant?
- **Context Recall**: Are all relevant docs retrieved?

RAGAS uses a separate judge LLM (Gemini, OpenRouter, or OpenAI) to score responses. Configure via `RAGAS_JUDGE_*` environment variables. Embeddings for the answer-relevancy metric are configured separately via `RAGAS_EMBEDDINGS_*` (defaults to Google AI Studio's `gemini-embedding-2`, since OpenRouter does not serve embeddings).

### Custom CKD Metrics
- Citation accuracy (regex-based source detection)
- CKD stage appropriateness (stage-specific keyword presence)
- Medical disclaimer presence
- Actionability score
- Medical accuracy indicators (NSAID avoidance, safe dosing)

### Retriever evaluation (latest)

Two retriever-only harnesses sit upstream of the end-to-end eval. Numbers
below are the latest runs; full per-topic and per-query breakdowns live in
[`docs/evaluation.md`](docs/evaluation.md).

**Source-level confusion matrix** (`tests/eval_retriever.py`, `CKDRetriever`,
`k=5`, 32 queries × 13 topics; ground truth in
`tests/retriever_eval_dataset.json`):

| Precision | Recall | F1 | Accuracy |
|---:|---:|---:|---:|
| 0.532 | 0.274 | 0.361 | 0.861 |

Per-topic F1 ranges from **0.833** (hyperkalaemia) down to **0.000** (RRT) —
keyword-based topics with broad source coverage (diet, medication,
lifestyle) systematically underperform here because ground truth is built
by lexical matching while the retriever scores semantically.

**End-to-end RAGAS retriever comparison** (`eval/run_retriever_comparison.py`,
flat retriever, 10 queries from `eval/test_queries.json`, run
2026-04-16):

| Faithfulness | Answer relevancy | Context precision | Context recall | Average |
|---:|---:|---:|---:|---:|
| 0.568 | 0.775 | 0.485 | 0.283 | 0.528 |

### Latest evaluation run (2026-04-24, ship snapshot)

Run: `eval/results/agentic_eval_20260424_204001.json` — 17 in-scope queries per
system.

**Run parameters (as-run, baseline configuration):**

| Component | Value |
|-----------|-------|
| Generator LLM | MedGemma 1.5 4B (`google/medgemma-1.5-4b-it`) on EC2 vLLM |
| Generation `temperature` | 0.7 (lowered to 0.3 in `scripts/startup.sh` after this run) |
| Embeddings | EmbeddingGemma 300M, dim 768 |
| Retriever (Agentic RAG) | `CKDRetriever` (flat semantic + medical-term expansion), `TOP_K_RESULTS=5`, `SIMILARITY_THRESHOLD=0.3` |
| Retriever (Multi-Agent → RAG agent) | Same `CKDRetriever` |
| Prompts | Baseline `RAG_SYSTEM_PROMPT` + `RAG_PROMPT_TEMPLATE` from `config.py` |
| Architecture | Default LangGraph pipeline + default orchestrator (no overrides) |
| Judge LLM | `google/gemini-2.0-flash-001` via OpenRouter |
| Judge embeddings (RAGAS relevancy) | `gemini-embedding-2` via Google AI Studio |
| Test queries | `eval/test_queries_agentic.json` (20 total, 17 in scope per system) |

| Metric | Agentic RAG | Multi-Agent RAG |
|--------|------------:|----------------:|
| Intent routing accuracy | **0.882** (15/17) | — |
| PII detection accuracy | **1.000** | — |
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

\* RAGAS scoring is not yet wired into the multi-agent path because the
synthesized response does not surface per-agent contexts — see "Known
limitations" below.

**Known limitations (planned for the next iteration):**

- **Faithfulness 0.45** is below the "Needs Work" threshold (0.50). MedGemma
  occasionally adds clinical reasoning beyond the retrieved chunks; tightening
  the prompt and lowering temperature (already changed to 0.3 in
  `scripts/startup.sh`, not yet re-evaluated) is the next experiment.
- **Context recall 0.26** mostly reflects sparse `reference` answers in
  `test_queries_agentic.json` rather than retriever miss — this metric needs
  more curated ground-truth references.
- **Two intent-routing misses**: `agentic_direct_02` ("What does eGFR stand
  for?") routes to RETRIEVAL instead of DIRECT; `clarification_01` ("What
  should I do about my levels?") routes to OUT_OF_SCOPE instead of
  CLARIFICATION. Both indicate the keyword-based classifier in
  `agentic_rag/nodes.py` needs broader coverage.
- **Multi-agent over-fans-out to RAG**: 9 of 17 queries triggered the
  `multi` route with the RAG agent included, hurting routing precision.
  Threshold tuning in `multi_agent_rag/orchestrator.py` is the obvious lever.
- **Multi-agent RAGAS scoring** is empty in this run — the orchestrator does
  not currently propagate retrieved contexts into the eval harness.

## Testing

```bash
uv run pytest -v                        # All tests
uv run pytest -v tests/test_ragas_eval.py  # RAGAS tests only
```

## Medical Disclaimer

**This tool is for educational purposes only.** It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.

## Competition

Built for the [Kaggle MedGemma Impact Challenge](https://www.kaggle.com/competitions/med-gemma-impact-challenge)

- **Deadline**: February 24, 2026
- **Prize Pool**: $100,000
- **Requirements**: 3-minute video, technical overview, reproducible code

## License

This project is for competition submission purposes. See LICENSE file for details.

## Acknowledgments

- Google Health AI for MedGemma
- NICE for clinical guidelines
- KidneyCareUK for patient resources
- KDIGO for international guidelines
- UK Kidney Association for clinical guidance
- LangChain and LangGraph teams
