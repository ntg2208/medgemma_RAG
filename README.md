# CKD Management RAG System

A 3-tier Retrieval-Augmented Generation (RAG) system for Chronic Kidney Disease (CKD) management, built for the **Kaggle MedGemma Impact Challenge**.

## Overview

This project provides an AI-powered assistant for CKD patients and healthcare providers, offering evidence-based information from:
- **NICE NG203 Guidelines** - CKD assessment and management
- **KidneyCareUK Resources** - Dietary and patient information
- **UK Kidney Association (UKKA)** - Clinical guidance
- **KDIGO Guidelines** - International CKD, anemia, and IgAN guidelines

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
│   ├── documents/               # Source PDFs (9 clinical guidelines)
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
│   ├── retriever.py             # CKDRetriever + HybridRetriever
│   ├── tree_retriever.py        # Section-route-then-chunk retrieval
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
├── tests/                       # Pytest test suite (36+ tests)
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

| Strategy | Description | Use Case |
|----------|-------------|----------|
| **CKDRetriever** | Flat similarity search with medical term expansion | Default for Agentic/Multi-Agent |
| **TreeRetriever** | Section-route-then-chunk (two-phase) | Default for Simple RAG |
| **HybridRetriever** | Reciprocal rank fusion of semantic results | Experimental |

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
uv run python main.py simple      # Level 1: Simple RAG
uv run python main.py agentic     # Level 2: Agentic RAG (LangGraph)
uv run python main.py multi       # Level 3: Multi-Agent
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
| `CHUNK_SIZE` | Document chunk size (tokens) | 800 |
| `CHUNK_OVERLAP` | Trailing blocks to carry as overlap | 1 |
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
| `RAGAS_EMBEDDINGS_MODEL` | Embeddings for relevancy metric | `text-embedding-3-small` |

### Remote Model Server Config

| Setting | Description | Default |
|---------|-------------|---------|
| `USE_REMOTE_MODELS` | Use remote vLLM server | `false` |
| `MODEL_SERVER_URL` | vLLM server URL | `http://localhost:8000` |
| `REMOTE_MODEL_ID` | Model ID on remote server | `google/medgemma-1.5-4b-it` |

## Evaluation

### RAGAS Metrics (v0.4.x)
- **Faithfulness**: Is the answer grounded in context?
- **Answer Relevancy**: Does it address the question?
- **Context Precision**: Are retrieved docs relevant?
- **Context Recall**: Are all relevant docs retrieved?

RAGAS uses a separate judge LLM (Gemini, OpenRouter, or OpenAI) to score responses. Configure via `RAGAS_JUDGE_*` environment variables.

### Custom CKD Metrics
- Citation accuracy (regex-based source detection)
- CKD stage appropriateness (stage-specific keyword presence)
- Medical disclaimer presence
- Actionability score
- Medical accuracy indicators (NSAID avoidance, safe dosing)

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
