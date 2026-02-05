# CLAUDE.md - Project Context for AI Assistants

## Project Overview

**MedGemma RAG** is a 3-tier Retrieval-Augmented Generation system for Chronic Kidney Disease (CKD) management, built for the Kaggle MedGemma Impact Challenge (deadline: Feb 24, 2026).

The system helps CKD patients and healthcare providers get evidence-based information from clinical guidelines (KDIGO, NHS, UKKA).

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    3-TIER RAG ARCHITECTURE                      │
├─────────────────────────────────────────────────────────────────┤
│ Level 1: Simple RAG                                             │
│   Query → EmbeddingGemma → ChromaDB → MedGemma → Response       │
├─────────────────────────────────────────────────────────────────┤
│ Level 2: Agentic RAG (LangGraph)                                │
│   Query → PII Check → Intent Classification → Router →          │
│   [Retrieval|Direct|Clarify] → RAGAS Evaluation → Response      │
├─────────────────────────────────────────────────────────────────┤
│ Level 3: Multi-Agent                                            │
│   Query → Orchestrator → [Diet|Medication|Lifestyle|RAG Agent]  │
│   → Synthesize → Response                                       │
└─────────────────────────────────────────────────────────────────┘
```

## Key Directories

| Directory | Purpose |
|-----------|---------|
| `1_Retrieval_Augmented_Generation/` | Level 1: Basic RAG components |
| `2_Agentic_RAG/` | Level 2: LangGraph workflow + evaluation |
| `3_MultiAgent_RAG/` | Level 3: Specialized agents |
| `Data/documents/` | Source PDFs (9 clinical guidelines) |
| `Data/processed/` | Chunked documents |
| `Data/vectorstore/` | ChromaDB persistent storage |
| `docs/deployment/` | AWS deployment guides |
| `scripts/` | Setup and automation scripts |

## Key Files

| File | Purpose |
|------|---------|
| `config.py` | Central configuration (models, chunking, prompts) |
| `app.py` | Gradio UI entry point |
| `main.py` | CLI entry point |
| `requirements.txt` | Python dependencies |
| `.env` | Environment variables (HF_TOKEN, LangSmith) |

## Tech Stack

- **LLM**: MedGemma 4B (`google/medgemma-4b-it`) - 4-bit quantized
- **Embeddings**: EmbeddingGemma 300M (`google/embeddinggemma-300m`)
- **Vector DB**: ChromaDB with persistent storage
- **Framework**: LangChain + LangGraph
- **PII Detection**: Microsoft Presidio with custom NHS/medical recognizers
- **Evaluation**: RAGAS + custom CKD metrics
- **UI**: Gradio (multi-tab interface)

## Code Patterns

### Import Convention
```python
# Level 1 components
from 1_Retrieval_Augmented_Generation.embeddings import EmbeddingGemmaWrapper
from 1_Retrieval_Augmented_Generation.vectorstore import CKDVectorStore
from 1_Retrieval_Augmented_Generation.retriever import CKDRetriever
from 1_Retrieval_Augmented_Generation.chain import SimpleRAGChain, MedGemmaLLM

# Level 2 components
from 2_Agentic_RAG.graph import AgenticRAGGraph
from 2_Agentic_RAG.pii_handler import PIIHandler

# Level 3 components
from 3_MultiAgent_RAG.orchestrator import MultiAgentOrchestrator
```

### Device Detection Pattern
All model wrappers auto-detect: CUDA → MPS (Apple Silicon) → CPU

### Configuration Access
```python
from config import (
    MEDGEMMA_MODEL_ID,
    EMBEDDING_MODEL_ID,
    CHUNK_SIZE,
    TOP_K_RESULTS,
    HF_TOKEN,
)
```

## Common Tasks

### Run the Application
```bash
python app.py  # Gradio UI at localhost:7860
```

### Process New Documents
```bash
python -c "from Data.preprocessing import preprocess_documents; preprocess_documents()"
```

### Test Components
```bash
python test.py
python Data/test.py  # Data processing tests
```

## Environment Variables

Required in `.env`:
```
HF_TOKEN=your_huggingface_token  # Required for MedGemma access
```

Optional:
```
LANGSMITH_API_KEY=...
LANGSMITH_PROJECT=ckd-rag-system
LANGSMITH_TRACING=true
LOG_LEVEL=INFO
```

## Medical Domain Context

### CKD Stages (by eGFR)
- Stage 1: ≥90 (normal function with damage evidence)
- Stage 2: 60-89 (mildly reduced)
- Stage 3: 30-59 (moderately reduced)
- Stage 4: 15-29 (severely reduced)
- Stage 5: <15 (kidney failure)

### Key Dietary Restrictions
- **Potassium**: Restricted in stages 3-5 (high K foods: bananas, oranges, potatoes)
- **Phosphorus**: Restricted in stages 3-5 (high P foods: dairy, nuts, cola)
- **Sodium**: <2000mg/day in stages 3-5
- **Protein**: 0.6-0.8 g/kg/day in stages 3-5 (pre-dialysis)

### Medications to Flag
- **NSAIDs** (ibuprofen, naproxen): Avoid - nephrotoxic
- **ACE inhibitors/ARBs**: First-line for CKD with proteinuria
- **Metformin**: Caution in stage 4+, contraindicated in stage 5

## Development Guidelines

### When Modifying Code
1. Follow existing patterns in each level's directory
2. Update corresponding README.md if adding features
3. All medical responses must include source citations
4. All medical responses must include disclaimers
5. Test with `python test.py` before committing

### When Adding Documents
1. Place PDFs in `Data/documents/`
2. Run preprocessing to chunk and index
3. Verify retrieval quality with test queries

### Safety Requirements
- PII detection must run before processing queries (Level 2+)
- Never store actual patient data
- Always include medical disclaimers in responses
- Cite sources for all medical claims

## Deployment

### Local Development
```bash
uv sync  # or pip install -r requirements.txt
python app.py
```

### AWS G4 Spot Instance
```bash
# See docs/deployment/aws-g4-spot-setup.md for full guide
curl -s https://raw.githubusercontent.com/.../setup-g4-instance.sh | bash -s YOUR_HF_TOKEN
```

## Competition Context

- **Challenge**: Kaggle MedGemma Impact Challenge
- **Deadline**: February 24, 2026
- **Prize**: $100,000
- **Requirements**: 3-min video, technical overview, reproducible code

## Troubleshooting

### Model Loading Issues
- Ensure HF_TOKEN is set and has MedGemma access
- Check GPU memory (need ~4GB with 4-bit quantization)
- Verify CUDA/MPS availability for GPU acceleration

### Retrieval Quality Issues
- Check SIMILARITY_THRESHOLD in config.py (default 0.7)
- Verify documents are properly chunked in Data/processed/
- Test embedding quality with sample queries

### Import Errors
- Directory names start with numbers, use quotes in imports if needed
- Ensure __init__.py exists in each package directory
