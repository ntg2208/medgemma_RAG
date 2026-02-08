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

## Deployment Modes

The system supports two deployment modes:

### Local Mode (Default)
All components run on the local machine:
```
MacBook/Local Machine
├── MedGemma 1.5 4B (MPS/CPU)
├── EmbeddingGemma 300M
├── ChromaDB
└── RAG Logic
```

### Remote Mode (Optional)
Models run on GPU server, logic runs locally:
```
Local MacBook              EC2 Spot Instance (us-east-2)
├── RAG Logic       ←─API──→  ├── vLLM (MedGemma 1.5 4B) :8000
├── ChromaDB                  ├── TEI (EmbeddingGemma)   :8001
└── Notebooks                 └── Docling OCR
```

**Infrastructure:**
- Instance: g4dn.xlarge (Tesla T4, 16GB VRAM)
- Storage: 75GB EBS (persistent) + 116GB instance storage (ephemeral)
- Models: `/data/models_cache/` on EBS (~9.3GB, persists across stop/start)
- Cost: ~$12/month (30 hours usage)

**Enable remote mode:**
```bash
export USE_REMOTE_MODELS=true
export MODEL_SERVER_URL=http://<ec2-ip>:8000
export EMBEDDING_SERVER_URL=http://<ec2-ip>:8001
```

See `docs/deployment/remote-model-server.md` for full setup and data persistence details.

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
| `infrastructure/terraform/` | EC2 infrastructure as code |

## Key Files

| File | Purpose |
|------|---------|
| `config.py` | Central configuration (models, chunking, prompts) |
| `app.py` | Gradio UI entry point |
| `main.py` | CLI entry point |
| `requirements.txt` | Python dependencies |
| `.env` | Environment variables (HF_TOKEN, LangSmith) |

## Tech Stack

### Core Models
- **LLM**: MedGemma 1.5 4B (`google/medgemma-1.5-4b-it`) - 4-bit quantized
- **Embeddings**: EmbeddingGemma 300M (`google/embeddinggemma-300m`)

### Frameworks
- **Vector DB**: ChromaDB with persistent storage
- **Framework**: LangChain + LangGraph
- **PII Detection**: Microsoft Presidio with custom NHS/medical recognizers
- **Evaluation**: RAGAS + custom CKD metrics
- **UI**: Gradio (multi-tab interface)

### Remote Inference (Optional)
- **vLLM**: v0.15.1, OpenAI-compatible API server for MedGemma
- **TEI**: HuggingFace Text Embeddings Inference for EmbeddingGemma
- **Docling**: v2.72.0, IBM document understanding for PDF OCR
- **Infrastructure**: Terraform for EC2 management (g4dn.xlarge spot instance)
- **Python**: 3.12 with uv package manager on EC2

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

### Model Factory Functions (New)
```python
from config import get_llm, get_embeddings

# Get LLM (local or remote based on USE_REMOTE_MODELS)
llm = get_llm()
response = llm.generate("What is CKD?")

# Get embeddings (local or remote)
embeddings = get_embeddings()
vector = embeddings.embed_query("chronic kidney disease")

# Toggle between modes via environment variables
# Local mode (default):
USE_REMOTE_MODELS=false

# Remote mode:
USE_REMOTE_MODELS=true MODEL_SERVER_URL=http://ec2-ip:8000
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

### Remote Model Server (Optional)
```bash
# Start EC2 spot instance
./scripts/ec2-start.sh
./scripts/ec2-status.sh  # Get new IP (changes on each start)

# SSH and start model servers (~2-3 min startup, models already on disk)
ssh -i ~/.ssh/medgemma-key.pem ubuntu@<ec2-ip>
cd /data/medgemma_RAG
./scripts/start-model-server.sh

# Test endpoints
curl http://<ec2-ip>:8000/v1/models
curl http://<ec2-ip>:8001/info

# Process PDFs with Docling OCR
./scripts/ocr-process.sh

# Stop servers and instance
./scripts/stop-model-server.sh
exit
./scripts/ec2-stop.sh  # Run locally
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
# Deploy infrastructure with Terraform
cd infrastructure/terraform
terraform init && terraform apply

# First-time setup on EC2 (~20-30 min for model downloads)
ssh -i ~/.ssh/medgemma-key.pem ubuntu@<ec2-ip>
cd /data/medgemma_RAG
bash scripts/setup-g4-instance.sh YOUR_HF_TOKEN

# See docs/deployment/aws-g4-spot-setup.md for full guide
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

### Remote Mode Issues
- **IP changed**: Run `./scripts/ec2-status.sh` after each start to get new IP
- **vLLM won't start**: Gemma3 models require `--dtype bfloat16`, check tmux logs with `tmux attach -t vllm`
- **Connection timeout**: Update security group with your current IP (`curl https://checkip.amazonaws.com`)
- **Disk full**: Models should be in `/data/models_cache/` (EBS), not `/opt/dlami/nvme/` (ephemeral)
