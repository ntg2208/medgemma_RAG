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
Local MacBook              EC2 Spot Instance (us-west-2)
├── RAG Logic       ←─API──→  ├── vLLM (MedGemma 1.5 4B) :8000
├── ChromaDB                  ├── TEI (EmbeddingGemma)   :8001
└── Notebooks                 └── Docling OCR
```

**Infrastructure:**
- Instance: g5.xlarge (A10G, 24GB) or g6.xlarge (L4, 24GB) - check availability first
- Storage: 75GB EBS gp3 (auto-deletes on termination to prevent orphaned volumes)
- **Models: S3 cached** (~9.3GB, 30-60 sec sync on startup) ⚡
- Spot Type: One-time (safer, no auto-restart surprises)
- Cost: ~$0.30-0.40/hr spot pricing + $0.21/month S3 storage

**S3 Model Caching (Implemented):**
- ✅ Models cached in persistent S3 bucket (survives EC2 termination)
- ✅ Fast startup: 30-60 sec S3 sync vs 5-10 min HuggingFace download
- ✅ One-time upload: `scripts/upload-models-to-s3.sh`
- ✅ Auto-sync on boot: `scripts/setup-gpu-instance.sh` checks S3 first
- ✅ S3 bucket protected: `prevent_destroy = true` in Terraform
- See: `docs/deployment/s3-model-cache-setup.md` for setup guide

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
- **Infrastructure**: Terraform for EC2 management (g5.xlarge spot instance)
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

### Check GPU Spot Availability
```bash
# Quick check (single region)
./scripts/check-gpu-spot-availability.sh us-west-2 g6.xlarge

# Comprehensive check (all US regions, both g5/g6)
./scripts/check-gpu-spot-multi-region.sh
# Shows best price/region combo and terraform values to use
```

### S3 Model Cache (One-Time Setup)
```bash
# Upload models to S3 (one-time, ~15-20 min first run, 0 sec after)
./scripts/upload-models-to-s3.sh medgemma-models-YOUR_ACCOUNT_ID

# Deploy infrastructure with S3 caching
cd infrastructure/terraform
terraform apply  # Creates S3 bucket with prevent_destroy=true
```

### Sync Code Between Local and EC2
```bash
# Backup from EC2 to local (excludes models, venv, cache)
./scripts/backup-from-ec2.sh <ec2-ip>

# Deploy local changes to EC2 (for quick testing)
./scripts/sync-to-ec2.sh <ec2-ip>
```

### Remote Model Server (Optional)
```bash
# Start EC2 spot instance
./scripts/ec2-start.sh
./scripts/ec2-status.sh  # Get new IP (changes on each start)

# SSH and start model servers (~2-3 min startup, models sync from S3 in 30-60s)
ssh -i ~/.ssh/medgemma-key.pem ubuntu@<ec2-ip>
cd ~/medgemma_RAG
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

### Codebase Refactoring (Completed Feb-Mar 2026)

The project has been refactored following LangChain/LangGraph best practices:

**Phase 1: Removed LangGraph State Wrapping**
- Removed `~56 lines` of wrapper functions (`_state_to_graph_state`, `_graph_state_to_dict`, `_wrap_node`)
- All nodes now use `dict` directly instead of TypedDict/dataclass conversions
- Removed `GraphState` dataclass in `2_Agentic_RAG/nodes.py`

**Phase 2: Added RetryPolicy**
- `RetryPolicy` from `langgraph.types` for error handling
- LLM nodes: `max_attempts=3, initial_interval=1.0s, max_interval=30.0s`
- API nodes: `max_attempts=3, initial_interval=2.0s, max_interval=60.0s`
- Evaluator: `max_attempts=2, initial_interval=1.0s`

**Phase 3: Created BaseAgent Interface**
- `3_MultiAgent_RAG/agents/base.py` with abstract `BaseAgent` class
- `AgentResponse` dataclass with `answer`, `confidence`, `disclaimer` fields
- All agents extend `BaseAgent`: Diet, Lifestyle, Medication, RAG
- Module loading via `importlib` with `sys.modules` caching

**Phase 4: Added Comprehensive Tests**
- 12 test files with 36+ passing tests
- `pytest.ini` and `tests/conftest.py` for configuration

### Testing

Run tests with pytest:
```bash
uv run pytest -v  # All tests
uv run pytest -v tests/test_base_agent.py  # Specific test file
```

Test patterns established:
- Agent interface tests (`test_*_agent.py`)
- BaseAgent contract tests (`test_base_agent.py`)
- Routing tests (`test_routing.py`)
- Retry policy tests (`test_retry_policy.py`)
- State consolidation tests (`test_state_consolidation.py`)

### When Modifying Code
1. Follow existing patterns in each level's directory
2. Update corresponding README.md if adding features
3. All medical responses must include source citations
4. All medical responses must include disclaimers
5. Test changes with `uv run pytest` before committing

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
cd ~/medgemma_RAG
bash scripts/setup-gpu-instance.sh YOUR_HF_TOKEN

# See docs/deployment/aws-gpu-spot-setup.md for full guide
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
- **llama.cpp won't start**: Check container logs with `docker logs -f llamacpp-server`
- **Connection timeout**: Update security group with your current IP (`curl https://checkip.amazonaws.com`)
- **Disk full**: Models should be in `~/models_cache/` (EBS home directory), not `/opt/dlami/nvme/` (ephemeral)
- **Low spot availability**: See `docs/deployment/gpu-spot-strategy.md` for multi-region/instance strategy

### GPU Instance Options
- **g5.xlarge**: NVIDIA A10G, 24GB VRAM, $0.30-0.40/hr spot, mature/stable
- **g6.xlarge**: NVIDIA L4, 24GB VRAM, $0.25-0.35/hr spot, better availability
- Both support bfloat16 (compute capability ≥8.0) required for MedGemma 1.5
- Check availability: `./scripts/check-gpu-spot-multi-region.sh`
- Strategy guide: `docs/deployment/gpu-spot-strategy.md`

### GPU Migration History
- **g4dn.xlarge (T4) → g5.xlarge (A10G)**: T4 lacks bfloat16 support
- **Persistent → One-time spots**: Prevents surprise costs (see incident below)

### AWS Spot Instance Cost Incident (Feb 2026)

**STATUS: RESOLVED - Switched from persistent to one-time spots (infrastructure/terraform/main.tf:70)**

**What happened:** Tried to launch spot in us-east-1, capacity unavailable, switched to us-east-2. But the persistent spot request in us-east-1 eventually got fulfilled and ran for 13+ hours unnoticed, costing ~$3 extra.

**The problem:** Persistent spot requests keep trying to launch instances until explicitly cancelled. Terminating the instance alone doesn't stop it - a new one will launch.

**Current solution:** Using one-time spots prevents auto-fulfillment after cancellation.

**Historical reference (for persistent spots only):**

**Prevention:** When spot capacity fails in a region:
```bash
# 1. CANCEL the spot request (not just terminate instance)
aws ec2 describe-spot-instance-requests --region us-east-1 \
  --query 'SpotInstanceRequests[*].[SpotInstanceRequestId,State]' --output table

aws ec2 cancel-spot-instance-requests --region us-east-1 \
  --spot-instance-request-ids <request-id>

# 2. Terminate any instances
aws ec2 terminate-instances --region us-east-1 --instance-ids <instance-id>

# 3. Delete orphaned EBS volumes
aws ec2 describe-volumes --region us-east-1 --query 'Volumes[*].[VolumeId,State]' --output table
aws ec2 delete-volume --region us-east-1 --volume-id <volume-id>
```

**Quick cleanup all in one region:**
```bash
REGION=us-east-1
# Cancel all spot requests
aws ec2 describe-spot-instance-requests --region $REGION --query 'SpotInstanceRequests[*].SpotInstanceRequestId' --output text | xargs -r aws ec2 cancel-spot-instance-requests --region $REGION --spot-instance-request-ids
# Terminate all instances
aws ec2 describe-instances --region $REGION --filters "Name=instance-state-name,Values=running,pending,stopped" --query 'Reservations[*].Instances[*].InstanceId' --output text | xargs -r aws ec2 terminate-instances --region $REGION --instance-ids
```
