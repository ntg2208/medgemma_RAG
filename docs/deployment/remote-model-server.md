# Remote Model Server Setup Guide

This guide explains how to set up a remote GPU server (EC2) running vLLM and TEI for MedGemma RAG, allowing you to code locally while leveraging GPU inference.

## Architecture Overview

```
Local MacBook (Development)          EC2 Spot Instance (GPU)
┌──────────────────────┐            ┌──────────────────────┐
│ Research notebooks   │   HTTP     │ vLLM Server          │
│ ChromaDB             │◄─────────►│ MedGemma 1.5 4B      │
│ RAG Logic            │  Port 8000 │                      │
│ PDF Processing       │            │ TEI Server           │
│ langchain-openai     │  Port 8001 │ EmbeddingGemma 300M  │
└──────────────────────┘            │                      │
                                    │ Docling OCR          │
                                    └──────────────────────┘
```

## Benefits

- **Cool MacBook**: No local GPU/CPU load during inference
- **Cost Effective**: ~$10/month (30 hours × $0.30/hr spot + $0.21/month S3)
- **Fast Startup**: ~5-6 min total (instance ~2 min, S3 model sync ~30-60 sec, server load ~2-3 min)
- **Flexible**: Easy to launch/terminate EC2 as needed
- **Production Ready**: vLLM and TEI are battle-tested inference servers

---

## Prerequisites

1. **AWS Account** with configured CLI (`aws configure`)
2. **Terraform** installed (`brew install terraform` on macOS)
3. **SSH Key** created (`~/.ssh/medgemma-key.pem`)
4. **Spot Instance Quota** approved (4 vCPUs for g5.xlarge)

---

## Initial Setup

### 1. Deploy Infrastructure with Terraform

```bash
cd infrastructure/terraform

# Copy example and configure
cp terraform.tfvars.example terraform.tfvars

# Edit terraform.tfvars with your settings
# IMPORTANT: Set allowed_cidr_blocks to your IP address
nano terraform.tfvars

# Initialize and apply
terraform init
terraform plan
terraform apply

# Save the instance ID for later
terraform output instance_id
```

### 2. Configure SSH

Add to `~/.ssh/config`:

```
Host medgemma-gpu
    HostName <instance-public-ip>
    User ubuntu
    IdentityFile ~/.ssh/medgemma-key.pem
    ServerAliveInterval 60
```

**Note**: IP changes when you stop/start the instance, so update `HostName` after each start.

### 3. Initial EC2 Setup

SSH into the instance and run the setup script:

```bash
ssh medgemma-gpu

# Run first-time setup (installs deps, syncs models from S3, starts servers)
cd ~/medgemma_RAG
bash scripts/startup.sh YOUR_HF_TOKEN

# This will:
# - Install system dependencies (apt, uv, Python 3.12, pip)
# - Pull TEI Docker image
# - Sync models from S3 (~30-60 sec) or download from HuggingFace
# - Start vLLM + TEI servers
# - Takes ~5-8 minutes (with S3 cache) or ~20-30 minutes (first time)
```

---

## Daily Workflow

### Starting Your Work Session

```bash
# 1. Launch spot instance (from local machine)
./scripts/start.sh

# 2. SSH into EC2
ssh medgemma-gpu

# 3. Start model servers (syncs models from S3, starts vLLM + TEI)
bash scripts/startup.sh --start

# 4. Check logs (optional)
tmux attach -t vllm   # Ctrl+B then D to detach
tmux attach -t tei

# 5. Test endpoints (wait 2-3 min for servers to load)
curl http://localhost:8000/v1/models
curl http://localhost:8001/info
```

### Working Locally

In your local terminal:

```bash
# Set environment variables
export USE_REMOTE_MODELS=true
export MODEL_SERVER_URL=http://<ec2-ip>:8000
export EMBEDDING_SERVER_URL=http://<ec2-ip>:8001

# Now run your code locally
python your_script.py

# Or start Jupyter notebook
jupyter notebook
```

### Stopping Your Session

```bash
# 1. Stop model servers (on EC2)
bash scripts/startup.sh --stop
exit

# 2. Terminate instance (from local machine)
./scripts/stop.sh
```

---

## Using Docling OCR on EC2

Process PDFs with GPU acceleration on EC2:

```bash
# SSH into EC2
ssh medgemma-gpu
cd ~/medgemma_RAG

# Process all PDFs in documents folder
./scripts/ocr-process.sh

# Or process a single PDF
python scripts/ocr-single.py Data/documents/guideline.pdf -o Data/processed_ocr/

# Download processed files to local
scp -i ~/.ssh/medgemma-key.pem -r ubuntu@<ec2-ip>:~/medgemma_RAG/Data/processed_ocr ./
```

---

## Testing the Setup

### 1. Test vLLM (LLM Generation)

```bash
# From local machine
curl http://<ec2-ip>:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "google/medgemma-1.5-4b-it",
    "messages": [{"role": "user", "content": "What is chronic kidney disease?"}],
    "max_tokens": 100
  }'
```

### 2. Test TEI (Embeddings)

```bash
curl http://<ec2-ip>:8001/embed \
  -H "Content-Type: application/json" \
  -d '{"inputs": "chronic kidney disease management"}'
```

### 3. Test from Python

```python
import os
os.environ["USE_REMOTE_MODELS"] = "true"
os.environ["MODEL_SERVER_URL"] = "http://<ec2-ip>:8000"
os.environ["EMBEDDING_SERVER_URL"] = "http://<ec2-ip>:8001"

from config import get_llm, get_embeddings

# Test LLM
llm = get_llm()
response = llm.generate("What is CKD stage 3?")
print(response)

# Test embeddings
embeddings = get_embeddings()
vector = embeddings.embed_query("kidney disease")
print(f"Embedding dimension: {len(vector)}")
```

---

## Troubleshooting

### vLLM fails to start

**Symptoms**: vLLM crashes or won't load model

**Solutions**:
```bash
# Check GPU memory
nvidia-smi

# Check vLLM logs
tmux attach -t vllm

# Reduce GPU memory utilization
# Edit the vLLM launch args in scripts/startup.sh
--gpu-memory-utilization 0.6  # Instead of 0.7
```

**Important**: MedGemma 1.5 (Gemma3 architecture) requires `--dtype bfloat16`. Do NOT use `--dtype half` or it will fail to load.

### TEI fails to start

**Symptoms**: TEI container exits immediately

**Solutions**:
```bash
# Check Docker logs
docker logs $(docker ps -a -q --filter ancestor=ghcr.io/huggingface/text-embeddings-inference | head -1)

# Ensure HF token is set
echo $HF_TOKEN

# Pull image again
docker pull ghcr.io/huggingface/text-embeddings-inference:latest
```

### IP address changed after launch

This is normal - spot instances get new IPs each launch.

**Solution**:
```bash
# Get current IP
./scripts/status.sh

# start.sh automatically updates ~/.ssh/config
# Update environment variables
export MODEL_SERVER_URL=http://<new-ip>:8000
export EMBEDDING_SERVER_URL=http://<new-ip>:8001
```

### Connection timeout from local machine

**Solutions**:
```bash
# 1. Check security group allows your IP
# In AWS Console → EC2 → Security Groups → medgemma-model-server-sg
# Verify inbound rules for ports 22, 8000, 8001

# 2. Get your current IP
curl https://checkip.amazonaws.com

# 3. Update security group if needed
# (via AWS Console or Terraform)
```

---

## Cost Optimization

### Current Setup (~$10/month for 30 hours)

- EC2 Spot: ~$0.30/hr × 30 hrs = $9
- S3 Storage: 9.3GB × $0.023/GB = $0.21/month
- Data Transfer (within AWS): Free

### To Reduce Costs Further

1. **Use fewer hours**: Only start EC2 when needed
2. **Terminate when done**: `./scripts/stop.sh` — EBS auto-deletes, no idle storage costs
3. **Use g6 when available**: Usually 10-15% cheaper than g5

---

## Advanced Configuration

### Environment Variables

Add to local `~/.bashrc` or `~/.zshrc`:

```bash
# Remote model configuration
export USE_REMOTE_MODELS=true
export MODEL_SERVER_URL=http://<ec2-ip>:8000
export EMBEDDING_SERVER_URL=http://<ec2-ip>:8001

# Alias for quick access
alias gpu-start='cd ~/path/to/medgemma_RAG && ./scripts/start.sh'
alias gpu-stop='cd ~/path/to/medgemma_RAG && ./scripts/stop.sh'
alias gpu-status='cd ~/path/to/medgemma_RAG && ./scripts/status.sh'
```

### Custom vLLM Configuration

Edit the vLLM launch args in `scripts/startup.sh` to customize:

```bash
--max-model-len 8192          # Longer context (uses more memory)
--gpu-memory-utilization 0.9  # Use more GPU memory
--tensor-parallel-size 1      # Multi-GPU (if using larger instance)
```

---

## Data Persistence

EBS volumes **auto-delete on termination** (to prevent orphaned volumes). Models persist in S3.

### What PERSISTS across instances
- ✅ Models in S3 (~9.3GB, synced on each startup in 30-60 sec)
- ✅ Code in git repository
- ✅ Terraform infrastructure (S3 bucket, IAM, SG)

### What is DELETED on termination
- ❌ EBS volume (code, venv, local model cache)
- ❌ Instance storage (`/opt/dlami/nvme/`)

**Note**: `start.sh` auto-syncs code to the new instance. `startup.sh --start` syncs models from S3.

### Quick Reference

**Stop for the night:**
```bash
# On EC2
bash scripts/startup.sh --stop
exit

# Locally
./scripts/stop.sh
```

**Resume tomorrow:**
```bash
# Locally — launch new spot instance
./scripts/start.sh

# SSH to EC2
ssh medgemma-gpu
bash scripts/startup.sh --start

# Wait 2-3 minutes, then use locally
# (start.sh prints the IP; status.sh shows it too)
export USE_REMOTE_MODELS=true
export MODEL_SERVER_URL=http://<new-ip>:8000
export EMBEDDING_SERVER_URL=http://<new-ip>:8001
```

### Emergency Recovery

**If spot instance is reclaimed:**
```bash
./scripts/start.sh  # Launch new spot instance
# Models persist in S3, synced on next startup
```

**If you accidentally terminate instance:**
- Models persist in S3 (not on EBS — EBS auto-deletes on termination)
- Launch a new instance: `./scripts/start.sh`
- Run first-time setup again: `bash scripts/startup.sh YOUR_HF_TOKEN`

---

## Technical Specifications

### Models
| Model | ID | Size | Notes |
|-------|-----|------|-------|
| LLM | `google/medgemma-1.5-4b-it` | ~8GB | Gemma3 architecture, requires bfloat16 |
| Embeddings | `google/embeddinggemma-300m` | ~1GB | 768-dim (Matryoshka) |

### Infrastructure
| Component | Version/Spec |
|-----------|--------------|
| Instance | g5.xlarge (NVIDIA A10G, 24GB VRAM) |
| Region | us-west-2 (Oregon) |
| vLLM | 0.15.1 |
| TEI | Latest (Docker) |
| Docling | 2.72.0 |
| Python | 3.12 (uv package manager) |
| Terraform | 1.5.7 |

### Server Configuration
| Setting | Value |
|---------|-------|
| vLLM Port | 8000 |
| TEI Port | 8001 |
| GPU Memory Utilization | 70% |
| Max Model Length | 4096 tokens |
| Dtype | bfloat16 |

### Files Created
| Category | Files |
|----------|-------|
| Infrastructure | `infrastructure/terraform/{main,variables,outputs}.tf` |
| EC2 Control | `scripts/{start,stop,status,sync}.sh` |
| Model Servers | `scripts/startup.sh` (--start / --stop) |
| Model Cache | `scripts/{download-models,upload-s3,model-upload}.sh` |
| OCR | `scripts/ocr-process.sh` |

---

## References

- [vLLM Documentation](https://docs.vllm.ai/)
- [TEI Documentation](https://huggingface.co/docs/text-embeddings-inference/)
- [Docling Documentation](https://docling-project.github.io/docling/)
- [MedGemma 1.5 Model Card](https://huggingface.co/google/medgemma-1.5-4b-it)
