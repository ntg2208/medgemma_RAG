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
- **Cost Effective**: ~$15-21/month (30 hours × $0.30-0.40/hr spot + 75GB storage)
- **Fast Startup**: ~2-3 minutes (models persist on EBS, no re-downloading)
- **Flexible**: Easy to start/stop EC2 as needed
- **Production Ready**: vLLM 0.15.1 and TEI are battle-tested inference servers

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
Host medgemma-g4
    HostName <instance-public-ip>
    User ubuntu
    IdentityFile ~/.ssh/medgemma-key.pem
    ServerAliveInterval 60
```

**Note**: IP changes when you stop/start the instance, so update `HostName` after each start.

### 3. Initial EC2 Setup

SSH into the instance and run the setup script:

```bash
ssh medgemma-g4

# Clone repo (if not already done)
cd /data
git clone https://github.com/ntg2208/medgemma_RAG.git
cd medgemma_RAG

# Run setup script
bash scripts/setup-g4-instance.sh YOUR_HF_TOKEN

# This will:
# - Install system dependencies
# - Install Python 3.12 + vLLM + Docling
# - Pull TEI Docker image
# - Download MedGemma 1.5 4B and EmbeddingGemma models
# - Takes ~20-30 minutes
```

---

## Daily Workflow

### Starting Your Work Session

```bash
# 1. Start EC2 instance (from local machine)
./scripts/ec2-start.sh

# 2. Check status and get IP
./scripts/ec2-status.sh

# 3. SSH into EC2
ssh medgemma-g4

# 4. Start model servers
cd /data/medgemma_RAG
./scripts/start-model-server.sh

# 5. Check logs (optional)
tmux attach -t vllm   # Ctrl+B then D to detach
tmux attach -t tei

# 6. Test endpoints
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
ssh medgemma-g4
cd /data/medgemma_RAG
./scripts/stop-model-server.sh

# 2. Stop EC2 instance (from local machine)
./scripts/ec2-stop.sh
```

---

## Using Docling OCR on EC2

Process PDFs with GPU acceleration on EC2:

```bash
# SSH into EC2
ssh medgemma-g4
cd /data/medgemma_RAG

# Process all PDFs in documents folder
./scripts/ocr-process.sh

# Or process a single PDF
python scripts/ocr-single.py Data/documents/guideline.pdf -o Data/processed_ocr/

# Download processed files to local
scp -i ~/.ssh/medgemma-key.pem -r ubuntu@<ec2-ip>:/data/medgemma_RAG/Data/processed_ocr ./
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
# Edit scripts/start-model-server.sh
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

### IP address changed after stop/start

This is normal - spot instances get new IPs when restarted.

**Solution**:
```bash
# Get new IP
./scripts/ec2-status.sh

# Update SSH config
nano ~/.ssh/config
# Change HostName to new IP

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

### Current Setup (~$12/month for 30 hours)

- EC2 Spot: ~$0.16/hr × 30 hrs = $5
- EBS Storage: 75GB × $0.08/GB = $6
- Data Transfer: ~$1

### Storage Breakdown (75GB EBS)

- System & packages: ~35GB
- Python venv: ~11GB
- Models (`/data/models_cache/`): ~9.3GB
  - MedGemma 1.5 4B: ~8.1GB
  - EmbeddingGemma 300M: ~1.2GB
- Available for data/experiments: ~21GB

### To Reduce Costs Further

1. **Use fewer hours**: Only start EC2 when needed
2. **Stop servers when idle**: Models reload from disk in 2-3 minutes
3. **Consider On-Demand for critical work**: More expensive but no interruptions

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
alias ec2-start='cd ~/path/to/medgemma_RAG && ./scripts/ec2-start.sh'
alias ec2-stop='cd ~/path/to/medgemma_RAG && ./scripts/ec2-stop.sh'
alias ec2-status='cd ~/path/to/medgemma_RAG && ./scripts/ec2-status.sh'
```

### Custom vLLM Configuration

Edit `scripts/start-model-server.sh` to customize:

```bash
--max-model-len 8192          # Longer context (uses more memory)
--gpu-memory-utilization 0.9  # Use more GPU memory
--tensor-parallel-size 1      # Multi-GPU (if using larger instance)
```

---

## Data Persistence

### What PERSISTS (EBS Volume - 75GB)

When you stop/start the instance, these survive:
- ✅ All code: `/data/medgemma_RAG/`
- ✅ Python venv: `/data/medgemma_RAG/.venv/` (11GB)
- ✅ All models: `/data/models_cache/` (9.3GB)
  - MedGemma 1.5 4B (~8.1GB)
  - EmbeddingGemma 300M (~1.2GB)
- ✅ Docker images, system configuration
- ✅ Any files in `/data/`, `/home/ubuntu/`, etc.

**Cost**: $6/month (75GB × $0.08/GB)

### What is DELETED (Instance Storage - 116GB)

When you stop the instance, this is wiped:
- ❌ Anything in `/opt/dlami/nvme/`
- ❌ Any temporary files or cache stored there

**Note**: Models are stored on EBS, so no re-downloading needed after stop/start.

### Quick Reference

**Stop for the night:**
```bash
# On EC2
./scripts/stop-model-server.sh
exit

# Locally
./scripts/ec2-stop.sh
```

**Resume tomorrow:**
```bash
# Locally
./scripts/ec2-start.sh
./scripts/ec2-status.sh  # Get new IP

# SSH to EC2
ssh -i ~/.ssh/medgemma-key.pem ubuntu@<new-ip>
cd /data/medgemma_RAG
./scripts/start-model-server.sh

# Wait 2-3 minutes, then use locally
export USE_REMOTE_MODELS=true
export MODEL_SERVER_URL=http://<new-ip>:8000
export EMBEDDING_SERVER_URL=http://<new-ip>:8001
```

### Emergency Recovery

**If spot instance is reclaimed:**
```bash
./scripts/ec2-start.sh  # Starts instance again
# All data on EBS preserved, new IP assigned
```

**If you accidentally terminate instance:**
- Your EBS volume still exists with all data
- Create new instance in Terraform and attach this volume
- Or restore from Terraform state: `terraform apply`

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
| EC2 Control | `scripts/ec2-{start,stop,status}.sh` |
| Model Servers | `scripts/{start,stop}-model-server.sh` |
| OCR | `scripts/ocr-{process.sh,single.py}` |
| Setup | `scripts/setup-g4-instance.sh` |

---

## References

- [vLLM Documentation](https://docs.vllm.ai/)
- [TEI Documentation](https://huggingface.co/docs/text-embeddings-inference/)
- [Docling Documentation](https://docling-project.github.io/docling/)
- [MedGemma 1.5 Model Card](https://huggingface.co/google/medgemma-1.5-4b-it)
