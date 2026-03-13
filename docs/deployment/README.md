# Deployment Guides

Documentation for deploying the MedGemma RAG system to various platforms.

## Available Guides

### [Remote Model Server](./remote-model-server.md) ⭐ Recommended

**Best for**: Local development with remote GPU inference

**Cost**: ~$10/month (30 hours spot + S3 model cache)

**What you get**:
- Code locally on MacBook, inference on EC2
- vLLM (MedGemma 1.5 4B) + TEI (EmbeddingGemma)
- Models cached in S3 (~30-60 sec sync on startup)
- Docling OCR for PDF processing

**Quick Start**:
```bash
# Deploy infrastructure with Terraform (one-time)
cd infrastructure/terraform
terraform init && terraform apply

# Launch spot instance and connect
./scripts/start.sh
ssh medgemma-gpu
bash scripts/startup.sh --start

# Use locally
export USE_REMOTE_MODELS=true
export MODEL_SERVER_URL=http://<ip>:8000
```

See the [full guide](./remote-model-server.md) for detailed instructions.

---

### [EC2 GPU Workflow](./ec2-workflow.md)

**Best for**: Understanding the full Terraform + scripts workflow

Explains how Terraform manages persistent infrastructure (S3, IAM, SG) while shell scripts handle ephemeral instances.

See the [full guide](./ec2-workflow.md) for detailed instructions.

---

## Choosing a Deployment Option

| Use Case | Recommended Platform | Estimated Cost |
|----------|---------------------|----------------|
| Local dev + remote GPU | Remote Model Server | ~$15-21/month |
| Local-only development | Your GPU machine | Free |
| Demo video recording | AWS GPU Spot (temporary) | ~$2-5 total |

---

## Other Guides

- [S3 Model Cache Setup](./s3-model-cache-setup.md) — One-time setup for fast model syncing
- [GPU Spot Strategy](./gpu-spot-strategy.md) — Multi-region, multi-instance type strategy
