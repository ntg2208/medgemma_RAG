# Deployment Guides

Documentation for deploying the MedGemma RAG system to various platforms.

## Available Guides

### [Remote Model Server](./remote-model-server.md) ⭐ Recommended

**Best for**: Local development with remote GPU inference

**Cost**: ~$12/month (2 hours/day, 15 days/month)

**What you get**:
- Code locally on MacBook, inference on EC2
- vLLM (MedGemma 1.5 4B) + TEI (EmbeddingGemma)
- Models persist on EBS (~2-3 min startup after stop/start)
- Docling OCR for PDF processing

**Quick Start**:
```bash
# Deploy with Terraform
cd infrastructure/terraform
terraform init && terraform apply

# Start and connect
./scripts/ec2-start.sh
./scripts/ec2-status.sh  # Get new IP
ssh -i ~/.ssh/medgemma-key.pem ubuntu@<ip>
./scripts/start-model-server.sh

# Use locally
export USE_REMOTE_MODELS=true
export MODEL_SERVER_URL=http://<ip>:8000
```

See the [full guide](./remote-model-server.md) for detailed instructions.

---

### [AWS G4 Spot Instance Setup](./aws-g4-spot-setup.md)

**Best for**: Running everything on EC2 (no local development)

**Cost**: ~$25/month (4 hours/day, 20 days/month)

**What you get**:
- NVIDIA T4 GPU (16GB VRAM)
- 4 vCPUs, 16GB RAM
- 75GB persistent EBS storage
- 70% cost savings with spot instances

See the [full guide](./aws-g4-spot-setup.md) for detailed instructions.

---

## Choosing a Deployment Option

| Use Case | Recommended Platform | Estimated Cost |
|----------|---------------------|----------------|
| Local dev + remote GPU | Remote Model Server | ~$12/month |
| Full EC2 development | AWS G4 Spot | ~$25/month |
| Production (low traffic) | AWS G4 On-demand + Load Balancer | ~$400/month |
| Production (high traffic) | AWS SageMaker Endpoint | ~$600+/month |
| Local-only development | Your GPU machine | Free |
| Demo video recording | AWS G4 Spot (temporary) | ~$2-5 total |

---

## Coming Soon

- Docker deployment guide
- AWS SageMaker deployment
- Google Cloud GPU setup
- Local GPU optimization guide

---

## Support

For issues or questions:
1. Check the [troubleshooting section](./remote-model-server.md#troubleshooting) in the guide
2. Open an issue on GitHub
3. Refer to AWS documentation
