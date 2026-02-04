# Deployment Guides

Documentation for deploying the MedGemma RAG system to various platforms.

## Available Guides

### [AWS G4 Spot Instance Setup](./aws-g4-spot-setup.md)

**Best for**: Development, testing, and demos on GPU

**Cost**: ~$25/month (4 hours/day, 20 days/month)

**What you get**:
- NVIDIA T4 GPU (16GB VRAM)
- 4 vCPUs, 16GB RAM
- 100GB persistent storage
- 70% cost savings with spot instances

**Quick Start**:
```bash
# 1. Launch g4dn.xlarge spot instance from AWS console
# 2. SSH into instance
# 3. Run automated setup:
curl -s https://raw.githubusercontent.com/ntg2208/medgemma_RAG/master/scripts/setup-g4-instance.sh | bash -s YOUR_HF_TOKEN

# 4. Upload PDFs and start working
```

See the [full guide](./aws-g4-spot-setup.md) for detailed instructions.

---

## Choosing a Deployment Option

| Use Case | Recommended Platform | Estimated Cost |
|----------|---------------------|----------------|
| Development & testing | AWS G4 Spot | ~$25/month |
| Production (low traffic) | AWS G4 On-demand + Load Balancer | ~$400/month |
| Production (high traffic) | AWS SageMaker Endpoint | ~$600+/month |
| Local development | Your GPU machine | Free |
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
1. Check the [troubleshooting section](./aws-g4-spot-setup.md#troubleshooting) in the guide
2. Open an issue on GitHub
3. Refer to AWS documentation
