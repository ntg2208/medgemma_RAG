# S3 Model Cache Setup Guide

This guide explains how to use S3 for model caching to speed up EC2 instance startup from **10-15 minutes** to **30-60 seconds**.

## Overview

Instead of downloading models from HuggingFace every time (slow), we:
1. Upload models to S3 once (one-time setup)
2. EC2 instances sync from S3 (fast!)

**Benefits:**
- ⚡ **10-20x faster startup**: 30-60s vs 10-15 min
- 💰 **Cost**: Only ~$0.21/month for S3 storage (9.3GB)
- 📦 **Smaller snapshots**: No need to include models in EC2 snapshots
- 🔄 **No HuggingFace rate limits**: Direct S3 access

## One-Time Setup

### Step 1: Upload Models to S3

You can do this from either:
- **Option A**: Your local machine (if you have models downloaded)
- **Option B**: Your current EC2 instance (if models are already there)

#### Option A: From Local Machine

```bash
# 1. Install dependencies
pip install awscli huggingface-hub

# 2. Set your HuggingFace token
export HF_TOKEN=your_huggingface_token

# 3. Run upload script (will download models if needed, then upload to S3)
./scripts/upload-s3.sh medgemma-models-YOUR_ACCOUNT_ID

# This will:
# - Create S3 bucket (if needed)
# - Download models from HuggingFace (~10 min)
# - Upload to S3 (~5-10 min)
```

#### Option B: Launch throwaway instance to upload

```bash
# Launches a cheap t3.large that downloads from HuggingFace,
# uploads to S3, then self-terminates
bash scripts/model-upload.sh
```

#### Option C: From Existing EC2 Instance

```bash
# SSH to your EC2 instance
ssh medgemma-gpu

# Run upload script (models already downloaded at ~/models_cache)
cd ~/medgemma_RAG
./scripts/upload-s3.sh medgemma-models-YOUR_ACCOUNT_ID
```

**Note**: The script will show you the S3 bucket name. Save this for the next step!

### Step 2: Deploy Infrastructure with Terraform

Now deploy (or update) your EC2 infrastructure to use S3:

```bash
cd infrastructure/terraform

# Initialize Terraform (first time only)
terraform init

# Option 1: Let Terraform auto-generate bucket name
terraform apply

# Option 2: Use specific bucket name
terraform apply -var="s3_models_bucket=medgemma-models-YOUR_ACCOUNT_ID"

# Save the outputs
terraform output -json > outputs.json
```

Terraform will:
- ✅ Create S3 bucket (or use existing one, with `prevent_destroy`)
- ✅ Create IAM role for EC2 to access S3
- ✅ Create security group and instance profile

### Step 3: Launch Instance and Run Setup

```bash
# Launch spot instance (uses Terraform outputs for SG, profile, etc.)
./scripts/start.sh

# SSH to instance
ssh medgemma-gpu

# Run setup script (will sync from S3 instead of downloading!)
cd ~/medgemma_RAG
bash scripts/startup.sh YOUR_HF_TOKEN
```

**Expected timeline:**
- [1-9/11] Package installation: ~5-7 min
- [10/11] **Model sync from S3**: ~30-60 seconds ⚡
- [11/11] Verification: ~30 seconds

**Total: ~6-8 minutes** (vs 20-30 min with HuggingFace download)

## Ongoing Usage

### Starting Fresh Instances

Every time you launch a new instance:

```bash
# 1. Launch spot instance
./scripts/start.sh

# 2. SSH and setup (fast because S3 sync!)
ssh medgemma-gpu
cd ~/medgemma_RAG
bash scripts/startup.sh YOUR_HF_TOKEN

# Models sync from S3 in 30-60 seconds!
```

### Updating Models

If you need to update models (e.g., new version of MedGemma):

```bash
# 1. Download new models locally
export HF_TOKEN=your_token
python -c "
from huggingface_hub import snapshot_download
snapshot_download('google/medgemma-1.5-4b-it', token='$HF_TOKEN')
"

# 2. Re-upload to S3
./scripts/upload-s3.sh YOUR_BUCKET_NAME

# 3. New instances will automatically get the updated models
```

## Cost Analysis

### S3 Storage
- **MedGemma 1.5 4B**: ~8GB
- **EmbeddingGemma 300M**: ~1.3GB
- **Total**: ~9.3GB

**Monthly cost (us-east-2):**
- Storage: 9.3GB × $0.023/GB = **$0.21/month**
- Data transfer (within AWS): **$0** (free)

### EC2 Savings
- **Old approach**: Keep instance running OR download every time (10-15 min)
- **New approach**: Stop instance when not needed, fast startup (30-60s)

**Example savings (using instance 4 hrs/day):**
- Old: Keep running 24/7 → $0.35/hr × 24 = **$8.40/day**
- New: Run 4 hrs/day → $0.35/hr × 4 = **$1.40/day**
- **Savings: $7/day = $210/month** (minus $0.21 S3 cost)

## Troubleshooting

### Models not syncing from S3

Check if S3_MODELS_BUCKET env var is set:
```bash
echo $S3_MODELS_BUCKET
# Should show: medgemma-models-XXXXX
```

If empty, manually set it:
```bash
export S3_MODELS_BUCKET=your-bucket-name
bash scripts/startup.sh YOUR_HF_TOKEN
```

### S3 permission denied

Verify IAM role is attached to instance:
```bash
aws sts get-caller-identity
# Should show the medgemma-model-server-role
```

If not attached, check Terraform:
```bash
cd infrastructure/terraform
terraform output
# Verify iam_instance_profile is set
```

### Fallback to HuggingFace

If S3 sync fails, setup script automatically falls back to HuggingFace download:
```bash
[10/11] Getting models...
Syncing models from S3: s3://medgemma-models-123/hub/
✗ Failed to sync from S3, falling back to HuggingFace download...
Downloading models from HuggingFace (this will take 10-15 minutes)...
```

This ensures setup always completes successfully.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    S3 MODEL CACHE FLOW                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. ONE-TIME SETUP (Local or existing EC2)                  │
│     HuggingFace → Download Models → Upload to S3            │
│     (10-15 min)         (~9.3GB)        (~5-10 min)         │
│                                                              │
│  2. NEW EC2 INSTANCE STARTUP                                │
│     S3 Bucket → AWS S3 Sync → ~/models_cache               │
│     (~9.3GB)    (30-60 sec!)    (local disk)                │
│                                                              │
│  3. MODEL LOADING (unchanged)                               │
│     ~/models_cache → vLLM loads into GPU VRAM               │
│     (local disk)       (2-3 min)                            │
│                                                              │
└─────────────────────────────────────────────────────────────┘

Total new instance startup: ~6-8 min
(vs 20-30 min with HuggingFace download)
```

## Summary

✅ **One-time**: Upload models to S3 (~15-20 min total)
✅ **Every instance**: Fast S3 sync (30-60 sec)
✅ **Cost**: Only $0.21/month for S3 storage
✅ **Fallback**: Auto-download from HuggingFace if S3 fails

This setup is especially valuable if you:
- Launch instances frequently for testing
- Use spot instances that get interrupted
- Want to minimize instance uptime (cost savings)
- Need fast iteration cycles
