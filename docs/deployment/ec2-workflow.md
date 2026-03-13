# EC2 GPU Workflow

## Overview

EC2 instances are **not** managed by Terraform. Terraform manages persistent infrastructure (S3 bucket, IAM role, security group). Instances are launched/terminated via shell scripts that read Terraform outputs.

```
Terraform (persistent)          Scripts (ephemeral)
├── S3 bucket (model cache)     ├── start.sh    → launch spot instance
├── IAM role + profile          ├── stop.sh     → terminate instance
├── Security group              ├── status.sh   → show instance info
└── Outputs (SG, profile, etc)  ├── sync.sh     → rsync code ↔ EC2
                                └── startup.sh  → setup & run servers (on EC2)
```

## One-Time Setup

### 1. Deploy infrastructure

```bash
cd infrastructure/terraform
terraform init
terraform apply
```

This creates: S3 bucket (with `prevent_destroy`), IAM role/profile for S3 access, security group.

### 2. Upload models to S3

Option A — from a machine with models already cached:
```bash
./scripts/upload-s3.sh <s3-bucket-name> [hf-token]
```

Option B — launch a cheap t3.large that downloads from HuggingFace, uploads to S3, then self-terminates:
```bash
bash scripts/model-upload.sh
```

Takes ~15-20 min first time, skips if models already exist in S3.

## Daily Workflow

### Start

```bash
# 1. Launch spot instance (auto-syncs code when ready)
./scripts/start.sh

# 2. SSH in
ssh medgemma-gpu

# 3. Start model servers (syncs models from S3, starts vLLM + TEI)
bash scripts/startup.sh --start

# 4. Verify (wait 2-3 min for servers to load)
curl http://localhost:8000/v1/models
curl http://localhost:8001/info
```

**Timeline:** ~5-6 min total (instance ~1-2 min, S3 sync ~30-60 sec, server load ~2-3 min)

### First time on a fresh instance

```bash
ssh medgemma-gpu
bash scripts/startup.sh <your-hf-token>
```

This runs full setup (apt packages, uv, Python 3.12, pip install, Docker pull) then auto-starts servers.

### Sync code changes

```bash
# Push local → EC2 (default)
./scripts/sync.sh

# Pull EC2 → local
./scripts/sync.sh --pull

# With explicit IP
./scripts/sync.sh 1.2.3.4
./scripts/sync.sh --pull 1.2.3.4
```

Auto-discovers the instance IP from AWS if not provided. Excludes `.venv`, `__pycache__`, `.git`, `models_cache`, vectorstore, `.env`, terraform state.

### Check status

```bash
./scripts/status.sh
```

Shows instance ID, state, IP, type, uptime, spot request ID.

### Stop

```bash
# On EC2: stop servers
bash scripts/startup.sh --stop
exit

# Locally: terminate instance
./scripts/stop.sh
```

EBS volumes auto-delete on termination. S3 bucket persists.

## Scripts Reference

| Script | Where | Purpose |
|--------|-------|---------|
| `start.sh` | Local | Launch spot instance, update SSH config, sync code |
| `stop.sh` | Local | Terminate instance |
| `status.sh` | Local | Show instance details |
| `sync.sh` | Local | rsync code between local ↔ EC2 |
| `startup.sh <token>` | EC2 | Full first-time setup + start servers |
| `startup.sh --start` | EC2 | Download models from S3 + start vLLM/TEI |
| `startup.sh --stop` | EC2 | Stop vLLM/TEI servers |
| `download-models.sh` | EC2 | Sync models from S3 → NVMe (called by startup.sh) |
| `upload-s3.sh` | Any | Upload models to S3 from local cache |
| `model-upload.sh` | Local | Launch throwaway instance to upload models to S3 |

## Server Details

- **vLLM** (port 8000): MedGemma 1.5 4B, bfloat16, 95% GPU utilization, prefix caching
- **TEI** (port 8001): EmbeddingGemma 300M via Docker

Both run in tmux sessions. Check logs:
```bash
tmux attach -t vllm    # Ctrl+B D to detach
tmux attach -t tei
```

## Cost

| Resource | Cost |
|----------|------|
| S3 storage (9.3GB) | ~$0.21/month |
| g6.xlarge spot | ~$0.25-0.35/hr |
| g5.xlarge spot | ~$0.30-0.40/hr |

Always run `./scripts/stop.sh` when done.

## Troubleshooting

**SSH connection refused after start.sh:** Wait 30-60 seconds. The instance needs time to boot. `start.sh` retries automatically.

**vLLM won't start:** Check tmux logs (`tmux attach -t vllm`). Common cause: wrong dtype — MedGemma requires `--dtype bfloat16`.

**IP changed:** Run `./scripts/status.sh`. Spot instances get a new IP each launch — `start.sh` updates `~/.ssh/config` automatically.

**Security group blocks access:** Your IP may have changed. Update `allowed_cidr_blocks` in `infrastructure/terraform/variables.tf` and run `terraform apply`.

**S3_MODELS_BUCKET not set on EC2:** Check `~/.bashrc` — it's set by user data on boot. Run `source ~/.bashrc` or set manually.

**Models not downloading from S3:** Verify the IAM instance profile is attached: `curl -s http://169.254.169.254/latest/meta-data/iam/security-credentials/`. Check bucket name: `echo $S3_MODELS_BUCKET`.
