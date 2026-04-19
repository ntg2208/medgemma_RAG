#!/bin/bash
# EC2 instance startup script — run this on the GPU instance after SSHing in.
#
# Usage:
#   bash startup.sh <hf-token>   # Full setup + start servers (first time or reinstall)
#   bash startup.sh --start      # Download models from S3 + start servers
#   bash startup.sh --stop       # Stop servers

set -e

export PATH="/home/ubuntu/.local/bin:/usr/local/bin:/usr/bin:/bin:$PATH"
# Models live on NVMe instance store (ephemeral, fast, ~250GB on g5/g6)
export HF_HOME="${HF_HOME:-/opt/dlami/nvme/models_cache}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ---------------------------------------------------------------------------
# Stop servers
# ---------------------------------------------------------------------------
if [ "$1" = "--stop" ]; then
  echo "Stopping model servers..."

  if docker ps -a --format '{{.Names}}' 2>/dev/null | grep -q vllm-server; then
    docker stop vllm-server 2>/dev/null && docker rm vllm-server 2>/dev/null && echo "  vLLM container removed"
  else
    echo "  vLLM not running"
  fi

  echo "Done."
  exit 0
fi

# ---------------------------------------------------------------------------
# Start servers (download models from S3 first)
# ---------------------------------------------------------------------------
if [ "$1" = "--start" ]; then
  # Ensure uv is installed
  if ! command -v uv &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
  fi

  if [ -z "$HF_TOKEN" ]; then
    HF_TOKEN=$(grep HF_TOKEN ~/medgemma_RAG/.env 2>/dev/null | cut -d= -f2 | tr -d ' ')
  fi

  if docker ps --format '{{.Names}}' 2>/dev/null | grep -q vllm-server; then
    echo "Servers already running. Stop first: bash startup.sh --stop"
    exit 1
  fi

  echo "Pulling Docker image..."
  docker pull vllm/vllm-openai:latest -q

  echo "Starting vLLM (MedGemma 1.5 4B IT)..."
  docker run -d --name vllm-server \
    --gpus all \
    -v "$HF_HOME":/root/.cache/huggingface \
    -e "HF_TOKEN=$HF_TOKEN" \
    -e "HUGGING_FACE_HUB_TOKEN=$HF_TOKEN" \
    -p 8000:8000 \
    vllm/vllm-openai:latest \
    --model google/medgemma-1.5-4b-it \
    --dtype bfloat16 \
    --override-generation-config '{"temperature":0.7,"top_p":0.9,"frequency_penalty":0.4,"presence_penalty":0.3}' \
    --max-model-len 65536 \
    --max-num-seqs 8 \
    --gpu-memory-utilization 0.85 \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 8000

  echo ""
  echo "vLLM starting (~2-3 min to load). Check logs:"
  echo "  docker logs -f vllm-server"
  echo ""
  echo "Test when ready:"
  echo "  curl http://localhost:8000/v1/models"
  echo ""
  echo "Note: EmbeddingGemma loads locally on CPU via Python (EMBEDDING_DEVICE=cpu in .env)"
  exit 0
fi

# ---------------------------------------------------------------------------
# Full setup (first time or reinstall)
# ---------------------------------------------------------------------------
if [ -z "$1" ]; then
  echo "Error: HuggingFace token required for full setup"
  echo "Usage: bash startup.sh <hf-token>"
  echo "       bash startup.sh --start    (if already set up)"
  exit 1
fi

HF_TOKEN="$1"

echo "================================================"
echo "MedGemma RAG — GPU Instance Setup"
echo "================================================"
echo ""

# GPU check
echo "[1/8] Checking GPU..."
if ! nvidia-smi &>/dev/null; then
  echo "Error: No GPU detected. Required: g5.xlarge or g6.xlarge (compute capability ≥8.0)"
  exit 1
fi
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader)
if (( $(echo "$COMPUTE_CAP < 8.0" | bc -l) )); then
  echo "Error: GPU compute capability $COMPUTE_CAP < 8.0 (bfloat16 not supported)"
  exit 1
fi
echo "  GPU: $GPU_NAME (compute $COMPUTE_CAP)"

echo ""
echo "[2/8] Installing system packages..."
sudo apt update -qq
sudo apt install -y git curl wget vim htop unzip bc tree > /dev/null

echo ""
echo "[3/8] Installing uv..."
if ! command -v uv &>/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
  grep -q '.local/bin' ~/.bashrc || echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
fi

echo ""
echo "[4/8] Setting up Python 3.12 environment..."
cd ~/medgemma_RAG
uv python install 3.12 2>/dev/null || true
uv venv --python 3.12
source .venv/bin/activate
uv pip install pip -q
uv pip install -r requirements.txt -q
uv pip install docling -q
uv pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl -q

echo ""
echo "[5/8] Moving Docker storage to NVMe..."
NVME=/opt/dlami/nvme
if [ ! -L /var/lib/docker ]; then
  sudo systemctl stop docker containerd
  sudo mkdir -p $NVME/docker $NVME/containerd
  if [ -d /var/lib/docker ]; then
    sudo rsync -aP /var/lib/docker/ $NVME/docker/
    sudo rm -rf /var/lib/docker
  fi
  if [ -d /var/lib/containerd ]; then
    sudo rsync -aP /var/lib/containerd/ $NVME/containerd/
    sudo rm -rf /var/lib/containerd
  fi
  sudo ln -sf $NVME/docker /var/lib/docker
  sudo ln -sf $NVME/containerd /var/lib/containerd
  sudo systemctl start containerd docker
  echo "  Docker now using NVMe storage"
else
  echo "  Already on NVMe"
fi

echo ""
echo "[6/8] Pulling Docker images..."
docker pull vllm/vllm-openai:latest -q

echo ""
echo "[7/8] Writing .env..."
cat > .env << EOF
HF_TOKEN=$HF_TOKEN
HF_HOME=/opt/dlami/nvme/models_cache
LOG_LEVEL=INFO
EOF

echo ""
echo "[8/8] Verifying PyTorch + CUDA..."
source .venv/bin/activate
python -c "
import torch
print(f'  PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}')"

echo ""
echo "Setup complete! Starting servers..."
echo ""
exec bash "$0" --start
