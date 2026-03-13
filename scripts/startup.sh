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

  if tmux has-session -t vllm 2>/dev/null; then
    tmux kill-session -t vllm 2>/dev/null && echo "  vLLM stopped"
  else
    echo "  vLLM not running"
  fi

  if tmux has-session -t tei 2>/dev/null; then
    tmux kill-session -t tei 2>/dev/null && echo "  TEI stopped"
  else
    echo "  TEI not running"
  fi

  if docker ps -a --format '{{.Names}}' 2>/dev/null | grep -q tei-server; then
    docker stop tei-server 2>/dev/null && docker rm tei-server 2>/dev/null && echo "  TEI container removed"
  fi

  echo "Done."
  exit 0
fi

# ---------------------------------------------------------------------------
# Start servers (download models from S3 first)
# ---------------------------------------------------------------------------
if [ "$1" = "--start" ]; then
  if [ -z "$HF_TOKEN" ]; then
    HF_TOKEN=$(grep HF_TOKEN ~/medgemma_RAG/.env 2>/dev/null | cut -d= -f2 | tr -d ' ')
  fi

  if tmux has-session -t vllm 2>/dev/null || tmux has-session -t tei 2>/dev/null; then
    echo "Servers already running. Stop first: bash startup.sh --stop"
    exit 1
  fi

  # Download models from S3 to NVMe
  bash "$SCRIPT_DIR/download-models.sh"

  # Resolve snapshot paths for local model loading
  MEDGEMMA_PATH=$(find "$HF_HOME/hub/models--google--medgemma-1.5-4b-it/snapshots" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | head -1)
  EMBEDDINGGEMMA_PATH=$(find "$HF_HOME/hub/models--google--embeddinggemma-300m/snapshots" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | head -1)

  if [ -z "$MEDGEMMA_PATH" ]; then
    echo "Error: MedGemma model not found in $HF_HOME/hub"
    echo "Run: bash download-models.sh"
    exit 1
  fi

  echo "MedGemma:       $MEDGEMMA_PATH"
  echo "EmbeddingGemma: $EMBEDDINGGEMMA_PATH"
  echo ""

  echo "Starting vLLM (MedGemma 1.5)..."
  tmux new-session -d -s vllm bash -c "
    export HF_TOKEN=$HF_TOKEN HF_HOME=$HF_HOME
    cd ~/medgemma_RAG
    uv run python -m vllm.entrypoints.openai.api_server \
      --model $MEDGEMMA_PATH \
      --host 0.0.0.0 --port 8000 \
      --dtype bfloat16 \
      --max-model-len 64000 \
      --gpu-memory-utilization 0.95 \
      --trust-remote-code \
      --disable-log-requests \
      --enable-prefix-caching \
      --max-num-seqs 16 \
      --max-num-batched-tokens 16384 \
      --enable-chunked-prefill
  "

  echo "Starting TEI (EmbeddingGemma)..."
  tmux new-session -d -s tei bash -c "
    docker run --name tei-server --gpus all -p 8001:80 \
      -v $EMBEDDINGGEMMA_PATH:/model:ro \
      ghcr.io/huggingface/text-embeddings-inference:latest \
      --model-id /model --port 80
  "

  echo ""
  echo "Servers starting (~2-3 min to load). Check logs:"
  echo "  tmux attach -t vllm   (Ctrl+B D to detach)"
  echo "  tmux attach -t tei"
  echo ""
  echo "Test when ready:"
  echo "  curl http://localhost:8000/v1/models"
  echo "  curl http://localhost:8001/info"
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
echo "[1/7] Checking GPU..."
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
echo "[2/7] Installing system packages..."
sudo apt update -qq
sudo apt install -y git curl wget vim htop tmux unzip bc tree > /dev/null

echo ""
echo "[3/7] Installing uv..."
if ! command -v uv &>/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  grep -q '.local/bin' ~/.bashrc || echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
fi

echo ""
echo "[4/7] Setting up Python 3.12 environment..."
cd ~/medgemma_RAG
uv python install 3.12 2>/dev/null || true
uv venv --python 3.12
source .venv/bin/activate
uv pip install pip -q
uv pip install -r requirements.txt -q
uv pip install vllm docling -q
uv pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl -q

echo ""
echo "[5/7] Pulling TEI Docker image..."
docker pull ghcr.io/huggingface/text-embeddings-inference:latest -q

echo ""
echo "[6/7] Writing .env..."
cat > .env << EOF
HF_TOKEN=$HF_TOKEN
HF_HOME=/opt/dlami/nvme/models_cache
LOG_LEVEL=INFO
EOF

echo ""
echo "[7/7] Verifying PyTorch + CUDA..."
source .venv/bin/activate
python -c "
import torch
print(f'  PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}')"

echo ""
echo "Setup complete! Starting servers..."
echo ""
exec bash "$0" --start
