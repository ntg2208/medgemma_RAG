#!/bin/bash
# Start vLLM and TEI servers on EC2

set -e

# Get HF token from environment or file
export HF_TOKEN=${HF_TOKEN:-$(cat ~/.hf_token 2>/dev/null || cat .env 2>/dev/null | grep HF_TOKEN | cut -d '=' -f2)}
export HF_HOME=~/models_cache

if [ -z "$HF_TOKEN" ]; then
  echo "Warning: HF_TOKEN not found. Models may fail to load if they require authentication."
fi

echo "═══════════════════════════════════════"
echo "Starting MedGemma Model Servers"
echo "═══════════════════════════════════════"
echo ""

# Check if servers are already running
if tmux has-session -t vllm 2>/dev/null; then
  echo "⚠ vLLM server already running"
  echo "  To restart: ./scripts/stop-model-server.sh && ./scripts/start-model-server.sh"
  exit 1
fi

echo "[1/2] Starting vLLM server for MedGemma 1.5..."
tmux new-session -d -s vllm "export HF_TOKEN=$HF_TOKEN; export HF_HOME=~/models_cache; export PATH=\"\$HOME/.local/bin:\$PATH\"; cd ~/medgemma_RAG && source .venv/bin/activate && python -m vllm.entrypoints.openai.api_server \
  --model google/medgemma-1.5-4b-it \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.7 \
  --trust-remote-code"

echo "[2/2] Starting TEI server for EmbeddingGemma..."
tmux new-session -d -s tei "docker run --rm --gpus all -p 8001:80 \
  -e HUGGING_FACE_HUB_TOKEN=$HF_TOKEN \
  ghcr.io/huggingface/text-embeddings-inference:latest \
  --model-id google/embeddinggemma-300m"

echo ""
echo "✓ Servers starting in background..."
echo ""
echo "Check logs:"
echo "  tmux attach -t vllm    # vLLM logs (Ctrl+B then D to detach)"
echo "  tmux attach -t tei     # TEI logs"
echo ""
echo "Endpoints (after models load ~2-3 minutes):"
echo "  LLM:        http://localhost:8000/v1"
echo "  Embeddings: http://localhost:8001"
echo ""
echo "Test:"
echo "  curl http://localhost:8000/v1/models"
echo "  curl http://localhost:8001/info"
