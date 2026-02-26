#!/bin/bash
# Start vLLM and TEI servers on EC2

set -e

# Ensure basic PATH is set
export PATH="/usr/local/bin:/usr/bin:/bin:/usr/local/sbin:/usr/sbin:/sbin:$PATH"

# Get HF token from environment or file
export HF_TOKEN=${HF_TOKEN:-$(cat ~/.hf_token 2>/dev/null || cat .env 2>/dev/null | /usr/bin/grep HF_TOKEN | /usr/bin/cut -d '=' -f2)}
export HF_HOME=~/models_cache

if [ -z "$HF_TOKEN" ]; then
  echo "Warning: HF_TOKEN not found. Models may fail to load if they require authentication."
fi

echo "═══════════════════════════════════════"
echo "Starting vLLM Server"
echo "═══════════════════════════════════════"
echo ""

# Check if server is already running
if tmux has-session -t vllm 2>/dev/null; then
  echo "⚠ vLLM server already running"
  echo "  To restart: ./scripts/stop-model-server.sh && ./scripts/start-model-server.sh"
  exit 1
fi

echo "Starting vLLM server for MedGemma 1.5..."
tmux new-session -d -s vllm bash -c "\
  source ~/.bashrc 2>/dev/null || true && \
  export HF_TOKEN=$HF_TOKEN && \
  export HF_HOME=~/models_cache && \
  export PATH=\"/home/ubuntu/.local/bin:\$PATH\" && \
  cd ~/medgemma_RAG && \
  /home/ubuntu/.local/bin/uv run python -m vllm.entrypoints.openai.api_server \
    --model google/medgemma-1.5-4b-it \
    --host 0.0.0.0 \
    --port 8000 \
    --dtype bfloat16 \
    --max-model-len 128000 \
    --gpu-memory-utilization 0.7 \
    --trust-remote-code \
    --disable-log-requests"

echo ""
echo "✓ Server starting in background..."
echo ""
echo "Check logs:"
echo "  tmux attach -t vllm    # vLLM logs (Ctrl+B then D to detach)"
echo ""
echo "Endpoint (after model loads ~2-3 minutes):"
echo "  LLM: http://localhost:8000/v1"
echo ""
echo "Test:"
echo "  curl http://localhost:8000/v1/models"
