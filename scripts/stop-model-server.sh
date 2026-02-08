#!/bin/bash
# Stop model servers

set -e

echo "Stopping model servers..."
echo ""

# Stop vLLM
if tmux has-session -t vllm 2>/dev/null; then
  echo "[1/2] Stopping vLLM..."
  tmux kill-session -t vllm
  echo "✓ vLLM stopped"
else
  echo "[1/2] vLLM not running"
fi

# Stop TEI
if tmux has-session -t tei 2>/dev/null; then
  echo "[2/2] Stopping TEI..."
  tmux kill-session -t tei
  # Also stop docker container
  docker stop $(docker ps -q --filter ancestor=ghcr.io/huggingface/text-embeddings-inference) 2>/dev/null || true
  echo "✓ TEI stopped"
else
  echo "[2/2] TEI not running"
fi

echo ""
echo "✓ All servers stopped"
