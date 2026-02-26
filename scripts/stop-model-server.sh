#!/bin/bash
# Stop model servers

set -e

echo "Stopping vLLM server..."
echo ""

# Stop vLLM
if tmux has-session -t vllm 2>/dev/null; then
  echo "Stopping vLLM..."
  tmux kill-session -t vllm
  echo "✓ vLLM stopped"
else
  echo "vLLM not running"
fi

echo ""
echo "✓ Server stopped"
