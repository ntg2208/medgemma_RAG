#!/bin/bash
# Simple model server without vLLM (uses transformers directly)

set -e

export HF_TOKEN=${HF_TOKEN:-$(cat ~/.hf_token 2>/dev/null || cat .env 2>/dev/null | grep HF_TOKEN | cut -d '=' -f2)}
export HF_HOME=~/models_cache

echo "═══════════════════════════════════════"
echo "Starting Simple MedGemma Server"
echo "═══════════════════════════════════════"
echo ""

# Check if server is already running
if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null ; then
  echo "⚠ Port 8000 already in use"
  echo "  Kill it: lsof -ti:8000 | xargs kill -9"
  exit 1
fi

echo "[1/1] Starting MedGemma server on port 8000..."
echo ""

# Start server in background
nohup python3 scripts/simple_model_server.py > server.log 2>&1 &
SERVER_PID=$!

echo "✓ Server starting (PID: $SERVER_PID)"
echo ""
echo "Check logs:"
echo "  tail -f server.log"
echo ""
echo "Test:"
echo "  curl http://localhost:8000/v1/models"
echo ""
echo "Stop:"
echo "  kill $SERVER_PID"
echo "  # or: lsof -ti:8000 | xargs kill -9"
echo ""
