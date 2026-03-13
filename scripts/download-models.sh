#!/bin/bash
# Download MedGemma + EmbeddingGemma from S3 to NVMe instance store.
# Run this on the EC2 instance before starting model servers.
#
# Usage: bash download-models.sh
#
# Requires: S3_MODELS_BUCKET env var (set automatically via user_data on boot)

set -e

S3_MODELS_BUCKET="${S3_MODELS_BUCKET:-}"
MODELS_DIR="/opt/dlami/nvme/models_cache"

if [ -z "$S3_MODELS_BUCKET" ]; then
  echo "Error: S3_MODELS_BUCKET not set"
  echo "Set it: export S3_MODELS_BUCKET=<your-bucket>"
  echo "Or check: cat ~/.bashrc | grep S3_MODELS_BUCKET"
  exit 1
fi

# Check NVMe is mounted
if [ ! -d "/opt/dlami/nvme" ]; then
  echo "Error: NVMe instance store not found at /opt/dlami/nvme"
  echo "Make sure you're on a g5.xlarge or g6.xlarge instance"
  exit 1
fi

# Skip if models already present
if [ -d "$MODELS_DIR/hub" ] && [ "$(ls -A "$MODELS_DIR/hub" 2>/dev/null)" ]; then
  echo "Models already cached at $MODELS_DIR"
  ls -1 "$MODELS_DIR/hub/"
  exit 0
fi

mkdir -p "$MODELS_DIR/hub"

echo "Syncing models from s3://$S3_MODELS_BUCKET/hub/ ..."
echo "(~30-60 seconds)"
echo ""

aws s3 sync "s3://$S3_MODELS_BUCKET/hub/" "$MODELS_DIR/hub/" --no-progress

echo ""
echo "Done. Models at $MODELS_DIR/hub/:"
ls -1 "$MODELS_DIR/hub/"
SIZE=$(du -sh "$MODELS_DIR/hub" | cut -f1)
echo "Total: $SIZE"
