#!/bin/bash
# Upload MedGemma + EmbeddingGemma models to S3
# Run once from a machine with models already cached, or let it download them.
#
# Usage: ./scripts/upload-s3.sh <s3-bucket> [hf-token]
# Example: ./scripts/upload-s3.sh medgemma-models-123456789

set -e

if [ -z "$1" ]; then
  echo "Usage: ./scripts/upload-s3.sh <s3-bucket> [hf-token]"
  exit 1
fi

S3_BUCKET="$1"
HF_TOKEN="${2:-${HF_TOKEN}}"
MODELS_DIR="${HF_HOME:-$HOME/models_cache}"

echo "S3 Bucket:  $S3_BUCKET"
echo "Models Dir: $MODELS_DIR"
echo ""

# Check if already uploaded
if aws s3 ls "s3://$S3_BUCKET/hub/" 2>/dev/null | grep -q "models--google--medgemma"; then
  echo "Models already exist in S3. Nothing to do."
  echo "To re-upload: aws s3 rm s3://$S3_BUCKET/hub/ --recursive then re-run."
  exit 0
fi

# Ensure bucket exists
if ! aws s3 ls "s3://$S3_BUCKET" 2>/dev/null; then
  echo "Creating bucket $S3_BUCKET..."
  REGION=$(aws configure get region 2>/dev/null || echo "us-east-2")
  if [ "$REGION" = "us-east-1" ]; then
    aws s3api create-bucket --bucket "$S3_BUCKET"
  else
    aws s3api create-bucket --bucket "$S3_BUCKET" --region "$REGION" \
      --create-bucket-configuration LocationConstraint="$REGION"
  fi
  echo "Bucket created."
fi

# Download from HuggingFace if not cached locally
if [ ! -d "$MODELS_DIR/hub" ]; then
  echo "Models not found locally. Downloading from HuggingFace..."

  if [ -z "$HF_TOKEN" ]; then
    echo "Error: HF_TOKEN required. Pass as second arg or: export HF_TOKEN=hf_..."
    exit 1
  fi

  mkdir -p "$MODELS_DIR"
  export HF_HOME="$MODELS_DIR"

  python3 - <<PYEOF
from huggingface_hub import snapshot_download
import os

token = os.getenv('HF_TOKEN') or '$HF_TOKEN'
cache = os.path.expanduser('$MODELS_DIR')

print('Downloading MedGemma 1.5 4B (~8GB)...')
snapshot_download('google/medgemma-1.5-4b-it', token=token, cache_dir=cache)
print('Downloading EmbeddingGemma 300M (~1GB)...')
snapshot_download('google/embeddinggemma-300m', token=token, cache_dir=cache)
print('Downloads complete.')
PYEOF
fi

echo ""
echo "Uploading to s3://$S3_BUCKET/hub/ ..."
TOTAL_SIZE=$(du -sh "$MODELS_DIR/hub" | cut -f1)
echo "Size: $TOTAL_SIZE"
echo ""

aws s3 sync "$MODELS_DIR/hub/" "s3://$S3_BUCKET/hub/" \
  --delete \
  --storage-class STANDARD \
  --no-progress

echo ""
echo "Upload complete!"
echo "  s3://$S3_BUCKET/hub/  ($TOTAL_SIZE)"
echo "  Monthly cost: ~\$0.21 for 9.3GB (us-east-2)"
echo ""
echo "Set in terraform: S3_MODELS_BUCKET=$S3_BUCKET"
