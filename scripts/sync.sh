#!/bin/bash
# Sync code between local and EC2
#
# Usage:
#   ./scripts/sync.sh [ec2-ip]        # push local → EC2 (default)
#   ./scripts/sync.sh --pull [ec2-ip] # pull EC2 → local

set -e

KEY_FILE="$HOME/.ssh/medgemma-key.pem"
REMOTE_DIR="/home/ubuntu/medgemma_RAG"
LOCAL_DIR="$(pwd)"

RSYNC_EXCLUDES=(
  --exclude '.venv/'
  --exclude '__pycache__/'
  --exclude '*.pyc'
  --exclude '.pytest_cache/'
  --exclude '.git/'
  --exclude 'models_cache/'
  --exclude 'Data/vectorstore/'
  --exclude 'Data/processed_ocr/'
  --exclude 'Data/cleaned_documents/'
  --exclude 'Data/test_output/'
  --exclude '*.log'
  --exclude '.env'
  --exclude '.terraform/'
  --exclude '*.tfstate*'
  --exclude 'experiments.ipynb'
)

# Parse args
PULL=false
if [ "$1" = "--pull" ]; then
  PULL=true
  shift
fi

EC2_HOST="$1"
if [ -z "$EC2_HOST" ]; then
  REGION=$(terraform -chdir=infrastructure/terraform output -raw region 2>/dev/null || echo "us-east-1")
  EC2_HOST=$(aws ec2 describe-instances \
    --region "$REGION" \
    --filters "Name=tag:Name,Values=medgemma-model-server" \
              "Name=instance-state-name,Values=running" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text 2>/dev/null)
  if [ -z "$EC2_HOST" ] || [ "$EC2_HOST" = "None" ]; then
    echo "Error: No running instance found. Run ./scripts/start.sh first."
    echo "Usage: ./scripts/sync.sh [--pull] <ec2-ip>"
    exit 1
  fi
  echo "Using IP from AWS: $EC2_HOST"
fi

if [ ! -f "$KEY_FILE" ]; then
  echo "Error: SSH key not found at $KEY_FILE"
  exit 1
fi

# Test SSH connection
if ! ssh -i "$KEY_FILE" -o ConnectTimeout=5 -o StrictHostKeyChecking=no ubuntu@"$EC2_HOST" "true" 2>/dev/null; then
  echo "Error: Cannot connect to $EC2_HOST — is the instance running?"
  exit 1
fi

if [ "$PULL" = true ]; then
  echo "Pulling from EC2 ($EC2_HOST) → local..."
  rsync -avz --progress --delete \
    -e "ssh -i $KEY_FILE -o StrictHostKeyChecking=no" \
    "${RSYNC_EXCLUDES[@]}" \
    ubuntu@"$EC2_HOST":"$REMOTE_DIR"/ \
    "$LOCAL_DIR"/
  echo ""
  echo "Done. Review changes: git status"
else
  echo "Pushing local → EC2 ($EC2_HOST)..."
  rsync -avz --progress --delete \
    -e "ssh -i $KEY_FILE -o StrictHostKeyChecking=no" \
    "${RSYNC_EXCLUDES[@]}" \
    "$LOCAL_DIR"/ \
    ubuntu@"$EC2_HOST":"$REMOTE_DIR"/
  echo ""
  echo "Done. Restart servers if needed: ssh $EC2_HOST then bash startup.sh --start"
fi
