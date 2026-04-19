#!/bin/bash
# Terminate EC2 spot instance
# Usage: ./scripts/stop.sh [--inspect] [--all]

set -e

REGION=$(terraform -chdir=infrastructure/terraform output -raw region 2>/dev/null || echo "us-west-2")

# Parse flags
NAMES=()
for arg in "$@"; do
  case "$arg" in
    --inspect) NAMES+=("medgemma-data-inspect") ;;
    --all)     NAMES+=("medgemma-model-server" "medgemma-data-inspect") ;;
  esac
done
# Default: model server only
if [ ${#NAMES[@]} -eq 0 ]; then
  NAMES=("medgemma-model-server")
fi

FOUND=false
for NAME in "${NAMES[@]}"; do
  INSTANCE_ID=$(aws ec2 describe-instances \
    --region "$REGION" \
    --filters "Name=tag:Name,Values=$NAME" \
              "Name=instance-state-name,Values=running,pending,stopping" \
    --query 'Reservations[0].Instances[0].InstanceId' \
    --output text)

  if [ -z "$INSTANCE_ID" ] || [ "$INSTANCE_ID" = "None" ]; then
    echo "No running instance found: $NAME"
    continue
  fi

  FOUND=true
  echo "Terminating $NAME ($INSTANCE_ID)..."
  aws ec2 terminate-instances --region "$REGION" --instance-ids "$INSTANCE_ID" --output text > /dev/null
  aws ec2 wait instance-terminated --region "$REGION" --instance-ids "$INSTANCE_ID"
  echo "  Terminated."
done

if [ "$FOUND" = false ]; then
  echo "Error: No running instances found in $REGION"
  exit 1
fi

echo ""
echo "Done."
