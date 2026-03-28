#!/bin/bash
# Create an AMI snapshot of the running EC2 instance
# Usage: ./scripts/snapshot.sh
#
# Captures the current state (code, deps, uv, Python venv, etc.)
# so next launch skips setup — straight to: bash startup.sh --start

set -e

INSTANCE_NAME="medgemma-model-server"
TF_DIR="infrastructure/terraform"

# Resolve region
if [ -n "$REGION" ]; then
  echo "Using region: $REGION"
else
  REGION=$(terraform -chdir="$TF_DIR" output -raw region 2>/dev/null || echo "")
  if [ -z "$REGION" ]; then
    echo "Error: REGION not set and no Terraform output."
    echo "Usage: REGION=eu-north-1 ./scripts/snapshot.sh"
    exit 1
  fi
fi

# Find the running instance
INSTANCE_ID=$(aws ec2 describe-instances --region "$REGION" \
  --filters "Name=tag:Name,Values=$INSTANCE_NAME" \
            "Name=instance-state-name,Values=running" \
  --query 'Reservations[0].Instances[0].InstanceId' \
  --output text)

if [ -z "$INSTANCE_ID" ] || [ "$INSTANCE_ID" = "None" ]; then
  echo "Error: No running instance found in $REGION"
  exit 1
fi

AMI_NAME="medgemma-ready-$(date +%Y%m%d-%H%M)"

echo "Creating AMI from $INSTANCE_ID in $REGION..."
echo "  Name: $AMI_NAME"
echo "  Instance keeps running (--no-reboot)"

AMI_ID=$(aws ec2 create-image --region "$REGION" \
  --instance-id "$INSTANCE_ID" \
  --name "$AMI_NAME" \
  --description "MedGemma RAG: uv, Python deps, spacy model, code pre-installed" \
  --no-reboot \
  --query 'ImageId' --output text)

echo ""
echo "AMI creation started: $AMI_ID"
echo ""
echo "Wait for it to be ready (~5-10 min):"
echo "  aws ec2 describe-images --region $REGION --image-ids $AMI_ID --query 'Images[0].State' --output text"
echo ""
echo "Launch with it:"
echo "  AMI_ID=$AMI_ID ./scripts/start.sh"
echo "  AMI_ID=$AMI_ID REGION=$REGION ./scripts/start.sh"
