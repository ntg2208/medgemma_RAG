#!/bin/bash
# Terminate EC2 spot instance
# Usage: ./scripts/stop.sh

set -e

REGION=$(terraform -chdir=infrastructure/terraform output -raw region 2>/dev/null || echo "us-west-2")

INSTANCE_ID=$(aws ec2 describe-instances \
  --region "$REGION" \
  --filters "Name=tag:Name,Values=medgemma-model-server" \
            "Name=instance-state-name,Values=running,pending,stopping" \
  --query 'Reservations[0].Instances[0].InstanceId' \
  --output text)

if [ -z "$INSTANCE_ID" ] || [ "$INSTANCE_ID" = "None" ]; then
  echo "Error: No running instance found (tag Name=medgemma-model-server in $REGION)"
  exit 1
fi

echo "Terminating instance $INSTANCE_ID..."
aws ec2 terminate-instances --region "$REGION" --instance-ids "$INSTANCE_ID" --output text > /dev/null

echo "Waiting for instance to terminate..."
aws ec2 wait instance-terminated --region "$REGION" --instance-ids "$INSTANCE_ID"

echo ""
echo "Instance terminated."
echo "Relaunch with: cd infrastructure/terraform && terraform apply"
