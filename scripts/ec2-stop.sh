#!/bin/bash
# Stop EC2 spot instance (preserves EBS)

set -e

INSTANCE_ID=$(terraform -chdir=infrastructure/terraform output -raw instance_id 2>/dev/null)

if [ -z "$INSTANCE_ID" ]; then
  echo "Error: Could not get instance ID from Terraform"
  echo "Make sure Terraform has been applied: cd infrastructure/terraform && terraform apply"
  exit 1
fi

echo "Stopping instance $INSTANCE_ID..."
aws ec2 stop-instances --instance-ids "$INSTANCE_ID"

echo "Waiting for instance to stop..."
aws ec2 wait instance-stopped --instance-ids "$INSTANCE_ID"

echo ""
echo "✓ Instance stopped"
echo "EBS volume preserved - no data lost"
echo ""
echo "To resume: ./scripts/ec2-start.sh"
