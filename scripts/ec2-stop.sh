#!/bin/bash
# Stop EC2 spot instance (preserves EBS)

set -e

# Get AWS region from Terraform variables (allow override via environment)
if [ -z "$AWS_REGION" ]; then
  AWS_REGION=$(grep -E '^\s*aws_region\s*=' infrastructure/terraform/terraform.tfvars | awk -F'=' '{print $2}' | awk '{print $1}' | tr -d '"')
  if [ -z "$AWS_REGION" ]; then
    echo "Warning: Could not determine AWS region from terraform.tfvars, defaulting to us-east-1"
    AWS_REGION="us-east-1"
  fi
fi

INSTANCE_ID=$(terraform -chdir=infrastructure/terraform output -raw instance_id 2>/dev/null)

if [ -z "$INSTANCE_ID" ]; then
  echo "Error: Could not get instance ID from Terraform"
  echo "Make sure Terraform has been applied: cd infrastructure/terraform && terraform apply"
  exit 1
fi

echo "Stopping instance $INSTANCE_ID..."
aws ec2 stop-instances --instance-ids "$INSTANCE_ID" --region "$AWS_REGION"

echo "Waiting for instance to stop..."
aws ec2 wait instance-stopped --instance-ids "$INSTANCE_ID" --region "$AWS_REGION"

echo ""
echo "✓ Instance stopped"
echo "EBS volume preserved - no data lost"
echo ""
echo "To resume: ./scripts/ec2-start.sh"
