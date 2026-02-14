#!/bin/bash
# Start EC2 spot instance

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

echo "Starting instance $INSTANCE_ID..."
aws ec2 start-instances --instance-ids "$INSTANCE_ID" --region "$AWS_REGION"

echo "Waiting for instance to be running..."
aws ec2 wait instance-running --instance-ids "$INSTANCE_ID" --region "$AWS_REGION"

# Get new public IP
PUBLIC_IP=$(aws ec2 describe-instances --instance-ids "$INSTANCE_ID" --region "$AWS_REGION" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)

echo ""
echo "✓ Instance running!"
echo "Instance ID: $INSTANCE_ID"
echo "Public IP:   $PUBLIC_IP"
echo ""
echo "SSH: ssh -i ~/.ssh/medgemma-key.pem ubuntu@$PUBLIC_IP"
echo ""
echo "Update your ~/.ssh/config with new IP:"
echo "  Host medgemma-g4"
echo "    HostName $PUBLIC_IP"
echo "    User ubuntu"
echo "    IdentityFile ~/.ssh/medgemma-key.pem"
