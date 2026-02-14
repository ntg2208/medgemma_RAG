#!/bin/bash
# Check EC2 instance status

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

STATUS=$(aws ec2 describe-instances --instance-ids "$INSTANCE_ID" --region "$AWS_REGION" \
  --query 'Reservations[0].Instances[0].State.Name' --output text)

PUBLIC_IP=$(aws ec2 describe-instances --instance-ids "$INSTANCE_ID" --region "$AWS_REGION" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)

echo "═══════════════════════════════════════"
echo "MedGemma Model Server Status"
echo "═══════════════════════════════════════"
echo "Instance ID: $INSTANCE_ID"
echo "Status:      $STATUS"
echo "Public IP:   $PUBLIC_IP"
echo ""

if [ "$STATUS" = "running" ]; then
  echo "Endpoints:"
  echo "  vLLM:  http://$PUBLIC_IP:8000/v1/models"
  echo "  TEI:   http://$PUBLIC_IP:8001/info"
  echo ""
  echo "SSH: ssh -i ~/.ssh/medgemma-key.pem ubuntu@$PUBLIC_IP"
elif [ "$STATUS" = "stopped" ]; then
  echo "Instance is stopped. Start with: ./scripts/ec2-start.sh"
else
  echo "Instance state: $STATUS"
fi
