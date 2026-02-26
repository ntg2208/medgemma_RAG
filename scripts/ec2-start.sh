#!/bin/bash
# Start EC2 spot instance

set -e

INSTANCE_ID=$(terraform -chdir=infrastructure/terraform output -raw instance_id 2>/dev/null)

if [ -z "$INSTANCE_ID" ]; then
  echo "Error: Could not get instance ID from Terraform"
  echo "Make sure Terraform has been applied: cd infrastructure/terraform && terraform apply"
  exit 1
fi

echo "Starting instance $INSTANCE_ID..."
aws ec2 start-instances --instance-ids "$INSTANCE_ID"

echo "Waiting for instance to be running..."
aws ec2 wait instance-running --instance-ids "$INSTANCE_ID"

# Get new public IP
PUBLIC_IP=$(aws ec2 describe-instances --instance-ids "$INSTANCE_ID" \
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
