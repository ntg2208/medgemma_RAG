#!/bin/bash
# Update SSH config with current EC2 public IP

set -e

# Get instance IP from Terraform
PUBLIC_IP=$(terraform -chdir=infrastructure/terraform output -raw public_ip 2>/dev/null)

if [ -z "$PUBLIC_IP" ] || [ "$PUBLIC_IP" = "null" ]; then
  echo "Error: Could not get public IP from Terraform"
  echo "Make sure the EC2 instance is running:"
  echo "  ./scripts/ec2-start.sh"
  echo "  ./scripts/ec2-status.sh"
  exit 1
fi

SSH_CONFIG="$HOME/.ssh/config"
HOST_NAME="medgemma-gpu"

echo "Current EC2 Public IP: $PUBLIC_IP"
echo ""

# Create SSH config if it doesn't exist
if [ ! -f "$SSH_CONFIG" ]; then
  touch "$SSH_CONFIG"
  chmod 600 "$SSH_CONFIG"
  echo "Created $SSH_CONFIG"
fi

# Check if host already exists
if grep -q "^Host $HOST_NAME$" "$SSH_CONFIG"; then
  echo "Updating existing host '$HOST_NAME' in $SSH_CONFIG"

  # Use sed to update the HostName line after the matching Host line
  # macOS/BSD sed requires backup extension, use empty string for in-place
  if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    sed -i '' "/^Host $HOST_NAME$/,/^Host / s|HostName .*|HostName $PUBLIC_IP|" "$SSH_CONFIG"
  else
    # Linux
    sed -i "/^Host $HOST_NAME$/,/^Host / s|HostName .*|HostName $PUBLIC_IP|" "$SSH_CONFIG"
  fi
else
  echo "Adding new host '$HOST_NAME' to $SSH_CONFIG"

  # Add new host entry
  cat >> "$SSH_CONFIG" << EOF

Host $HOST_NAME
    HostName $PUBLIC_IP
    User ubuntu
    IdentityFile ~/.ssh/medgemma-key.pem
    ServerAliveInterval 60
    ServerAliveCountMax 3
EOF
fi

echo ""
echo "✓ SSH config updated successfully"
echo ""
echo "Test connection:"
echo "  ssh $HOST_NAME"
echo ""
echo "Connect VSCode:"
echo "  1. Cmd+Shift+P → 'Remote-SSH: Connect to Host'"
echo "  2. Select '$HOST_NAME'"
echo "  3. Wait for VSCode Server to install (first time)"
echo "  4. Open folder: /data/medgemma_RAG (or verify actual path)"
