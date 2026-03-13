#!/bin/bash
# Launch EC2 spot instance for MedGemma model server
# Usage: ./scripts/start.sh
#
# Reads SG, IAM profile, S3 bucket, region from Terraform outputs.
# Override instance config with env vars:
#   INSTANCE_TYPE=g5.xlarge ./scripts/start.sh

set -e

INSTANCE_NAME="medgemma-model-server"
KEY_FILE="$HOME/.ssh/medgemma-key.pem"
SSH_HOST="medgemma-gpu"
SSH_CONFIG="$HOME/.ssh/config"

# Instance config (override with env vars if needed)
INSTANCE_TYPE="${INSTANCE_TYPE:-g6.xlarge}"
AMI_ID="${AMI_ID:-ami-01bc785757b863550}"   # Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.6.0 (Ubuntu 22.04) us-east-2
KEY_PAIR="${KEY_PAIR:-medgemma-key}"

# Read persistent infrastructure from Terraform
TF_DIR="infrastructure/terraform"

REGION=$(terraform -chdir="$TF_DIR" output -raw region 2>/dev/null)
SG_ID=$(terraform -chdir="$TF_DIR" output -raw security_group_id 2>/dev/null)
INSTANCE_PROFILE=$(terraform -chdir="$TF_DIR" output -raw instance_profile_name 2>/dev/null)
S3_BUCKET=$(terraform -chdir="$TF_DIR" output -raw s3_models_bucket 2>/dev/null)

if [ -z "$SG_ID" ] || [ -z "$INSTANCE_PROFILE" ] || [ -z "$S3_BUCKET" ]; then
  echo "Error: Could not read Terraform outputs."
  echo "Run once: cd infrastructure/terraform && terraform apply"
  exit 1
fi

# Check for already-running instance
EXISTING=$(aws ec2 describe-instances \
  --region "$REGION" \
  --filters "Name=tag:Name,Values=$INSTANCE_NAME" \
            "Name=instance-state-name,Values=running,pending" \
  --query 'Reservations[0].Instances[0].InstanceId' \
  --output text)

if [ -n "$EXISTING" ] && [ "$EXISTING" != "None" ]; then
  echo "Instance already running: $EXISTING"
  echo "Run ./scripts/status.sh for details."
  exit 0
fi

# User data: format + mount 100GB data drive, write env vars
USER_DATA=$(cat <<'USERDATA'
#!/bin/bash
DATA_DEV=$(lsblk -rno NAME,SIZE | grep -E "^(nvme[0-9]+n[0-9]+|xvdf)" | grep -v "^nvme0" | awk '{print "/dev/"$1}' | head -1)
if [ -n "$DATA_DEV" ] && ! blkid "$DATA_DEV" &>/dev/null; then
  mkfs.ext4 -q "$DATA_DEV"
fi
if [ -n "$DATA_DEV" ]; then
  mkdir -p /data
  mount "$DATA_DEV" /data
  echo "$DATA_DEV /data ext4 defaults,nofail 0 2" >> /etc/fstab
  chown ubuntu:ubuntu /data
fi
USERDATA
)
# Append env vars that need interpolation
USER_DATA+="
echo 'export S3_MODELS_BUCKET=$S3_BUCKET' >> /home/ubuntu/.bashrc
echo 'export HF_HOME=/opt/dlami/nvme/models_cache' >> /home/ubuntu/.bashrc
echo 'export DATA_DIR=/data' >> /home/ubuntu/.bashrc
"

echo "Launching $INSTANCE_TYPE spot instance in $REGION..."

INSTANCE_ID=$(aws ec2 run-instances \
  --region "$REGION" \
  --image-id "$AMI_ID" \
  --instance-type "$INSTANCE_TYPE" \
  --key-name "$KEY_PAIR" \
  --security-group-ids "$SG_ID" \
  --iam-instance-profile Name="$INSTANCE_PROFILE" \
  --instance-market-options 'MarketType=spot,SpotOptions={SpotInstanceType=one-time,InstanceInterruptionBehavior=terminate}' \
  --block-device-mappings '[{"DeviceName":"/dev/sdf","Ebs":{"VolumeSize":100,"VolumeType":"gp3","DeleteOnTermination":true}}]' \
  --user-data "$USER_DATA" \
  --tag-specifications \
    "ResourceType=instance,Tags=[{Key=Name,Value=$INSTANCE_NAME},{Key=Project,Value=medgemma-rag},{Key=Type,Value=spot-instance}]" \
    "ResourceType=volume,Tags=[{Key=Name,Value=$INSTANCE_NAME-volume},{Key=Project,Value=medgemma-rag}]" \
  --query 'Instances[0].InstanceId' \
  --output text)

echo "Instance $INSTANCE_ID requested. Waiting for it to be running..."
aws ec2 wait instance-running --region "$REGION" --instance-ids "$INSTANCE_ID"

PUBLIC_IP=$(aws ec2 describe-instances \
  --region "$REGION" \
  --instance-ids "$INSTANCE_ID" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' \
  --output text)

# Update ~/.ssh/config
if [ ! -f "$SSH_CONFIG" ]; then
  touch "$SSH_CONFIG" && chmod 600 "$SSH_CONFIG"
fi

if grep -q "^Host $SSH_HOST$" "$SSH_CONFIG" 2>/dev/null; then
  if [[ "$OSTYPE" == "darwin"* ]]; then
    sed -i '' "/^Host $SSH_HOST$/,/^Host / s|HostName .*|HostName $PUBLIC_IP|" "$SSH_CONFIG"
  else
    sed -i "/^Host $SSH_HOST$/,/^Host / s|HostName .*|HostName $PUBLIC_IP|" "$SSH_CONFIG"
  fi
else
  cat >> "$SSH_CONFIG" <<EOF

Host $SSH_HOST
    HostName $PUBLIC_IP
    User ubuntu
    IdentityFile $KEY_FILE
    ServerAliveInterval 60
    ServerAliveCountMax 3
EOF
fi

echo ""
echo "Instance running!"
echo "  ID:    $INSTANCE_ID"
echo "  IP:    $PUBLIC_IP"
echo "  SSH:   ssh $SSH_HOST"
echo "  vLLM:  http://$PUBLIC_IP:8000"
echo "  TEI:   http://$PUBLIC_IP:8001"
echo ""

# Wait for SSH to be ready, then sync code
echo "Waiting for SSH to be ready..."
for i in $(seq 1 30); do
  if ssh -o ConnectTimeout=3 -o StrictHostKeyChecking=no "$SSH_HOST" "true" 2>/dev/null; then
    break
  fi
  sleep 2
done

echo "Syncing code to instance..."
"$(dirname "$0")/sync.sh" "$PUBLIC_IP"

echo ""
echo "Next: ssh $SSH_HOST  then  bash scripts/startup.sh --start"
