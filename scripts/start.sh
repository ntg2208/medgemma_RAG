#!/bin/bash
# Launch EC2 instance for MedGemma model server
# Usage: ./scripts/start.sh [--on-demand] [--inspect]
#
# Flags:
#   --inspect     Launch a cheap g4dn.xlarge (T4) for data inspection (no vLLM)
#   --on-demand   Use on-demand pricing instead of spot
#
# Override with env vars:
#   REGION=ap-northeast-2 ./scripts/start.sh
#   INSTANCE_TYPE=g6.xlarge ./scripts/start.sh --on-demand

set -e

KEY_FILE="$HOME/.ssh/medgemma-key.pem"
SSH_CONFIG="$HOME/.ssh/config"
KEY_PAIR="${KEY_PAIR:-medgemma-key}"

# Parse flags
ON_DEMAND=false
INSPECT=false
for arg in "$@"; do
  case "$arg" in
    --on-demand) ON_DEMAND=true ;;
    --inspect)   INSPECT=true ;;
  esac
done

# Instance config — inspect mode uses cheaper g4dn (T4, no bfloat16)
if [ "$INSPECT" = true ]; then
  INSTANCE_TYPE="${INSTANCE_TYPE:-g4dn.xlarge}"
  INSTANCE_NAME="medgemma-data-inspect"
  SSH_HOST="medgemma-inspect"
  EBS_SIZE=150
else
  INSTANCE_TYPE="${INSTANCE_TYPE:-g6.xlarge}"
  INSTANCE_NAME="medgemma-model-server"
  SSH_HOST="medgemma-gpu"
  EBS_SIZE=150
fi

# ---------------------------------------------------------------------------
# Region + infrastructure resolution
# ---------------------------------------------------------------------------
# If REGION is set via env var, use it (multi-region mode).
# Otherwise read from Terraform (default single-region mode).
TF_DIR="infrastructure/terraform"

if [ -n "$REGION" ]; then
  echo "Using override region: $REGION"
  # Multi-region mode: create/find infrastructure on the fly
  INSTANCE_PROFILE=$(terraform -chdir="$TF_DIR" output -raw instance_profile_name 2>/dev/null || true)

  # --- AMI: find latest Deep Learning AMI in target region ---
  AMI_ID=$(aws ec2 describe-images --region "$REGION" \
    --owners amazon \
    --filters "Name=name,Values=*Deep Learning OSS Nvidia Driver AMI GPU PyTorch*Ubuntu*22.04*" \
    --query 'Images | sort_by(@, &CreationDate)[-1].ImageId' --output text 2>/dev/null)
  if [ -z "$AMI_ID" ] || [ "$AMI_ID" = "None" ]; then
    echo "Error: No Deep Learning AMI found in $REGION"
    exit 1
  fi
  echo "AMI: $AMI_ID"

  # --- Key Pair: import if missing ---
  if ! aws ec2 describe-key-pairs --region "$REGION" --key-names "$KEY_PAIR" &>/dev/null; then
    echo "Importing key pair '$KEY_PAIR' into $REGION..."
    PUB_KEY="${KEY_FILE%.pem}.pub"
    if [ ! -f "$PUB_KEY" ]; then
      # Generate public key from private key
      ssh-keygen -y -f "$KEY_FILE" > "$PUB_KEY"
    fi
    aws ec2 import-key-pair --region "$REGION" \
      --key-name "$KEY_PAIR" \
      --public-key-material fileb://"$PUB_KEY"
    echo "  Key pair imported"
  fi

  # --- Security Group: create if missing ---
  SG_ID=$(aws ec2 describe-security-groups --region "$REGION" \
    --filters "Name=group-name,Values=medgemma-sg" \
    --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null)

  if [ -z "$SG_ID" ] || [ "$SG_ID" = "None" ]; then
    echo "Creating security group in $REGION..."
    SG_ID=$(aws ec2 create-security-group --region "$REGION" \
      --group-name medgemma-sg \
      --description "MedGemma model server" \
      --query 'GroupId' --output text)

    MY_IP=$(curl -s https://checkip.amazonaws.com)
    aws ec2 authorize-security-group-ingress --region "$REGION" \
      --group-id "$SG_ID" --protocol tcp --port 22 --cidr "$MY_IP/32"
    aws ec2 authorize-security-group-ingress --region "$REGION" \
      --group-id "$SG_ID" --protocol tcp --port 8000-8001 --cidr "$MY_IP/32"
    aws ec2 authorize-security-group-ingress --region "$REGION" \
      --group-id "$SG_ID" --protocol tcp --port 7860 --cidr "$MY_IP/32"
    echo "  SG created: $SG_ID (allowed $MY_IP)"
  else
    echo "SG: $SG_ID (existing)"
  fi

else
  # Default: read everything from Terraform
  REGION=$(terraform -chdir="$TF_DIR" output -raw region 2>/dev/null)
  SG_ID=$(terraform -chdir="$TF_DIR" output -raw security_group_id 2>/dev/null)
  INSTANCE_PROFILE=$(terraform -chdir="$TF_DIR" output -raw instance_profile_name 2>/dev/null)
  AMI_ID="${AMI_ID:-ami-0223098712feea80c}"

  if [ -z "$SG_ID" ] || [ -z "$INSTANCE_PROFILE" ]; then
    echo "Error: Could not read Terraform outputs."
    echo "Run once: cd infrastructure/terraform && terraform apply"
    echo "Or use: REGION=ap-northeast-2 ./scripts/start.sh"
    exit 1
  fi
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

# User data: move Docker/containerd to NVMe instance store for space
USER_DATA=$(cat <<'USERDATA'
#!/bin/bash
NVME=/opt/dlami/nvme

# Move Docker and containerd storage to NVMe (~250GB)
systemctl stop docker containerd
mkdir -p $NVME/docker $NVME/containerd
if [ -d /var/lib/docker ] && [ ! -L /var/lib/docker ]; then
  rsync -aP /var/lib/docker/ $NVME/docker/
  rm -rf /var/lib/docker
fi
if [ -d /var/lib/containerd ] && [ ! -L /var/lib/containerd ]; then
  rsync -aP /var/lib/containerd/ $NVME/containerd/
  rm -rf /var/lib/containerd
fi
ln -sf $NVME/docker /var/lib/docker
ln -sf $NVME/containerd /var/lib/containerd
systemctl start containerd docker
USERDATA
)
# Append env vars
USER_DATA+="
echo 'export HF_HOME=/opt/dlami/nvme/models_cache' >> /home/ubuntu/.bashrc
"

# Build launch command
INSTANCE_MODE="spot"
if [ "$ON_DEMAND" = true ]; then
  INSTANCE_MODE="on-demand"
fi

SPOT_FLAG=""
if [ "$ON_DEMAND" != true ]; then
  SPOT_FLAG='--instance-market-options {"MarketType":"spot","SpotOptions":{"SpotInstanceType":"one-time","InstanceInterruptionBehavior":"terminate"}}'
fi

IAM_FLAG=""
if [ -n "$INSTANCE_PROFILE" ] && [ "$INSTANCE_PROFILE" != "None" ]; then
  IAM_FLAG="--iam-instance-profile Name=$INSTANCE_PROFILE"
fi

echo "Launching $INSTANCE_TYPE $INSTANCE_MODE instance in $REGION..."

INSTANCE_ID=$(aws ec2 run-instances \
  --region "$REGION" \
  --image-id "$AMI_ID" \
  --instance-type "$INSTANCE_TYPE" \
  --key-name "$KEY_PAIR" \
  --security-group-ids "$SG_ID" \
  $IAM_FLAG \
  $SPOT_FLAG \
  --block-device-mappings "[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":$EBS_SIZE,\"VolumeType\":\"gp3\",\"DeleteOnTermination\":true}}]" \
  --user-data "$USER_DATA" \
  --tag-specifications \
    "ResourceType=instance,Tags=[{Key=Name,Value=$INSTANCE_NAME},{Key=Project,Value=medgemma-rag},{Key=Type,Value=$INSTANCE_MODE}]" \
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
    LocalForward 7860 localhost:7860
    ServerAliveInterval 60
    ServerAliveCountMax 3
EOF
fi

echo ""
echo "Instance running!"
echo "  ID:     $INSTANCE_ID"
echo "  Type:   $INSTANCE_TYPE ($INSTANCE_MODE)"
echo "  Region: $REGION"
echo "  IP:     $PUBLIC_IP"
echo "  SSH:    ssh $SSH_HOST"

if [ "$INSPECT" = true ]; then
  echo ""
  echo "Data inspection instance (g4dn/T4 — no bfloat16, no vLLM)."
  echo "Next: ssh $SSH_HOST"
else
  echo "  LLM:    http://$PUBLIC_IP:8000"
  echo ""
  echo "Next: ssh $SSH_HOST  then  bash scripts/startup.sh --start"
fi
