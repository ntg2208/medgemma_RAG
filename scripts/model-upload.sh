#!/bin/bash
# Launch a cheap t3.large spot instance to download models from HuggingFace and upload to S3.
# The instance self-terminates when done.
#
# Usage: bash scripts/model-upload.sh
# Requires: .env with HF_TOKEN, terraform output s3_models_bucket

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TF_DIR="$SCRIPT_DIR/infrastructure/terraform"
INSTANCE_TYPE="t3.large"
AMI_ID="${AMI_ID:-ami-04a131f50a7b86648}"  # Ubuntu 22.04 us-east-2
KEY_NAME="medgemma-key"

# Read persistent infrastructure from Terraform
REGION=$(terraform -chdir="$TF_DIR" output -raw region 2>/dev/null)
SG_ID=$(terraform -chdir="$TF_DIR" output -raw security_group_id 2>/dev/null)
INSTANCE_PROFILE=$(terraform -chdir="$TF_DIR" output -raw instance_profile_name 2>/dev/null)
S3_BUCKET=$(terraform -chdir="$TF_DIR" output -raw s3_models_bucket 2>/dev/null)

if [ -z "$REGION" ] || [ -z "$SG_ID" ] || [ -z "$INSTANCE_PROFILE" ] || [ -z "$S3_BUCKET" ]; then
  echo "Error: Could not read Terraform outputs."
  echo "Run once: cd infrastructure/terraform && terraform apply"
  exit 1
fi

# Read HF_TOKEN from .env
HF_TOKEN=$(grep ^HF_TOKEN "$SCRIPT_DIR/.env" 2>/dev/null | cut -d= -f2 | tr -d ' ')
if [ -z "$HF_TOKEN" ]; then
  echo "Error: HF_TOKEN not found in .env"
  exit 1
fi

echo "S3 Bucket:  $S3_BUCKET"
echo "Region:     $REGION"
echo "Instance:   $INSTANCE_TYPE (spot)"
echo ""

# Check if models already in S3
if aws s3 ls "s3://$S3_BUCKET/hub/" --region "$REGION" 2>/dev/null | grep -q "models--google--medgemma"; then
  echo "Models already in S3. Nothing to do."
  echo "To re-upload: aws s3 rm s3://$S3_BUCKET/hub/ --recursive"
  exit 0
fi

# User data: download models and upload to S3, then self-terminate
USER_DATA=$(cat << EOF
#!/bin/bash
exec > /var/log/model-upload.log 2>&1
set -e

echo "=== Model Upload Started ==="
date

# Install dependencies
apt-get update -qq
apt-get install -y python3-pip awscli -qq
pip3 install huggingface_hub -q

# Download models
export HF_TOKEN="$HF_TOKEN"
export HF_HOME="/home/ubuntu/models_cache"
mkdir -p "\$HF_HOME"

python3 - << 'PYEOF'
from huggingface_hub import snapshot_download
import os

token = os.environ['HF_TOKEN']


print('Downloading MedGemma 1.5 4B (~8GB)...')
snapshot_download('google/medgemma-1.5-4b-it', token=token)
print('Downloading EmbeddingGemma 300M (~1GB)...')
snapshot_download('google/embeddinggemma-300m', token=token)
print('Downloads complete.')
PYEOF

# Upload to S3
echo ""
echo "Uploading to s3://$S3_BUCKET/hub/ ..."
aws s3 sync "\$HF_HOME/hub/" "s3://$S3_BUCKET/hub/" --region $REGION --no-progress

echo ""
echo "=== Upload complete ==="
date
du -sh "\$HF_HOME/hub"

# Self-terminate
INSTANCE_ID=\$(TOKEN=\$(curl -s -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 21600") && curl -s -H "X-aws-ec2-metadata-token: \$TOKEN" http://169.254.169.254/latest/meta-data/instance-id)
echo "Terminating instance \$INSTANCE_ID..."
aws ec2 terminate-instances --instance-ids "\$INSTANCE_ID" --region $REGION
EOF
)

# Launch spot instance (20GB EBS for model downloads)
echo "Launching t3.large spot instance..."
INSTANCE_ID=$(aws ec2 run-instances \
  --region "$REGION" \
  --image-id "$AMI_ID" \
  --instance-type "$INSTANCE_TYPE" \
  --key-name "$KEY_NAME" \
  --security-group-ids "$SG_ID" \
  --iam-instance-profile Name="$INSTANCE_PROFILE" \
  --instance-market-options '{"MarketType":"spot","SpotOptions":{"SpotInstanceType":"one-time","InstanceInterruptionBehavior":"terminate"}}' \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":20,"VolumeType":"gp3","DeleteOnTermination":true}}]' \
  --user-data "$USER_DATA" \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=medgemma-model-uploader},{Key=Project,Value=medgemma-rag}]' \
  --query 'Instances[0].InstanceId' \
  --output text)

echo "Instance launched: $INSTANCE_ID"
echo ""
echo "Waiting for instance to be running..."
aws ec2 wait instance-running --instance-ids "$INSTANCE_ID" --region "$REGION"

PUBLIC_IP=$(aws ec2 describe-instances --instance-ids "$INSTANCE_ID" --region "$REGION" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)

echo "Instance running: $PUBLIC_IP"
echo ""
echo "Monitor upload logs (takes ~15-20 min):"
echo "  ssh -i ~/.ssh/medgemma-key.pem ubuntu@$PUBLIC_IP 'tail -f /var/log/model-upload.log'"
echo ""
echo "Instance will self-terminate when done."
