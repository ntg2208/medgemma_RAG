# AWS G4 Spot Instance Setup Guide

Complete guide for deploying the MedGemma RAG system on AWS G4 spot instances with persistent EBS storage.

**Estimated Cost**: ~$25/month (4 hours/day, 20 days/month)

---

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Launch Spot Instance](#launch-spot-instance)
3. [Initial Setup](#initial-setup)
4. [Install CUDA and Dependencies](#install-cuda-and-dependencies)
5. [Clone and Setup Project](#clone-and-setup-project)
6. [Daily Workflow](#daily-workflow)
7. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### On Your Local Machine

1. **AWS CLI installed and configured**
   ```bash
   # Install AWS CLI (if not already)
   curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
   unzip awscliv2.zip
   sudo ./aws/install

   # Configure with your credentials
   aws configure
   # Enter: Access Key ID, Secret Access Key, Region (e.g., us-east-1), Output format (json)
   ```

2. **Create EC2 Key Pair** (if you don't have one)
   ```bash
   # Create new key pair
   aws ec2 create-key-pair \
     --key-name medgemma-key \
     --query 'KeyMaterial' \
     --output text > ~/.ssh/medgemma-key.pem

   # Set correct permissions
   chmod 400 ~/.ssh/medgemma-key.pem
   ```

3. **Get your HuggingFace Token**
   - Go to https://huggingface.co/settings/tokens
   - Create a token with read access
   - Request access to `google/medgemma-4b-it` and `google/embeddinggemma-300m`

---

## Launch Spot Instance

### Option 1: Using AWS Console (Beginner-Friendly)

1. **Go to EC2 Dashboard**
   - Open https://console.aws.amazon.com/ec2/
   - Click "Launch Instance"

2. **Configure Instance**

   **Name**: `medgemma-rag-dev`

   **AMI**:
   - Click "Browse more AMIs"
   - Search: "Deep Learning AMI GPU PyTorch"
   - Select: **Deep Learning AMI GPU PyTorch 2.1.0 (Ubuntu 20.04)**

   **Instance Type**:
   - Click "Compare instance types"
   - Filter by "g4dn" family
   - Select: **g4dn.xlarge** (1x T4 GPU, 16GB VRAM, 4 vCPUs, 16GB RAM)

   **Key Pair**:
   - Select your existing key pair (e.g., `medgemma-key`)

   **Network Settings**:
   - Click "Edit"
   - Auto-assign public IP: **Enable**
   - Firewall (Security Group):
     - Create new security group: `medgemma-rag-sg`
     - Add rules:
       - SSH (port 22): Your IP only (for security)
       - Custom TCP (port 7860): Your IP only (for Gradio UI)
       - Custom TCP (port 8888): Your IP only (for Jupyter - optional)

   **Storage (Configure gp3)**:
   - Root volume:
     - Size: **50 GB**
     - Volume type: **gp3**
     - **IMPORTANT**: Uncheck "Delete on termination"
   - Click "Add new volume":
     - Size: **100 GB**
     - Volume type: **gp3**
     - Device: `/dev/sdf`
     - **IMPORTANT**: Uncheck "Delete on termination"

   **Advanced Details**:
   - Scroll to bottom
   - **Purchasing option**: Check "Request Spot Instances"
   - **Maximum price**: Leave blank (use current spot price)
   - **Request type**: Persistent
   - **Interruption behavior**: **Stop** (NOT Terminate!)

3. **Review and Launch**
   - Review summary on right side
   - Click "Launch instance"
   - Wait 2-3 minutes for instance to start

### Option 2: Using AWS CLI (Advanced)

```bash
# Create security group
aws ec2 create-security-group \
  --group-name medgemma-rag-sg \
  --description "Security group for MedGemma RAG development"

# Get security group ID
SG_ID=$(aws ec2 describe-security-groups \
  --group-names medgemma-rag-sg \
  --query 'SecurityGroups[0].GroupId' \
  --output text)

# Add SSH access from your IP
MY_IP=$(curl -s https://checkip.amazonaws.com)
aws ec2 authorize-security-group-ingress \
  --group-id $SG_ID \
  --protocol tcp \
  --port 22 \
  --cidr ${MY_IP}/32

# Add Gradio access
aws ec2 authorize-security-group-ingress \
  --group-id $SG_ID \
  --protocol tcp \
  --port 7860 \
  --cidr ${MY_IP}/32

# Find Deep Learning AMI ID (Ubuntu 20.04, PyTorch)
AMI_ID=$(aws ec2 describe-images \
  --owners amazon \
  --filters "Name=name,Values=Deep Learning AMI GPU PyTorch 2.1.0 (Ubuntu 20.04)*" \
            "Name=state,Values=available" \
  --query 'Images | sort_by(@, &CreationDate) | [-1].ImageId' \
  --output text)

echo "Using AMI: $AMI_ID"

# Create launch template for spot instance
cat > spot-config.json <<EOF
{
  "ImageId": "$AMI_ID",
  "InstanceType": "g4dn.xlarge",
  "KeyName": "medgemma-key",
  "SecurityGroupIds": ["$SG_ID"],
  "BlockDeviceMappings": [
    {
      "DeviceName": "/dev/sda1",
      "Ebs": {
        "VolumeSize": 50,
        "VolumeType": "gp3",
        "DeleteOnTermination": false
      }
    },
    {
      "DeviceName": "/dev/sdf",
      "Ebs": {
        "VolumeSize": 100,
        "VolumeType": "gp3",
        "DeleteOnTermination": false
      }
    }
  ],
  "TagSpecifications": [
    {
      "ResourceType": "instance",
      "Tags": [
        {"Key": "Name", "Value": "medgemma-rag-dev"}
      ]
    }
  ]
}
EOF

# Launch spot instance
aws ec2 request-spot-instances \
  --spot-price "0.526" \
  --instance-count 1 \
  --type "persistent" \
  --instance-interruption-behavior stop \
  --launch-specification file://spot-config.json

# Wait for instance to launch (takes 2-3 minutes)
echo "Waiting for instance to launch..."
sleep 120

# Get instance public IP
INSTANCE_IP=$(aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=medgemma-rag-dev" \
            "Name=instance-state-name,Values=running" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' \
  --output text)

echo "Instance launched at: $INSTANCE_IP"
```

---

## Initial Setup

### 1. Connect to Instance

```bash
# Get your instance public IP
INSTANCE_IP=<your-instance-ip>

# SSH into instance
ssh -i ~/.ssh/medgemma-key.pem ubuntu@$INSTANCE_IP

# Optional: Add to SSH config for easier access
cat >> ~/.ssh/config <<EOF

Host medgemma-g4
    HostName $INSTANCE_IP
    User ubuntu
    IdentityFile ~/.ssh/medgemma-key.pem
    ServerAliveInterval 60
EOF

# Now you can connect with just:
# ssh medgemma-g4
```

### 2. Verify GPU

```bash
# Check NVIDIA driver
nvidia-smi

# Expected output:
# +-----------------------------------------------------------------------------+
# | NVIDIA-SMI 525.xx.xx    Driver Version: 525.xx.xx    CUDA Version: 12.0   |
# |-------------------------------+----------------------+----------------------+
# | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
# | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
# |===============================+======================+======================|
# |   0  Tesla T4            Off  | 00000000:00:1E.0 Off |                    0 |
# | N/A   32C    P0    26W /  70W |      0MiB / 15360MiB |      0%      Default |
# +-------------------------------+----------------------+----------------------+
```

### 3. Mount Data Volume

```bash
# Check if data volume is attached
lsblk

# Expected output shows nvme1n1 (100GB volume)
# NAME        MAJ:MIN RM  SIZE RO TYPE MOUNTPOINT
# nvme0n1     259:0    0   50G  0 disk
# └─nvme0n1p1 259:1    0   50G  0 part /
# nvme1n1     259:2    0  100G  0 disk

# Format data volume (ONLY FIRST TIME!)
sudo mkfs.ext4 /dev/nvme1n1

# Create mount point
sudo mkdir -p /data

# Mount volume
sudo mount /dev/nvme1n1 /data

# Set ownership
sudo chown -R ubuntu:ubuntu /data

# Auto-mount on reboot - add to /etc/fstab
echo '/dev/nvme1n1 /data ext4 defaults,nofail 0 2' | sudo tee -a /etc/fstab

# Verify mount
df -h /data
```

### 4. Update System

```bash
# Update packages
sudo apt update
sudo apt upgrade -y

# Install essential tools
sudo apt install -y \
  git \
  curl \
  wget \
  vim \
  htop \
  tmux \
  unzip \
  tree
```

---

## Install CUDA and Dependencies

The Deep Learning AMI already has CUDA installed, but let's verify and set up Python:

### 1. Verify CUDA

```bash
# Check CUDA version
nvcc --version

# Should show CUDA 12.x
```

### 2. Install UV (Fast Python Package Manager)

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Restart shell or source
source $HOME/.cargo/env

# Verify
uv --version
```

### 3. Install Python 3.12

```bash
# Install Python 3.12 with uv
uv python install 3.12

# Verify
python3.12 --version
```

---

## Clone and Setup Project

### 1. Clone Repository

```bash
# Navigate to data volume
cd /data

# Clone your project
git clone https://github.com/ntg2208/medgemma_RAG.git
cd medgemma_RAG
```

### 2. Setup Python Environment

```bash
# Create virtual environment with uv
uv venv --python 3.12

# Activate environment
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt

# Install spaCy model for PII detection
python -m spacy download en_core_web_sm
```

### 3. Configure Environment

```bash
# Copy .env.example to .env
cp .env.example .env

# Edit .env with your HuggingFace token
vim .env
# or
nano .env

# Add your token:
# HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx
```

### 4. Download Models (First Time Only)

```bash
# Create models cache directory
mkdir -p /data/models_cache
export HF_HOME=/data/models_cache

# Pre-download models to avoid timeout during first run
python -c "
from huggingface_hub import snapshot_download
import os

token = os.getenv('HF_TOKEN')

print('Downloading MedGemma 4B...')
snapshot_download(
    'google/medgemma-4b-it',
    token=token,
    cache_dir='/data/models_cache'
)

print('Downloading EmbeddingGemma...')
snapshot_download(
    'google/embeddinggemma-300m',
    token=token,
    cache_dir='/data/models_cache'
)

print('Models downloaded!')
"

# Add HF_HOME to .bashrc for persistence
echo 'export HF_HOME=/data/models_cache' >> ~/.bashrc
```

### 5. Setup Data

```bash
# Your PDFs should be in Data/documents/
# If you need to upload them from local machine:

# On your LOCAL machine:
scp -i ~/.ssh/medgemma-key.pem \
  Data/documents/*.pdf \
  ubuntu@$INSTANCE_IP:/data/medgemma_RAG/Data/documents/

# Back on the G4 instance:
# Create processed directory
mkdir -p /data/medgemma_RAG/Data/processed

# Process documents
cd /data/medgemma_RAG
uv run Data/export_chunks.py
```

### 6. Test Setup

```bash
# Quick test
uv run python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')
"

# Expected output:
# PyTorch version: 2.x.x
# CUDA available: True
# CUDA device: Tesla T4
```

---

## Daily Workflow

### Starting Work

#### If Instance is Stopped

**Using AWS Console:**
1. Go to EC2 Dashboard
2. Select your instance `medgemma-rag-dev`
3. Click "Instance state" → "Start instance"
4. Wait 1-2 minutes
5. Note the new public IP (changes on each start)

**Using AWS CLI:**
```bash
# Get instance ID
INSTANCE_ID=$(aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=medgemma-rag-dev" \
  --query 'Reservations[0].Instances[0].InstanceId' \
  --output text)

# Start instance
aws ec2 start-instances --instance-ids $INSTANCE_ID

# Wait for running state
aws ec2 wait instance-running --instance-ids $INSTANCE_ID

# Get new public IP
INSTANCE_IP=$(aws ec2 describe-instances \
  --instance-ids $INSTANCE_ID \
  --query 'Reservations[0].Instances[0].PublicIpAddress' \
  --output text)

echo "Instance IP: $INSTANCE_IP"

# Update SSH config with new IP
sed -i.bak "s/HostName .*/HostName $INSTANCE_IP/" ~/.ssh/config
```

#### Connect and Resume Work

```bash
# SSH into instance
ssh medgemma-g4  # or ssh -i ~/.ssh/medgemma-key.pem ubuntu@$INSTANCE_IP

# Navigate to project
cd /data/medgemma_RAG

# Activate environment
source .venv/bin/activate

# Set HuggingFace cache
export HF_HOME=/data/models_cache

# Start working!
```

### Running the Application

#### Option 1: Direct Run (Testing)

```bash
# Run Gradio app
uv run app.py

# Access from local browser (if port 7860 is open in security group):
# http://<instance-ip>:7860
```

#### Option 2: SSH Tunnel (More Secure)

```bash
# On your LOCAL machine:
ssh -i ~/.ssh/medgemma-key.pem -L 7860:localhost:7860 ubuntu@$INSTANCE_IP

# On the G4 instance (in SSH session):
cd /data/medgemma_RAG
source .venv/bin/activate
uv run app.py

# Access from local browser:
# http://localhost:7860
```

#### Option 3: Using tmux (Persistent Sessions)

```bash
# Start tmux session
tmux new -s medgemma

# Run your app
cd /data/medgemma_RAG
source .venv/bin/activate
uv run app.py

# Detach from tmux: Press Ctrl+B, then D

# Later, reattach to session
tmux attach -t medgemma

# List sessions
tmux ls

# Kill session when done
tmux kill-session -t medgemma
```

### Stopping Work

#### Save Your Work

```bash
# Commit any code changes
git add .
git commit -m "Your changes"
git push

# Or sync files back to local machine
# On LOCAL machine:
scp -i ~/.ssh/medgemma-key.pem -r \
  ubuntu@$INSTANCE_IP:/data/medgemma_RAG/Data/processed \
  ./Data/
```

#### Stop Instance

**Using AWS Console:**
1. EC2 Dashboard → Instances
2. Select `medgemma-rag-dev`
3. Instance state → Stop instance

**Using AWS CLI:**
```bash
aws ec2 stop-instances --instance-ids $INSTANCE_ID
```

**IMPORTANT**:
- EBS volumes persist (your data is safe!)
- You only pay for EBS storage (~$12/month)
- No compute charges when stopped

---

## VS Code Remote SSH Setup

### 1. Install VS Code Extension

On your local machine:
1. Open VS Code
2. Install "Remote - SSH" extension (ms-vscode-remote.remote-ssh)

### 2. Configure SSH

```bash
# If you haven't already, add to ~/.ssh/config:
Host medgemma-g4
    HostName <current-instance-ip>
    User ubuntu
    IdentityFile ~/.ssh/medgemma-key.pem
    ServerAliveInterval 60
```

### 3. Connect from VS Code

1. Press `F1` or `Cmd+Shift+P`
2. Type: "Remote-SSH: Connect to Host"
3. Select `medgemma-g4`
4. Wait for connection
5. Open folder: `/data/medgemma_RAG`

### 4. VS Code Extensions on Remote

Install these on the remote instance:
- Python
- Pylance
- Jupyter (if using notebooks)

---

## Handling Spot Interruptions

### What Happens During Interruption

1. AWS sends 2-minute warning
2. Instance stops (doesn't terminate because we set `interruption-behavior: stop`)
3. EBS volumes remain attached and intact
4. All data is safe

### Receiving Interruption Warnings

**Set up a monitoring script (optional):**

```bash
# Create warning monitor
cat > ~/check-spot-interruption.sh <<'EOF'
#!/bin/bash
METADATA_URL="http://169.254.169.254/latest/meta-data/spot/instance-action"

while true; do
    RESPONSE=$(curl -s $METADATA_URL)
    if [ -n "$RESPONSE" ]; then
        echo "SPOT INTERRUPTION WARNING!"
        echo "Time: $(date)"
        echo "Action: $RESPONSE"

        # Send notification (optional - requires AWS SNS setup)
        # aws sns publish --topic-arn arn:aws:sns:region:account:topic --message "Spot interruption!"

        # Save work automatically
        cd /data/medgemma_RAG
        git add .
        git commit -m "Auto-save before spot interruption $(date)"

        break
    fi
    sleep 5
done
EOF

chmod +x ~/check-spot-interruption.sh

# Run in background
# nohup ~/check-spot-interruption.sh &
```

### After Interruption

1. Request new spot instance (usually available immediately)
2. EBS volumes automatically reattach
3. Mount data volume if needed: `sudo mount /dev/nvme1n1 /data`
4. Continue working

---

## Cost Management

### Monitor Costs

```bash
# Check current month's costs (requires AWS CLI setup)
aws ce get-cost-and-usage \
  --time-period Start=$(date -u +%Y-%m-01),End=$(date -u +%Y-%m-%d) \
  --granularity MONTHLY \
  --metrics BlendedCost \
  --group-by Type=SERVICE

# Set up billing alerts
# Go to AWS Console → Billing → Budgets
# Create budget: $30/month with alert at 80% ($24)
```

### Optimize Costs

1. **Always stop instance when done** - saves ~$12/day
2. **Use spot instances** - 70% cheaper than on-demand
3. **Delete old snapshots** - if you create EBS snapshots
4. **Use gp3 instead of gp2** - ~20% cheaper for same performance
5. **Monitor idle time** - set up auto-shutdown after inactivity

**Auto-shutdown script (optional):**

```bash
# Auto-shutdown after 30 minutes of idle
cat > ~/auto-shutdown.sh <<'EOF'
#!/bin/bash
IDLE_TIME=1800  # 30 minutes in seconds

while true; do
    IDLE=$(who -s | wc -l)
    if [ $IDLE -eq 0 ]; then
        IDLE_SECONDS=$((IDLE_SECONDS + 60))
        if [ $IDLE_SECONDS -ge $IDLE_TIME ]; then
            echo "No active users for 30 minutes. Shutting down..."
            sudo shutdown -h now
        fi
    else
        IDLE_SECONDS=0
    fi
    sleep 60
done
EOF

chmod +x ~/auto-shutdown.sh
# Run in background: nohup ~/auto-shutdown.sh &
```

---

## Troubleshooting

### Instance Won't Start

```bash
# Check instance status
aws ec2 describe-instance-status --instance-ids $INSTANCE_ID

# Check spot request status
aws ec2 describe-spot-instance-requests \
  --filters "Name=instance-id,Values=$INSTANCE_ID"

# If spot capacity unavailable, try different region or instance type
```

### Can't Connect via SSH

```bash
# Check security group allows your IP
MY_IP=$(curl -s https://checkip.amazonaws.com)
echo "Your IP: $MY_IP"

# Update security group if IP changed
aws ec2 authorize-security-group-ingress \
  --group-id $SG_ID \
  --protocol tcp \
  --port 22 \
  --cidr ${MY_IP}/32

# Verify instance is running
aws ec2 describe-instances --instance-ids $INSTANCE_ID \
  --query 'Reservations[0].Instances[0].State.Name'
```

### GPU Not Detected

```bash
# Check NVIDIA driver
nvidia-smi

# If not working, reinstall driver
sudo apt install -y nvidia-driver-525

# Reboot
sudo reboot
```

### Out of Disk Space

```bash
# Check disk usage
df -h

# Clean up
# Remove old PyTorch cache
rm -rf ~/.cache/torch

# Clean apt cache
sudo apt clean

# Find large files
sudo du -h /data | sort -h | tail -20
```

### Models Download Slowly

```bash
# Use HuggingFace CLI for better download resumption
pip install huggingface-hub[cli]

huggingface-cli download google/medgemma-4b-it \
  --token $HF_TOKEN \
  --cache-dir /data/models_cache

huggingface-cli download google/embeddinggemma-300m \
  --token $HF_TOKEN \
  --cache-dir /data/models_cache
```

### Data Volume Not Mounting

```bash
# Check if volume is attached
lsblk

# If not attached, attach manually (get volume ID from AWS console)
aws ec2 attach-volume \
  --volume-id vol-xxxxxxxxx \
  --instance-id $INSTANCE_ID \
  --device /dev/sdf

# Wait and mount
sleep 10
sudo mount /dev/nvme1n1 /data
```

---

## Quick Reference

### Essential Commands

```bash
# Start instance
aws ec2 start-instances --instance-ids $INSTANCE_ID

# Stop instance
aws ec2 stop-instances --instance-ids $INSTANCE_ID

# Get instance IP
aws ec2 describe-instances --instance-ids $INSTANCE_ID \
  --query 'Reservations[0].Instances[0].PublicIpAddress' --output text

# SSH connect
ssh -i ~/.ssh/medgemma-key.pem ubuntu@$INSTANCE_IP

# Mount data volume (if needed)
sudo mount /dev/nvme1n1 /data

# Navigate and activate
cd /data/medgemma_RAG
source .venv/bin/activate
export HF_HOME=/data/models_cache

# Run app
uv run app.py
```

### First-Time Setup Checklist

- [ ] Create AWS key pair
- [ ] Launch g4dn.xlarge spot instance
- [ ] Configure security group (SSH, port 7860)
- [ ] Attach 100GB EBS data volume
- [ ] Set "Delete on termination" = No
- [ ] SSH into instance
- [ ] Mount data volume
- [ ] Install uv and Python 3.12
- [ ] Clone repository
- [ ] Create virtual environment
- [ ] Configure .env with HF_TOKEN
- [ ] Download models
- [ ] Process documents
- [ ] Test GPU access
- [ ] Run application

### Daily Use Checklist

- [ ] Start instance (AWS console or CLI)
- [ ] Get new public IP
- [ ] SSH connect
- [ ] Navigate to `/data/medgemma_RAG`
- [ ] Activate venv: `source .venv/bin/activate`
- [ ] Set cache: `export HF_HOME=/data/models_cache`
- [ ] Work on project
- [ ] Commit changes to git
- [ ] Stop instance when done

---

## Next Steps

1. **Set up instance** following this guide
2. **Test basic workflow** (start, connect, run app, stop)
3. **Configure VS Code Remote** for better development experience
4. **Set up billing alerts** to avoid surprise costs
5. **Create EBS snapshots** periodically for backup

For questions or issues, refer to:
- AWS EC2 Spot Instances: https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/using-spot-instances.html
- AWS EBS Volumes: https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ebs-volumes.html
