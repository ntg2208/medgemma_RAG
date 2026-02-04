#!/bin/bash
#
# Automated setup script for AWS G4 instance
# Run this after first SSH into a fresh Deep Learning AMI
#
# Usage: bash setup-g4-instance.sh <your-hf-token>
#

set -e  # Exit on error

echo "================================================"
echo "MedGemma RAG - AWS G4 Instance Setup"
echo "================================================"

# Check if HF token provided
if [ -z "$1" ]; then
    echo "Error: HuggingFace token required"
    echo "Usage: bash setup-g4-instance.sh <your-hf-token>"
    exit 1
fi

export HF_TOKEN=$1

echo ""
echo "[1/9] Updating system packages..."
sudo apt update
sudo apt upgrade -y

echo ""
echo "[2/9] Installing essential tools..."
sudo apt install -y \
    git \
    curl \
    wget \
    vim \
    htop \
    tmux \
    unzip \
    tree

echo ""
echo "[3/9] Mounting data volume..."
# Check if already mounted
if mountpoint -q /data; then
    echo "Data volume already mounted"
else
    # Check if volume exists
    if lsblk | grep -q nvme1n1; then
        # Check if formatted
        if ! sudo file -s /dev/nvme1n1 | grep -q "ext4"; then
            echo "Formatting data volume (first time only)..."
            sudo mkfs.ext4 /dev/nvme1n1
        fi

        sudo mkdir -p /data
        sudo mount /dev/nvme1n1 /data
        sudo chown -R ubuntu:ubuntu /data

        # Add to fstab if not already there
        if ! grep -q "/dev/nvme1n1" /etc/fstab; then
            echo '/dev/nvme1n1 /data ext4 defaults,nofail 0 2' | sudo tee -a /etc/fstab
        fi

        echo "Data volume mounted at /data"
    else
        echo "Warning: No additional volume found. Using root volume."
        sudo mkdir -p /data
        sudo chown -R ubuntu:ubuntu /data
    fi
fi

echo ""
echo "[4/9] Installing uv (Python package manager)..."
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.cargo/env
    echo 'source $HOME/.cargo/env' >> ~/.bashrc
else
    echo "uv already installed"
fi

echo ""
echo "[5/9] Installing Python 3.12..."
uv python install 3.12

echo ""
echo "[6/9] Cloning MedGemma RAG repository..."
cd /data
if [ -d "medgemma_RAG" ]; then
    echo "Repository already exists, pulling latest changes..."
    cd medgemma_RAG
    git pull
else
    git clone https://github.com/ntg2208/medgemma_RAG.git
    cd medgemma_RAG
fi

echo ""
echo "[7/9] Setting up Python environment..."
uv venv --python 3.12
source .venv/bin/activate

echo "Installing dependencies (this may take 5-10 minutes)..."
uv pip install -r requirements.txt

echo "Installing spaCy model..."
python -m spacy download en_core_web_sm

echo ""
echo "[8/9] Configuring environment..."
# Create .env file
cat > .env <<EOF
# HuggingFace Configuration
HF_TOKEN=$HF_TOKEN

# Model Cache
HF_HOME=/data/models_cache

# LangSmith (Optional - uncomment and add your keys if using)
# LANGSMITH_API_KEY=your_api_key_here
# LANGSMITH_PROJECT=medgemma-rag
# LANGSMITH_TRACING=false

# Logging
LOG_LEVEL=INFO
EOF

# Add HF_HOME to bashrc
if ! grep -q "HF_HOME" ~/.bashrc; then
    echo 'export HF_HOME=/data/models_cache' >> ~/.bashrc
fi

# Create models cache directory
mkdir -p /data/models_cache

echo ""
echo "[9/9] Pre-downloading models (this will take 10-15 minutes)..."
export HF_HOME=/data/models_cache

python <<'PYEOF'
from huggingface_hub import snapshot_download
import os

token = os.getenv('HF_TOKEN')

print('\n[9.1/9] Downloading MedGemma 4B (this is large, ~8GB)...')
try:
    snapshot_download(
        'google/medgemma-4b-it',
        token=token,
        cache_dir='/data/models_cache'
    )
    print('✓ MedGemma 4B downloaded')
except Exception as e:
    print(f'✗ Failed to download MedGemma: {e}')
    print('You may need to request access at: https://huggingface.co/google/medgemma-4b-it')

print('\n[9.2/9] Downloading EmbeddingGemma 300M...')
try:
    snapshot_download(
        'google/embeddinggemma-300m',
        token=token,
        cache_dir='/data/models_cache'
    )
    print('✓ EmbeddingGemma downloaded')
except Exception as e:
    print(f'✗ Failed to download EmbeddingGemma: {e}')
    print('You may need to request access at: https://huggingface.co/google/embeddinggemma-300m')

print('\nModel downloads complete!')
PYEOF

echo ""
echo "================================================"
echo "Setup Complete!"
echo "================================================"
echo ""
echo "Next steps:"
echo "1. Upload PDFs to /data/medgemma_RAG/Data/documents/"
echo "   From your LOCAL machine:"
echo "   scp -i ~/.ssh/medgemma-key.pem Data/documents/*.pdf ubuntu@\$(instance-ip):/data/medgemma_RAG/Data/documents/"
echo ""
echo "2. Process documents:"
echo "   cd /data/medgemma_RAG"
echo "   source .venv/bin/activate"
echo "   uv run Data/export_chunks.py"
echo ""
echo "3. Run the application:"
echo "   uv run app.py"
echo ""
echo "4. Access Gradio UI:"
echo "   http://\$(instance-ip):7860"
echo "   (Make sure port 7860 is open in security group)"
echo ""
echo "Quick test GPU:"
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
" || echo "Run 'source ~/.bashrc && source .venv/bin/activate' first"

echo ""
echo "To resume work later:"
echo "  cd /data/medgemma_RAG"
echo "  source .venv/bin/activate"
echo "  export HF_HOME=/data/models_cache"
echo ""
