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

# Store token securely (not exported to avoid visibility in process list)
HF_TOKEN="$1"

echo ""
echo "[0/10] Verifying GPU..."
if ! nvidia-smi &> /dev/null; then
    echo "ERROR: No GPU detected. Make sure you're on a GPU instance (g4dn.xlarge)."
    echo "Run 'nvidia-smi' to debug."
    exit 1
fi
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo "✓ GPU detected: $GPU_NAME"

echo ""
echo "[1/10] Updating system packages..."
sudo apt update
sudo apt upgrade -y

echo ""
echo "[2/10] Installing essential tools..."
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
echo "[3/10] Setting up /data directory..."
# Check if /data exists and is accessible
if [ -d "/data" ]; then
    echo "/data directory already exists"
    sudo chown -R ubuntu:ubuntu /data
else
    # Check if instance storage is available and not already mounted
    if lsblk | grep -q nvme1n1 && ! mountpoint -q /opt/dlami/nvme; then
        # Only try to mount if not already mounted elsewhere
        if ! sudo file -s /dev/nvme1n1 | grep -q "filesystem"; then
            echo "Formatting instance storage (first time only)..."
            sudo mkfs.ext4 /dev/nvme1n1
        fi

        sudo mkdir -p /data
        sudo mount /dev/nvme1n1 /data
        sudo chown -R ubuntu:ubuntu /data

        # Add to fstab if not already there
        if ! grep -q "/dev/nvme1n1" /etc/fstab; then
            echo '/dev/nvme1n1 /data ext4 defaults,nofail 0 2' | sudo tee -a /etc/fstab
        fi

        echo "Instance storage mounted at /data"
    else
        echo "Using /data on root volume"
        sudo mkdir -p /data
        sudo chown -R ubuntu:ubuntu /data
    fi
fi

echo ""
echo "[4/10] Installing uv (Python package manager)..."
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Add uv to PATH
    export PATH="$HOME/.local/bin:$PATH"
    if ! grep -q '.local/bin' ~/.bashrc; then
        echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
    fi
else
    echo "uv already installed"
fi

# Ensure uv is in PATH for this session
export PATH="$HOME/.local/bin:$PATH"

echo ""
echo "[5/10] Installing Python 3.12..."
if uv python list 2>/dev/null | grep -q "3.12"; then
    echo "Python 3.12 already installed"
else
    uv python install 3.12
fi

echo ""
echo "[6/10] Cloning MedGemma RAG repository..."
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
echo "[7/10] Setting up Python environment..."
uv venv --python 3.12
source .venv/bin/activate

echo "Installing dependencies (this may take 5-10 minutes)..."
uv pip install -r requirements.txt

echo "Installing vLLM (GPU inference server)..."
uv pip install vllm

echo "Installing Docling (PDF OCR)..."
uv pip install docling

echo "Installing spaCy model..."
uv run python -m spacy download en_core_web_sm

echo ""
echo "[8/10] Pulling TEI Docker image..."
docker pull ghcr.io/huggingface/text-embeddings-inference:latest

echo ""
echo "[9/10] Configuring environment..."
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
echo "[10/11] Pre-downloading models (this will take 10-15 minutes)..."
export HF_HOME=/data/models_cache
export HF_TOKEN="$HF_TOKEN"

python <<'PYEOF'
from huggingface_hub import snapshot_download
import os

token = os.getenv('HF_TOKEN')

print('\n[10.1/11] Downloading MedGemma 1.5 4B (this is large, ~8GB)...')
try:
    snapshot_download(
        'google/medgemma-1.5-4b-it',
        token=token,
        cache_dir='/data/models_cache'
    )
    print('✓ MedGemma 4B downloaded')
except Exception as e:
    print(f'✗ Failed to download MedGemma: {e}')
    print('You may need to request access at: https://huggingface.co/google/medgemma-4b-it')

print('\n[10.2/11] Downloading EmbeddingGemma 300M...')
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
echo "[11/11] Final verification..."
source .venv/bin/activate
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print('✓ GPU setup verified')
else:
    print('✗ WARNING: CUDA not available in Python environment')
"

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
echo "To resume work later:"
echo "  cd /data/medgemma_RAG"
echo "  source .venv/bin/activate"
echo "  export HF_HOME=/data/models_cache"
echo ""
