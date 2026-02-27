#!/usr/bin/env bash
# Setup script for Ubuntu 20.04 + CUDA 12.2 (RTX 3080)
# Run: chmod +x setup-gpu.sh && ./setup-gpu.sh
set -euo pipefail

echo "=== smolvla-inspect GPU setup ==="

# 1. Install Python 3.11 (Ubuntu 20.04 ships with 3.8)
if ! command -v python3.11 &>/dev/null; then
    echo "Installing Python 3.11 via deadsnakes PPA..."
    sudo apt update
    sudo apt install -y software-properties-common
    sudo add-apt-repository -y ppa:deadsnakes/ppa
    sudo apt update
    sudo apt install -y python3.11 python3.11-venv python3.11-dev
else
    echo "Python 3.11 already installed."
fi

# 2. Create virtual environment
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3.11 -m venv .venv
else
    echo "Virtual environment already exists."
fi

source .venv/bin/activate
pip install --upgrade pip

# 3. Install PyTorch with CUDA 12.1 support (compatible with CUDA 12.2 driver)
echo "Installing PyTorch with CUDA support..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 4. Install project dependencies
echo "Installing project dependencies..."
pip install -r requirements.txt

# 5. Verify
echo ""
echo "=== Verification ==="
python -c "
import torch
print(f'Python:  {__import__(\"sys\").version}')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
"

echo ""
echo "=== Done! ==="
echo "To run: python inspect_attention.py --config configs/gpu.yaml"
