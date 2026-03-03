#!/usr/bin/env bash
# Setup script for Ubuntu 20.04 + CUDA 12.2 (RTX 3080)
# Run: chmod +x setup-gpu.sh && ./setup-gpu.sh
set -euo pipefail

echo "=== smolvla-inspect GPU setup ==="

# 0. Pull latest code on extended-attribution branch
echo "Switching to extended-attribution branch..."
git fetch origin
git checkout extended-attribution
git pull origin extended-attribution

# 1. Ensure Python >= 3.10
PYTHON=$(command -v python3)
PY_VERSION=$($PYTHON --version 2>&1 | awk '{print $2}')
PY_MINOR=$(echo "$PY_VERSION" | cut -d. -f2)
if [ "$PY_MINOR" -lt 10 ]; then
    echo "Error: Python >= 3.10 required, found $PY_VERSION"
    exit 1
fi
echo "Using Python $PY_VERSION"

# 2. Create virtual environment
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    $PYTHON -m venv .venv
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

# 5. Install Node.js >= 18 (needed for web viewer frontend)
NODE_REQUIRED=18
INSTALL_NODE=false

if command -v node &>/dev/null; then
    NODE_VERSION=$(node --version | sed 's/v//' | cut -d. -f1)
    if [ "$NODE_VERSION" -lt "$NODE_REQUIRED" ]; then
        echo "Node.js v$NODE_VERSION found, but >= $NODE_REQUIRED required."
        INSTALL_NODE=true
    else
        echo "Node.js $(node --version) already installed."
    fi
else
    echo "Node.js not found."
    INSTALL_NODE=true
fi

if [ "$INSTALL_NODE" = true ]; then
    echo "Installing Node.js 20 LTS via NodeSource..."
    curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
    sudo apt-get install -y nodejs
    echo "Node.js $(node --version) installed."
fi

# 6. Verify
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
