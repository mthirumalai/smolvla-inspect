#!/usr/bin/env bash
# Clone the repo and run GPU setup.
# Run: chmod +x clone-and-setup.sh && ./clone-and-setup.sh
set -euo pipefail

REPO_URL="https://github.com/ddl-subir-m/smolvla-inspect.git"
DIR="smolvla-inspect"

if [ ! -d "$DIR" ]; then
    echo "Cloning $REPO_URL..."
    git clone "$REPO_URL"
else
    echo "$DIR already exists, skipping clone."
fi

cd "$DIR"
chmod +x setup-gpu.sh
./setup-gpu.sh
