#!/bin/bash
# This script installs the necessary dependencies for the NL2UniProt project.

use_flash_attn=$1
pip install -e .

# You mau need to change the cuda version to match your system
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

if [ $use_flash_attn == "True" ]; then
    pip install flash-attn --no-build-isolation
    pip install faesm[flash_attn]
else
    echo "Not installing flash-attn and faesm[flash_attn]"
fi

pre-commit install
