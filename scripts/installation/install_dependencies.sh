#!/bin/bash
# This script installs the necessary dependencies for the NL2UniProt project.

use_flash_attn="True"
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --no_flash) use_flash_attn="False" ;;  # Disable when flag is present
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done
pip install -e .

# You may need to change the cuda version to match your system
pip uninstall torch torchvision torchaudio
pip install torch

# Optional: Install flash-attn and faesm[flash_attn]
if [ $use_flash_attn == "True" ]; then
    pip install flash-attn --no-build-isolation
    pip install faesm[flash_attn]
else
    echo "Not installing flash-attn and faesm[flash_attn]"
fi

pre-commit install

clear
echo "Installation complete! Creating .env file..."
touch .env
echo "MODEL_CACHE=model_cache" >> .env
echo "GOOGLE_APPLICATION_CREDENTIALS=google_cloud/credential.json" >> .env
echo "Complete! You can adjust the .env file as needed."

echo "===================="
echo "Suggested next steps: setup wandb and google cloud credentials, download data and models."
