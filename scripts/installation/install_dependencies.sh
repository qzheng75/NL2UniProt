#!/bin/bash
# This script installs the necessary dependencies for the NL2UniProt project.

pip install -e .
pip install flash-attn --no-build-isolation
pip install faesm[flash_attn]
pre-commit install
