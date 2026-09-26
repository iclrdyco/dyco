#!/usr/bin/env bash
# Run inside a dedicated Python environment with a compatible CUDA-enabled PyTorch.
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python -m pip install -e "${PROJECT_ROOT}/src/open-r1-multimodal"
# The attention implementation requires FlashAttention.
# Installation is opt-in because its CUDA build must match the local environment.
if [[ "${INSTALL_FLASH_ATTN:-0}" == "1" ]]; then
    python -m pip install "flash-attn==2.7.4.post1" --no-build-isolation
fi
