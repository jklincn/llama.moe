#!/usr/bin/env bash

set -euo pipefail

# models
# - /mnt/data/gguf/Qwen3-30B-A3B-Q8_0.gguf
# - /mnt/data/gguf/GLM-4.5-Air-Q8_0.gguf
# - /mnt/data/gguf/Qwen3-235B-A22B-Q8_0.gguf
# - /mnt/data/gguf/GLM-4.5-Q8_0.gguf
# - /mnt/data/gguf/MiniMax-M2-Q4_K_M.gguf
# - /mnt/data/gguf/Qwen3-Next-80B-A3B-Instruct-Q8_0-pruned_cov95.gguf

export LLAMA_MOE_DEBUG=0

model="/mnt/data/gguf/GLM-4.5-Air-Q4_K_M-pruned_code_cov90.gguf"

llama-moe --model $model \
    --seed 0 \
    --ctx-size 4096 \
    --api-key sk-1234 \
    --no-log-file
