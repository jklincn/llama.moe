#!/usr/bin/env bash

# models
# - /mnt/data/gguf/Qwen3-30B-A3B-Q8_0.gguf
# - /mnt/data/gguf/GLM-4.5-Air-Q8_0.gguf
# - /mnt/data/gguf/Qwen3-235B-A22B-Q8_0.gguf
# - /mnt/data/gguf/GLM-4.5-Q8_0.gguf
# - /mnt/data/gguf/MiniMax-M2-Q4_K_M.gguf

export LLAMA_MOE_DEBUG=0
export LLAMA_MOE_COUNTER=0

model="/mnt/data/gguf/Qwen3-30B-A3B-Q8_0.gguf"

llama-moe --model $model \
    --seed 0 \
    --ctx-size 4096 \
    --api-key sk-1234 \
    --no-log-file
