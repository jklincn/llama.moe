#!/usr/bin/env bash

# models
# - /mnt/data/gguf/Qwen3-30B-A3B-Q8_0.gguf
# - /mnt/data/gguf/GLM-4.5-Air-Q8_0.gguf --n-gpu-layers 10
# - /mnt/data/gguf/Qwen3-235B-A22B-Q8_0.gguf
# - /mnt/data/gguf/GLM-4.5-Q8_0.gguf
# - /mnt/data/gguf/MiniMax-M2-Q4_K_M.gguf

model="/mnt/data/gguf/Qwen3-Next-80B-A3B-Instruct-Q8_0.gguf"

dir=$(cd $(dirname "$0") && pwd)
server="$dir/../llama.cpp/build/bin/llama-server"
$server --model $model \
    --seed 0 \
    --ctx-size 4096 \
    --api-key sk-1234 \
    --n-gpu-layers 14