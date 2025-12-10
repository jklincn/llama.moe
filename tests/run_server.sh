#!/usr/bin/env bash

# models
# - /mnt/data/gguf/Qwen3-30B-A3B-Q8_0.gguf
# - /mnt/data/gguf/GLM-4.5-Air-Q8_0.gguf
# - /mnt/data/gguf/Qwen3-235B-A22B-Q8_0.gguf
# - /mnt/data/gguf/GLM-4.5-Q8_0.gguf
model="/mnt/data/gguf/GLM-4.5-Air-Q8_0.gguf.smart_pruned_96"
llama-moe --model $model \
    --seed 0 \
    --ctx-size 4096 \
    --api-key sk-1234

dir=$(cd $(dirname "$0") && pwd)
server="$dir/../llama.cpp/build/bin/llama-server"
# $server --model $model \
#     --seed 0 \
#     --ctx-size 4096 \
#     --api-key sk-1234 \
#     --log-verbosity 0 \
#     --n-gpu-layers 999 \
#     --n-cpu-moe 13