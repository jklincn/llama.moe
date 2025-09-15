#!/usr/bin/env bash

# model="/mnt/data/gguf/DeepSeek-R1-Q4_K_M.gguf"
model="/mnt/gguf/GLM-4.5-Q8_0/GLM-4.5-Q8_0-00001-of-00008.gguf"
# model="/mnt/gguf/Qwen3-235B-A22B-Q8_0/Qwen3-235B-A22B-Q8_0-00001-of-00009.gguf"
# llama-moe --model $model \
#     --seed 0 \
#     --override-tensor exps=CPU \
#     --n-gpu-layers 999 \
#     --ctx-size 16384 \
#     --api-key sk-1234 \
#     --log-verbosity 0

server="llama.cpp/build/bin/llama-server"
$server --model $model \
    --seed 0 \
    --n-gpu-layers 6 \
    --ctx-size 16384 \
    --api-key sk-1234 \
    --log-verbosity 0