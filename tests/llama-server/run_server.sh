#!/usr/bin/env bash

# model="/mnt/data/gguf/DeepSeek-R1-Q4_K_M.gguf"
model="/mnt/data/gguf/Qwen3-30B-A3B-Q8_0.gguf"
llama-moe --model $model \
    --alias Qwen3-30B-A3B \
    --port 8088 \
    --seed 0 \
    --override-tensor exps=CPU \
    --n-gpu-layers 999 \
    --ctx-size 16384 \
    --threads 8 \
    --metrics \
    --api-key sk-1234 \
    --slots \
    --log-verbosity 0 \
    --cont-batching \
    --parallel 1 \
    --threads-http 4