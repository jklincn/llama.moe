#!/usr/bin/env bash

dir=$(cd $(dirname "$0") && pwd)
server="$dir/../../llama.cpp/build/bin/llama-server"
model="/mnt/data/gguf/DeepSeek-R1-Q4_K_M.gguf"

$server --model $model \
    --port 8088 \
    --seed 0 \
    --override-tensor exps=CPU \
    --n-gpu-layers 999 \
    --ctx-size 1024 \
    --metrics \
    --slots