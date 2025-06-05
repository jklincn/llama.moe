#!/usr/bin/env bash
model_path="/mnt/data/gguf/DeepSeek-R1-Q4_K_M.gguf"
prompt='Please help me write a paragraph introducing Beijing.'
n_predict=100
n_gpu_layers=999

if [ ! -f "$model_path" ]; then
    echo "Model $model_path not found."
    exit 1
fi

llama.cpp/build/bin/llama-cli \
    -m $model_path \
    --prompt "$prompt" \
    --seed 0 \
    --n-predict $n_predict \
    --n-gpu-layers $n_gpu_layers \
    --ctx-size 1024 \
    -ot exps=CPU \
    --single-turn
    