#!/usr/bin/env bash
model_path="/mnt/data/gguf/Qwen3-30B-A3B-Q8_0.gguf"
prompt='Please help me write a paragraph introducing Beijing.'
n_predict=2
n_gpu_layers=4

if [ ! -f "$model_path" ]; then
    echo "Model $model_path not found."
    exit 1
fi

llama.cpp/build/bin/llama-cli \
    -m $model_path \
    --prompt "$prompt" \
    --n-predict $n_predict \
    --n-gpu-layers $n_gpu_layers \
    --single-turn --color