#!/bin/bash
set -e

mkdir -p pruned_models

MODEL_IN="/mnt/data/gguf/Qwen3-30B-A3B-Q8_0.gguf"
ACT_REPORT="/home/lin/bs/llama.moe/tests/results-20260105-141511/Qwen3-30B-A3B-Q8_0/llama.moe/expert_activations.csv"
ACT_WEIGHTS="/home/lin/bs/llama.moe/tests/results-20260105-141511/Qwen3-30B-A3B-Q8_0/llama.moe/expert_weights.csv"

coverages=(95 94 93 92 91 90 85)

for cov in "${coverages[@]}"; do
    echo "Pruning with coverage ${cov}%..."
    uv run python -m llama_moe.pruner "$MODEL_IN" \
        --prune-coverage "$cov" \
        --activation-report "$ACT_REPORT" \
        --activation-weights "$ACT_WEIGHTS" \
        --output "pruned_models/Qwen3-30B-pruned-cov${cov}.gguf"
    echo "----------------------------------------"
done
