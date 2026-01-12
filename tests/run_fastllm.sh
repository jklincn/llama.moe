#!/usr/bin/env bash

# modelscope download --model zai-org/GLM-4.5-Air --exclude "*safetensors*"
ftllm server /mnt/data/gguf/Qwen3-235B-A22B-Q4_K_M.gguf \
    --ori /home/lin/.cache/modelscope/hub/models/Qwen/Qwen3-235B-A22B \
    --device cuda \
    --moe_device "{'cuda':10,'cpu':84}"