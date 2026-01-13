#!/usr/bin/env bash

# ftllm server /mnt/data/safetensors/Qwen3-Next-80B-A3B-Instruct-FP8 \
#     --device cuda \
#     --moe_device "{'cuda':13,'cpu':35}"

# modelscope download --model zai-org/GLM-4.5-Air --exclude "*safetensors*"
# ftllm server /mnt/data/gguf/GLM-4.5-Air-Q4_K_M.gguf \
#     --ori /home/lin/.cache/modelscope/hub/models/zai-org/GLM-4.5-Air \
#     --device cuda \
#     --moe_device "{'cuda':11,'cpu':35}"

# modelscope download --model Qwen/Qwen3-235B-A22B --exclude "*safetensors*"
# ftllm server /mnt/data/gguf/Qwen3-235B-A22B-Q4_K_M.gguf \
#     --ori /home/lin/.cache/modelscope/hub/models/Qwen/Qwen3-235B-A22B \
#     --device cuda \
#     --moe_device "{'cuda':10,'cpu':84}"
