#!/usr/bin/env bash

llama.cpp/build/bin/llama-gguf-split \
    --merge \
    "/mnt/data/gguf/deepseek-r1-q4_k_m/DeepSeek-R1-Q4_K_M-00001-of-00009.gguf" \
    ./deepseek-r1-q4_k_m.gguf