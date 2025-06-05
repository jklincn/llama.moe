#!/usr/bin/env bash

dir=$(cd $(dirname "$0") && pwd)
split="$dir/../llama.cpp/build/bin/llama-gguf-split"

$split --merge \
    /mnt/data/gguf/deepseek-r1-q4_k_m/DeepSeek-R1-Q4_K_M-00001-of-00009.gguf \
    /mnt/data/gguf/deepseek-r1-q4_k_m/DeepSeek-R1-Q4_K_M.gguf