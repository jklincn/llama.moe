#!/usr/bin/env bash

first_gguf="/mnt/data/gguf/GLM-4.5-Q8_0/GLM-4.5-Q8_0-00001-of-00008.gguf"
output_gguf="/mnt/data/gguf/GLM-4.5-Q8_0.gguf"

dir=$(cd $(dirname "$0") && pwd)
split="$dir/../llama.cpp/build/bin/llama-gguf-split"

$split --merge $first_gguf $output_gguf