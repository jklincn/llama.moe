#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "${script_dir}/build_llama_moe.sh" -r
bash "${script_dir}/build_stream.sh"
bash "${script_dir}/build_pcm.sh"
bash "${script_dir}/build_nvbandwidth.sh"