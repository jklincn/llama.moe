#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="${script_dir}/../tests/utils/pcm"

rm -rf build/pcm
mkdir -p build/pcm
cd build/pcm
cmake "$repo_root"
cmake --build . --parallel --config Release
exit 0

## usage
sudo build/pcm/bin/pcm-memory --help
sudo build/pcm/bin/pcm-memory 0.1 -csv=stream_bandwidth.csv -- numactl --interleave=all build/stream/stream