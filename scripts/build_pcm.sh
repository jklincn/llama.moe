#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="${script_dir}/../tests/utils/pcm"

cd "$repo_root"
rm -rf build
mkdir build
cd build
cmake ..
cmake --build . --parallel --config Release
exit 0


## usage
ls tests/utils/pcm/build/bin/
sudo tests/utils/pcm/build/bin/pcm-memory --help
sudo tests/utils/pcm/build/bin/pcm-memory 0.1 -csv=stream_bandwidth.csv -- numactl --interleave=all tests/utils/stream/stream