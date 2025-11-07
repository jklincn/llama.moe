#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="${script_dir}/../tests/utils/stream"

rm -rf build/stream
mkdir -p build/stream
cd build/stream
gcc -O3 -fopenmp -DSTREAM_ARRAY_SIZE=700000000 -mcmodel=medium -DNTIMES=10 "$repo_root"/stream.c -o stream
exit 0

## usage
export OMP_NUM_THREADS=$(nproc)
numactl --interleave=all build/stream/stream