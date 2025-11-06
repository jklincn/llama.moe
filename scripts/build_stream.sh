#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="${script_dir}/../tests/utils/stream"

cd "$repo_root"

gcc -O3 -fopenmp -DSTREAM_ARRAY_SIZE=700000000 -mcmodel=medium -DNTIMES=10 stream.c -o stream

exit 0


## usage
export OMP_NUM_THREADS=$(nproc)

numactl --interleave=all tests/utils/stream/stream