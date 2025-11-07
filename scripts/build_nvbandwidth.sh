#!/usr/bin/env bash

# sudo apt install build-essential cmake libboost-program-options-dev

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="${script_dir}/../tests/utils/nvbandwidth"

rm -rf build/nvbandwidth
mkdir -p build/nvbandwidth
cd build/nvbandwidth
cmake "$repo_root"
make -j4
exit 0


## usage
sudo build/pcm/bin/pcm-memory 0.1 -csv=pcie_bandwidth.csv -- build/nvbandwidth/nvbandwidth