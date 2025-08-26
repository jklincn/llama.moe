# llama.moe

A lightweight inference framework for Mixture-of-Experts (MoE) models, built on top of llama.cpp with dynamic expert offloading.

## Get Started

### Installation dependencies

```
sudo apt update
sudo apt install build-essential cmake ccache libcurl4-openssl-dev ninja-build
```

**CUDA is also required.**

### Build

```
git clone --recurse-submodules https://github.com/jklincn/llama.moe.git
cd llama.moe
scripts/build.sh -r

uv sync
uv pip install .
```

### Run

Usage is the same as [llama-server](https://github.com/ggml-org/llama.cpp/tree/master/tools/server). 

Simply replace `llama-server` with `llama-moe`.

## Development Setup

```
uv sync --extra dev
```

