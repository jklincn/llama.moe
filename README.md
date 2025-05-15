# llama.moe

Optimized inference of MoE models based on llama.cpp, with dynamic expert offloading.

# Build

## Dependency

```
sudo apt update
sudo apt install build-essential cmake ccache libcurl4-openssl-dev ninja-build
git clone --recurse-submodules https://github.com/jklincn/llama.moe.git
cd llama.moe
pip install -r requirements.txt
```

## llama.cpp

```
scripts/build.sh -r # remove -r for debug build
```