#!/usr/bin/env bash

set -e
set -x

dir=$(cd $(dirname "$0") && pwd)

BUILD_TYPE="Debug"
EXTRA_FLAGS="-g -O0 -fno-omit-frame-pointer -fno-optimize-sibling-calls -fno-inline"

while getopts "r" opt; do
  case $opt in
    r)
      BUILD_TYPE="Release"
      EXTRA_FLAGS=""
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
  esac
done

export CXX=/usr/bin/g++
cd $dir/../llama.cpp
if [ "$BUILD_TYPE" = "Debug" ]; then
  # GGML Debug config
  sed -i "s/#define GGML_DEBUG 0/#define GGML_DEBUG 1/g" ggml/src/ggml-impl.h
  # Debug build
  cmake -B build \
      -DGGML_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES="89" \
      -DCMAKE_BUILD_TYPE=Debug \
      -DCMAKE_CXX_FLAGS="$EXTRA_FLAGS"
  cmake --build build -j 16
  sed -i "s/#define GGML_DEBUG 1/#define GGML_DEBUG 0/g" ggml/src/ggml-impl.h
else
  # Release build
  cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="89" -DCMAKE_BUILD_TYPE=Release
  cmake --build build --config Release -j 16
fi