#!/usr/bin/env bash
# build.sh  —— Debug by default, add -r for Release

set -euo pipefail

# ----------------------------------------------------------------------
# 1. 解析参数
# ----------------------------------------------------------------------
build_type="Release"
extra_flags=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --debug)
            build_type="Debug"
            extra_flags="-g -O0 -fno-omit-frame-pointer -fno-optimize-sibling-calls -fno-inline"
            shift
            ;;
        *)
            echo "Usage: $0 [--debug]" >&2
            exit 1
            ;;
    esac
done

# ----------------------------------------------------------------------
# 2. 路径 & 变量
# ----------------------------------------------------------------------
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="${script_dir}/../llama.cpp"
build_dir="${repo_root}/build"

# ----------------------------------------------------------------------
# 3. 组装共用 CMake 选项
# ----------------------------------------------------------------------
cmake_args=(
    -G Ninja
    -S "${repo_root}"
    -B "${build_dir}"
    -D CMAKE_BUILD_TYPE="${build_type}"
    -D GGML_CUDA=ON
    -D CMAKE_CUDA_ARCHITECTURES=89
    -D LLAMA_BUILD_COMMON=ON
    -D LLAMA_BUILD_TOOLS=ON
    -D LLAMA_BUILD_SERVER=ON
)

[[ -n "${extra_flags}" ]] && cmake_args+=( -D CMAKE_CXX_FLAGS="${extra_flags}" )

# ----------------------------------------------------------------------
# 4. 在 Debug 模式打开 GGML_DEBUG 宏（自动恢复）
# ----------------------------------------------------------------------
pushd "${repo_root}" >/dev/null
rm -rf build
if [[ "${build_type}" == "Debug" ]]; then
    # sed -i 's/#define GGML_DEBUG 0/#define GGML_DEBUG 1/' ggml/src/ggml-impl.h
    :
elif [[ "${build_type}" == "Release" ]]; then
    # sed -i 's/#define GGML_DEBUG 1/#define GGML_DEBUG 0/' ggml/src/ggml-impl.h
    :
fi

# ----------------------------------------------------------------------
# 5. 配置 & 构建
# ----------------------------------------------------------------------
cmake "${cmake_args[@]}"
cmake --build "${build_dir}"

echo -e "✔  Build finished: ${build_type}"
popd >/dev/null
