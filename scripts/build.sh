#!/usr/bin/env bash
# build.sh  —— Debug by default, add -r for Release

set -euo pipefail


# ----------------------------------------------------------------------
# 1. 解析参数
# ----------------------------------------------------------------------
build_type="Debug"
extra_flags="-g -O0 -fno-omit-frame-pointer -fno-optimize-sibling-calls -fno-inline"

while getopts ":r" opt; do
  case "$opt" in
    r) build_type="Release"; extra_flags="";;
    *) echo "Usage: $0 [-r]  # -r = Release" >&2; exit 1;;
  esac
done

# ----------------------------------------------------------------------
# 2. 路径 & 变量
# ----------------------------------------------------------------------
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="${script_dir}/../llama.cpp"
build_dir="${repo_root}/build"
threads="$(nproc)"

# ----------------------------------------------------------------------
# 3. 选择生成器：优先 Ninja
# ----------------------------------------------------------------------
generator=""
command -v ninja >/dev/null 2>&1 && generator="-G Ninja"

# ----------------------------------------------------------------------
# 4. 组装共用 CMake 选项
# ----------------------------------------------------------------------
cmake_args=(
  ${generator}
  -S "${repo_root}"
  -B "${build_dir}"
  -D CMAKE_BUILD_TYPE="${build_type}"
  -D GGML_CUDA=ON
  -D CMAKE_CUDA_ARCHITECTURES=89
  -D LLAMA_BUILD_COMMON=ON
  -D LLAMA_BUILD_TOOLS=ON        # llama-cli 位于 tools/
#   -D LLAMA_BUILD_EXAMPLES=OFF
#   -D LLAMA_BUILD_TESTS=OFF
#   -D LLAMA_BUILD_SERVER=OFF
)

[[ -n "${extra_flags}" ]] && cmake_args+=( -D CMAKE_CXX_FLAGS="${extra_flags}" )

# ----------------------------------------------------------------------
# 5. 可选：在 Debug 模式打开 GGML_DEBUG 宏（自动恢复）
# ----------------------------------------------------------------------
pushd "${repo_root}" >/dev/null
rm -rf build
if [[ "${build_type}" == "Debug" ]]; then
  ggml_header="ggml/src/ggml-impl.h"
  cp "${ggml_header}" "${ggml_header}.bak"
  sed -i 's/#define GGML_DEBUG 0/#define GGML_DEBUG 1/' "${ggml_header}"
  trap 'mv "${ggml_header}.bak" "${ggml_header}"' EXIT
fi

# ----------------------------------------------------------------------
# 6. 配置 & 构建（只编译 llama-cli）
# ----------------------------------------------------------------------
cmake "${cmake_args[@]}"
cmake --build "${build_dir}" --target llama-cli -j"${threads}"

echo -e "\n✔  Build finished: ${build_type} (target: llama-cli)"
popd >/dev/null
