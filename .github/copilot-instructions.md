# Llama.moe Copilot Instructions

## Project Overview
`llama.moe` is a lightweight inference framework for Mixture-of-Experts (MoE) models, built as a Python wrapper around `llama.cpp`. It enables dynamic expert offloading and specialized execution strategies.

## Architecture
- **Backend**: Uses `llama.cpp` (as a submodule) for the heavy lifting. The `llama-server` binary is the core execution engine.
- **Frontend**: Python package `llama_moe` (in `src/llama_moe`) wraps the binary, handling process management, argument parsing, and environment configuration (NUMA, caching).
- **Build System**: Custom shell scripts in `scripts/` drive `cmake` builds of the C++ backend. Python dependencies are managed with `uv`.

## Key Components
- `src/llama_moe/wrapper.py`: Manages the `llama-server` subprocess, including I/O redirection and lifecycle.
- `src/llama_moe/core.py`: Orchestrates the execution, handles NUMA configuration, and file cache management.
- `src/llama_moe/cli.py`: Entry point for the `llama-moe` command.
- `scripts/build_llama_moe.sh`: Main build script for the C++ backend. **Note**: Hardcodes `CMAKE_CUDA_ARCHITECTURES=89` (adjust if targeting different hardware).

## Developer Workflows

### Building
1. **C++ Backend**:
   - Debug (default): `scripts/build_llama_moe.sh`
   - Release: `scripts/build_llama_moe.sh -r`
   - The build uses `Ninja` and requires CUDA.

2. **Python Environment**:
   - Sync dependencies: `uv sync`
   - Install package: `uv pip install -e .`
   - **Activation**: ALWAYS activate the virtual environment (`source .venv/bin/activate`) before running any Python commands or scripts.

### Running
- Use the `llama-moe` command (installed via `project.scripts` in `pyproject.toml`).
- Example: `llama-moe -m /path/to/model.gguf -c 4096`
- See `tests/run_server.sh` for a concrete usage example.

### Testing
- Integration tests often involve running the server and checking output.
- Check `tests/` for scripts like `run_server.sh`.

## Conventions & Patterns
- **NUMA Awareness**: The code explicitly checks for NUMA nodes in `core.py` and configures `numactl` accordingly.
- **Path Handling**: Use `pathlib.Path` for file system operations.
- **Logging**: Use the standard `logging` module. `wrapper` and `main` loggers are configured.
- **Submodules**: Always ensure `llama.cpp` submodule is initialized and updated (`git submodule update --init --recursive`).

## Common Issues
- **CUDA Architecture**: If build fails or runs slowly, check `scripts/build_llama_moe.sh` for the `CMAKE_CUDA_ARCHITECTURES` setting.
- **Missing Binary**: `wrapper.py` expects `llama-server` at `llama.cpp/build/bin/llama-server`. Ensure the build script has run successfully.
