pip install -U modelscope

local_dir=/mnt/gguf

# ==================== Qwen3-30B-A3B-Q8_0 ========================
modelscope download --local_dir $local_dir \
    --model Qwen/Qwen3-30B-A3B-GGUF Qwen3-30B-A3B-Q8_0.gguf
sha256sum $local_dir/Qwen3-30B-A3B-Q8_0.gguf
# 4ad960d180b16f56024f5b704697e5dd5b0837167c2e515ef0569abfc599743c

# ==================== Qwen3-235B-A22B-Q8_0 ========================
modelscope download --local_dir $local_dir \
    --model Qwen/Qwen3-235B-A22B-GGUF --include Q8_0/*

sha256sum $local_dir/Qwen3-235B-A22B-Q8_0-00001-of-00009.gguf