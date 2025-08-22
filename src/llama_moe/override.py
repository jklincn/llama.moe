import argparse
import re
from collections import defaultdict
from typing import Any

import pynvml
from gguf import GGUFReader

LAYER_RE = re.compile(r"^blk\.(\d+)\.")
threshold = 0.9


def fmt_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    s = float(n)
    for u in units:
        if s < 1024 or u == units[-1]:
            return f"{int(s)} {u}" if u == "B" else f"{s:.2f} {u}"
        s /= 1024.0


def pretty(val: Any, max_len: int = 200) -> str:
    s = val if isinstance(val, str) else repr(val)
    return s if len(s) <= max_len else s[:max_len] + f"... (len={len(s)})"


def read_gguf_file(gguf_file_path: str):
    reader = GGUFReader(gguf_file_path)

    # ---------- 打印 KV ----------
    print("Key-Value Pairs:")
    keys = list(reader.fields.keys())
    max_key_length = max((len(k) for k in keys), default=0)

    for key in keys:
        field = reader.fields[key]
        value = field.contents()
        print(f"{key:{max_key_length}} : {pretty(value)}")

    print("----")

    # ---------- 打印 Tensors ----------
    print("Tensors:")
    header = "{:<30} | {:<12} | {:>12} | {:>10} | {}"
    print(header.format("Tensor Name", "Shape", "Elements", "Size", "Quant"))
    print("-" * 80)

    total_bytes = 0
    exps_bytes = 0
    exps_count = 0

    for tensor in reader.tensors:
        dims = [int(x) for x in tensor.shape.tolist()]
        shape_str = "x".join(map(str, dims))

        n_bytes = tensor.n_bytes
        total_bytes += n_bytes

        if "exps" in tensor.name.lower():
            exps_bytes += n_bytes
            exps_count += 1

        print(
            header.format(
                tensor.name,
                shape_str,
                f"{tensor.n_elements:,}",
                fmt_bytes(n_bytes),
                tensor.tensor_type.name,
            )
        )

    print("-" * 80)
    other_bytes = total_bytes - exps_bytes
    other_count = len(reader.tensors) - exps_count
    print(
        f"Tensor count: {len(reader.tensors)} | Total size: {fmt_bytes(total_bytes)} ({total_bytes:,} bytes)"
    )
    print(
        f"exps tensors: {exps_count} | Total size: {fmt_bytes(exps_bytes)} ({exps_bytes:,} bytes)"
    )
    print(
        f"other tensors: {other_count} | Total size: {fmt_bytes(other_bytes)} ({other_bytes:,} bytes)"
    )


def free_memory() -> int:
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    pynvml.nvmlShutdown()
    return info.free


def kv_cache_size_bytes(
    kv_size: int,  # KV 缓存的 cell 数
    n_kv_head: int,  # KV 头数
    head_dim_k: int,  # K 的每头维度
    head_dim_v: int,  # V 的每头维度（很多模型与 K 相同）
    dtype_bytes_k: int,  # K 的存储字节数
    dtype_bytes_v: int,  # V 的存储字节数
    n_layer: int,  # 层数
) -> int:
    size_k = kv_size * n_kv_head * head_dim_k * dtype_bytes_k * n_layer
    size_v = kv_size * n_kv_head * head_dim_v * dtype_bytes_v * n_layer
    return size_k + size_v


def non_exps_size(reader: GGUFReader) -> int:
    total_bytes = 0
    exps_bytes = 0

    for tensor in reader.tensors:
        n_bytes = tensor.n_bytes
        total_bytes += n_bytes

        if "exps" in tensor.name.lower():
            exps_bytes += n_bytes

    other_bytes = total_bytes - exps_bytes
    return other_bytes


def layers_exps_bytes(reader: GGUFReader) -> dict[int, int]:
    """统计每一层的 experts 权重总字节数。"""
    by_layer = defaultdict(int)
    for t in reader.tensors:
        name = t.name
        if "exps" not in name.lower():
            continue
        m = LAYER_RE.search(name)
        if not m:
            continue
        layer_idx = int(m.group(1))
        by_layer[layer_idx] += t.n_bytes
    return dict(by_layer)


def plan_exps_offload(
    model_path: str,
    ctx_size: int,
    *,
    n_layer_hint: int | None = None,  # 若已知总层数可传，否则按有 exps 的层计算
    threshold: float = 0.9,  # 预留比例，避免把显存用满
) -> dict:
    reader = GGUFReader(model_path)

    # 可用显存：free * 阈值 - KV cache - 非 exps
    free = int(free_memory() * threshold)
    kv_cache = kv_cache_size_bytes(
        kv_size=ctx_size,
        n_kv_head=4,
        head_dim_k=128,
        head_dim_v=128,
        dtype_bytes_k=2,
        dtype_bytes_v=2,
        n_layer=48,  # 若可从模型元信息读到，建议用真实值
    )
    available = free - kv_cache - non_exps_size(reader)

    per_layer = layers_exps_bytes(reader)
    if not per_layer:
        return 0

    max_layer = max(per_layer) if n_layer_hint is None else (n_layer_hint - 1)

    remain = available
    N = 0  # 默认全装得下 => N=0
    for L in range(max_layer, -1, -1):
        need = per_layer.get(L, 0)
        if remain - need >= 0:
            remain -= need
        else:
            N = L + 1  # 前 N 层在 CPU：0..L
            break

    return max(N, 0)


def get_override_rules(model: str, ctx_size: int) -> list[str]:
    print("正在分析 gguf 文件...")

    offload_layers = plan_exps_offload(model, ctx_size)

    return ["--n-gpu-layers", str(9999)] + ["--n-cpu-moe", str(offload_layers)]


def main():
    ap = argparse.ArgumentParser(description="GGUF 文件分析脚本")
    ap.add_argument("gguf_path", help="GGUF 文件路径")
    args = ap.parse_args()
    read_gguf_file(args.gguf_path)


if __name__ == "__main__":
    main()
