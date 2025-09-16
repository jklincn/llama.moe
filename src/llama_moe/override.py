import argparse
import logging
import re
from collections import defaultdict
from typing import Any, Mapping, Optional, Tuple

import pynvml
from gguf import GGUFReader
from gguf.constants import GGMLQuantizationType, Keys

logger = logging.getLogger("override")


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


def read_gguf_file(reader: GGUFReader):
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
    header = "{:<36} | {:<15} | {:>12} | {:>10} | {}"
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


layer_re = re.compile(r"^blk\.(\d+)\.")


def layers_bytes(tensors: Mapping[str, int], keyword_pattern: str) -> dict[int, int]:
    """
    统计每一层中，名称匹配 keyword_pattern (正则表达式) 的张量字节数。
    """
    by_layer = defaultdict(int)
    pattern = re.compile(keyword_pattern)
    for name, n_bytes in tensors.items():
        if not pattern.search(name):
            continue
        m = layer_re.search(name)
        if not m:
            continue
        layer_idx = int(m.group(1))
        by_layer[layer_idx] += n_bytes

    return dict(by_layer)


def _calculate_reverse_offload(
    layer_bytes: dict[int, int], available_memory: int
) -> tuple[list[int], int]:
    """
    计算为将组件的后几层放入GPU，需要移动到CPU的前几层的索引列表。

    Args:
        layer_bytes: 组件每层的字节数。
        available_memory: 当前可用的GPU显存。

    Returns:
        A tuple containing:
        - list[int]: 需要移动到CPU的层索引列表。
        - int: 该组件最终在GPU上占用的字节数。
    """
    total_size = sum(layer_bytes.values())

    # 情况1: 整个组件都能放入GPU
    if total_size <= available_memory:
        return [], total_size  # 无需移动到CPU，GPU占用全部大小

    # 情况2: 显存不足，需要从前向后计算哪些层要移到CPU
    cpu_layers = []
    gpu_size = total_size

    # 从第0层开始遍历
    sorted_layers = sorted(layer_bytes.keys())
    for layer_idx in sorted_layers:
        # 预演：将这一层从GPU移走
        gpu_size -= layer_bytes.get(layer_idx, 0)
        cpu_layers.append(layer_idx)

        # 检查剩余部分是否已经可以放入GPU
        if gpu_size <= available_memory:
            return cpu_layers, gpu_size  # 找到了分割点，返回需要移至CPU的层和GPU的占用

    # 情况3: 即使只留最后一层，显存也不够（非常罕见），则全部移至CPU
    return sorted_layers, 0


def get_override_rules(
    reader: GGUFReader, ctx_size: int, kv_offload: bool = True
) -> list[str]:
    """
    根据模型和上下文长度，计算出适合的 override 规则。
    可用显存: free * threshold - KV cache - non-exps
    优先级:
      - KV cache
      - non-exps (output/gate/norm/nextn > attention > dense ffn > shared_exps)
      - exps
    """

    arch = _detect_arch(reader)
    n_layer = _read_required(
        reader, Keys.LLM.BLOCK_COUNT.format(arch=arch), "block_count"
    )
    logger.debug(f"Model architecture: {arch}, layers: {n_layer}")

    threshold = 0.9
    cur = int(free_memory() * threshold)
    logger.debug(f"Available (free * threshold): {fmt_bytes(cur)}")

    # KV Cache
    if kv_offload:
        kv_cache = calculate_kv_cache_size(reader, ctx_size)
        cur -= kv_cache
        logger.debug(
            f"Offloaded KV cache: {fmt_bytes(kv_cache)}, remain: {fmt_bytes(cur)}"
        )
    else:
        logger.debug("no offload KV Cache to GPU")

    tensors = {t.name.lower(): t.n_bytes for t in reader.tensors}
    cpu_rules = []

    # output/gate/norm/nextn(GLM-4.5)
    for keyword in ["output.weight", "gate_inp", "norm", "nextn"]:
        matched_tensors = {
            name: size for name, size in tensors.items() if keyword in name
        }
        total_size = sum(matched_tensors.values())
        if cur - total_size > 0:
            cur -= total_size
            tensors = {
                name: size for name, size in tensors.items() if keyword not in name
            }
            logger.debug(
                f"Offloaded '{keyword}': {fmt_bytes(total_size)}, remain: {fmt_bytes(cur)}"
            )
        else:
            logger.debug(f"Not enough GPU memory for '{keyword}")

    # Attention
    attention_per_layer = layers_bytes(tensors, r"attn")
    if attention_per_layer:
        cpu_attn_layers, attn_gpu_usage = _calculate_reverse_offload(
            attention_per_layer, cur
        )
        cur -= attn_gpu_usage
        if cpu_attn_layers:
            layer_range = "|".join(map(str, cpu_attn_layers))
            rule = f"blk\\.({layer_range})\\..*attn.*=CPU"
            cpu_rules.append(rule)
        logger.debug(
            f"Offloaded {n_layer - len(cpu_attn_layers)}/{n_layer} 'Attention': {fmt_bytes(attn_gpu_usage)}, remain: {fmt_bytes(cur)}"
        )
        tensors = {
            name: size
            for name, size in tensors.items()
            if not (layer_re.search(name) and "attn" in name)
        }

    # Dense ffn
    dense_ffn_per_layer = layers_bytes(tensors, r"ffn_(down|gate|up)\b")
    if dense_ffn_per_layer:
        cpu_ffn_layers, ffn_gpu_usage = _calculate_reverse_offload(
            dense_ffn_per_layer, cur
        )
        cur -= ffn_gpu_usage
        if cpu_ffn_layers:
            layer_range = "|".join(map(str, cpu_ffn_layers))
            rule = f"blk\\.({layer_range})\\..*ffn_(down|gate|up)\\b.*=CPU"
            cpu_rules.append(rule)
        logger.debug(
            f"Offloaded {n_layer - len(cpu_ffn_layers)}/{n_layer} 'Dense FFN': {fmt_bytes(ffn_gpu_usage)}, remain: {fmt_bytes(cur)}"
        )
        tensors = {
            name: size
            for name, size in tensors.items()
            if not (layer_re.search(name) and re.search(r"ffn_(down|gate|up)\b", name))
        }

    # Shared exp
    shared_exps_per_layer = layers_bytes(tensors, r"shexp\b")
    if shared_exps_per_layer:
        cpu_shexp_layers, shexp_gpu_usage = _calculate_reverse_offload(
            shared_exps_per_layer, cur
        )
        cur -= shexp_gpu_usage
        if cpu_shexp_layers:
            layer_range = "|".join(map(str, cpu_shexp_layers))
            rule = f"blk\\.({layer_range})\\..*shexp\\b.*=CPU"
            cpu_rules.append(rule)
        logger.debug(
            f"Offloaded {n_layer - len(cpu_shexp_layers)}/{n_layer} 'Shared EXP': {fmt_bytes(shexp_gpu_usage)}, remain: {fmt_bytes(cur)}"
        )
        tensors = {
            name: size
            for name, size in tensors.items()
            if not (layer_re.search(name) and "shexp" in name)
        }

    # exp
    exp_per_layer = layers_bytes(tensors, r"exp[s_.]")
    if exp_per_layer:
        cpu_exp_layers, exp_gpu_usage = _calculate_reverse_offload(exp_per_layer, cur)
        cur -= exp_gpu_usage
        if cpu_exp_layers:
            layer_range = "|".join(map(str, cpu_exp_layers))
            rule = f"blk\\.({layer_range})\\.ffn_(?:up|down|gate)_exps.*=CPU"
            cpu_rules.append(rule)
        logger.debug(
            f"Offloaded {n_layer - len(cpu_exp_layers)}/{n_layer} 'EXP': {fmt_bytes(exp_gpu_usage)}, remain: {fmt_bytes(cur)}"
        )
        tensors = {
            name: size
            for name, size in tensors.items()
            if not (layer_re.search(name) and re.search(r"exp[s_.]", name))
        }

    # 固定在 CPU 上的张量就不用写规则了
    cpu_fixed_tensors = ["token_embd.weight"]
    tensors = {
        name: size for name, size in tensors.items() if name not in cpu_fixed_tensors
    }

    # 处理规则之外的张量
    if tensors:
        escaped_names = [re.escape(name) for name in tensors.keys()]
        rule = f"^({'|'.join(escaped_names)})$=CPU"
        cpu_rules.append(rule)
        logger.debug("No Rule Matched:")
        for name, size in tensors.items():
            logger.debug(f"  - {name:40} {fmt_bytes(size)}")

    if not cpu_rules:
        logger.debug("All tensors fit in GPU, no override needed")
        return ["-ngl", "999"]

    final_rules = ",".join(cpu_rules)

    final = ["-ngl", "999", "-ot", final_rules]
    if not kv_offload:
        final.append("--no-kv-offload")

    logger.debug(f"final override rules: {final}")

    return final


def _get_scalar(reader: GGUFReader, key: str) -> Optional[int]:
    """Return scalar value if present, else None."""
    field = reader.get_field(key)
    if field is None:
        return None
    val = field.contents()
    # contents() returns Python scalars/lists depending on type
    if isinstance(val, (int, float)):
        return int(val)
    # Some builds may return NumPy scalars already converted to Python types via .tolist()
    try:
        return int(val)  # attempt coercion
    except Exception:
        return None


def _detect_arch(reader: GGUFReader) -> str:
    field = reader.get_field(Keys.General.ARCHITECTURE)
    if field is None:
        raise KeyError("Missing required key: general.architecture")
    arch = field.contents()
    if not isinstance(arch, str) or not arch:
        raise ValueError(f"Bad general.architecture value: {arch!r}")
    return arch


def _read_required(reader: GGUFReader, key: str, label: str) -> int:
    val = _get_scalar(reader, key)
    if val is None:
        raise KeyError(f"Missing required key: {key} ({label})")
    return val


def _read_model_kv_params(reader: GGUFReader) -> Tuple[str, int, int, int, int, int]:
    """
    Returns (arch, n_layer, n_ctx, n_kv_head, key_len, val_len)
    Preference: key_length/value_length -> *_MLA -> head_dim
    """
    arch = _detect_arch(reader)

    k_block_count = Keys.LLM.BLOCK_COUNT.format(arch=arch)
    k_ctx_len = Keys.LLM.CONTEXT_LENGTH.format(arch=arch)
    k_head_kv = Keys.Attention.HEAD_COUNT_KV.format(arch=arch)
    k_key_len = Keys.Attention.KEY_LENGTH.format(arch=arch)
    k_val_len = Keys.Attention.VALUE_LENGTH.format(arch=arch)
    k_key_len_mla = Keys.Attention.KEY_LENGTH_MLA.format(arch=arch)
    k_val_len_mla = Keys.Attention.VALUE_LENGTH_MLA.format(arch=arch)
    k_embd = Keys.LLM.EMBEDDING_LENGTH.format(arch=arch)
    k_head_count = Keys.Attention.HEAD_COUNT.format(arch=arch)

    n_layer = _read_required(reader, k_block_count, "block_count")
    n_ctx = _read_required(reader, k_ctx_len, "context_length")
    n_kv_head = _read_required(reader, k_head_kv, "attention.head_count_kv")

    # 1) try explicit key/value_length
    key_len = _get_scalar(reader, k_key_len)
    val_len = _get_scalar(reader, k_val_len)

    # 2) try MLA-specific lengths
    if key_len is None or val_len is None:
        key_len_mla = _get_scalar(reader, k_key_len_mla)
        val_len_mla = _get_scalar(reader, k_val_len_mla)
        if key_len is None and key_len_mla is not None:
            key_len = key_len_mla
        if val_len is None and val_len_mla is not None:
            val_len = val_len_mla

    # 3) fallback to head_dim
    if key_len is None or val_len is None:
        n_embd = _read_required(reader, k_embd, "embedding_length")
        n_head = _read_required(reader, k_head_count, "attention.head_count")
        head_dim = n_embd // n_head
        if key_len is None:
            key_len = head_dim
        if val_len is None:
            val_len = head_dim

    # sanity checks
    if n_layer <= 0 or n_ctx <= 0 or n_kv_head <= 0 or key_len <= 0 or val_len <= 0:
        raise ValueError(
            f"Bad kv params: layer={n_layer}, ctx={n_ctx}, kv_head={n_kv_head}, "
            f"key_len={key_len}, val_len={val_len}"
        )

    return arch, n_layer, n_ctx, n_kv_head, key_len, val_len


def calculate_kv_cache_size(
    reader: GGUFReader,
    ctx_size: int,
    kv_dtype: GGMLQuantizationType = GGMLQuantizationType.F16,
    n_seq_max: int = 1,
) -> int:
    """
    Returns total KV cache bytes for all layers (tensor payload only).
    """
    arch, n_layer, n_ctx, n_kv_head, key_len, val_len = _read_model_kv_params(reader)
    n_ctx = ctx_size
    type_size = 2  # F16

    # per-layer per-sequence
    k_per_layer_per_seq = n_ctx * n_kv_head * key_len * type_size
    v_per_layer_per_seq = n_ctx * n_kv_head * val_len * type_size

    total_k = k_per_layer_per_seq * n_layer * n_seq_max
    total_v = v_per_layer_per_seq * n_layer * n_seq_max
    total = total_k + total_v

    return total


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="GGUF 文件分析脚本")
    ap.add_argument("gguf_path", help="GGUF 文件路径")
    args = ap.parse_args()
    reader = GGUFReader(args.gguf_path)
    read_gguf_file(reader)
    calculate_kv_cache_size(reader, 16384)
