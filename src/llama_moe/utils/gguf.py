import re
from collections import defaultdict
from typing import Any, Mapping, Optional, Tuple

from gguf import GGUFReader, Keys


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


def detect_arch(reader: GGUFReader) -> str:
    field = reader.get_field(Keys.General.ARCHITECTURE)
    if field is None:
        raise KeyError("Missing required key: general.architecture")
    arch = field.contents()
    if not isinstance(arch, str) or not arch:
        raise ValueError(f"Bad general.architecture value: {arch!r}")
    return arch


def get_llm_value(reader: GGUFReader, key: str) -> str:
    arch = detect_arch(reader)
    full_key = f"{arch}.{key}"
    field = reader.get_field(full_key)
    if field is None:
        raise KeyError(f"Missing key: {full_key}")
    return field.contents()


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
    arch = detect_arch(reader)

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
