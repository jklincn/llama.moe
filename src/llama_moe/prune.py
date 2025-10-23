import argparse
import re
from gguf import GGUFReader, GGUFWriter, GGUFValueType

from .utils.common import fmt_bytes, pretty
from .utils.gguf import detect_arch, get_llm_value

EXPS_RE = re.compile(r"^blk\.(\d+)\.(ffn_(?:down|gate|up)_exps)\.weight$")


def _normalize_scalar(x):
    # 兼容 numpy 标量
    try:
        import numpy as np

        if isinstance(x, np.generic):
            return x.item()
    except Exception:
        pass
    return x


def _kv_from_reader_field(field):
    """
    -> (value, vtype, sub_type)
    - vtype 来自 field.types[0]
    - 如果 vtype 是 ARRAY，则 sub_type = field.types[-1]，否则 None
    - value 用 field.contents() 得到（Python 标量 / 字符串 / list[...]）
    """
    vtypes = getattr(field, "types", []) or []
    if not vtypes:
        return None, None, None

    main_type = vtypes[0]
    sub_type = vtypes[-1] if main_type == GGUFValueType.ARRAY else None

    val = field.contents()

    # 规整一下数值类型，避免 numpy 标量/数组残留
    if isinstance(val, list):
        val = [_normalize_scalar(v) for v in val]
    else:
        val = _normalize_scalar(val)

    return val, main_type, sub_type

def _tensor_raw_info(t):
    """从 reader 的 tensor 对象拿 shape / raw_dtype / nbytes / raw_bytes。"""
    # 形状
    shape = tuple(getattr(t, "shape", None) or t.ne)             # ← 任选其一

    # 原始 ggml dtype（量化类型，如 Q8_0）
    raw_dtype = getattr(t, "dtype", None) or getattr(t, "ggml_type", None)  # ← 任选其一

    # 原始字节大小
    nbytes = getattr(t, "nbytes", None)
    if nbytes is None:
        raw = getattr(t, "raw", None)
        if raw is not None:
            nbytes = len(raw)
        else:
            raw = getattr(t, "data", None)
            nbytes = len(raw) if raw is not None else None

    # 原始字节数据（bytes / memoryview）
    raw_data = getattr(t, "raw", None)
    if raw_data is None:
        raw_data = getattr(t, "data", None)   # 某些实现把“原始块”也挂在 data 上（注意不要是解码后的 numpy）

    if raw_dtype is None or nbytes is None or raw_data is None:
        raise RuntimeError("Cannot obtain raw tensor info; check your GGUFReader tensor attributes")

    return shape, raw_dtype, nbytes, raw_data

def add_tensor_raw(writer: GGUFWriter, name, shape, raw_dtype, nbytes, raw_data):
    """
    走 'info + data' 的原样写回通道：
      - 先声明 tensor 信息（包括 raw_dtype）
      - 再喂入 raw bytes
    这样不会触发 'Only F16/F32/...' 的 dtype 限制。
    """
    writer.add_tensor_info(name, shape, raw_dtype, nbytes, raw_dtype=raw_dtype)
    writer.add_tensor_data(raw_data)

def prune_gguf_file(reader: GGUFReader, writer: GGUFWriter):
    # ---------- 打印 KV ----------
    print("Key-Value Pairs:")
    keys = list(reader.fields.keys())
    max_key_length = max((len(k) for k in keys), default=0)

    for key in keys:
        field = reader.fields[key]
        value = field.contents()
        print(f"{key:{max_key_length}} : {pretty(value)}")

    print("----")

    expert_count = get_llm_value(reader, "expert_count")
    expert_used_count = get_llm_value(reader, "expert_used_count")
    embedding_length = get_llm_value(reader, "embedding_length")
    expert_feed_forward_length = get_llm_value(reader, "expert_feed_forward_length")

    print(f"专家总数: {expert_count}, 使用的专家数: {expert_used_count}")
    print(f"嵌入维度: {embedding_length}")
    print(f"专家前馈长度: {expert_feed_forward_length}")

    # 复制所有 kv
    for key, field in reader.fields.items():
        val, vtype, sub_type = _kv_from_reader_field(field)
        if vtype is None:
            # 跳过空字段（理论上不会发生）
            continue
        writer.add_key_value(key, val, vtype, sub_type)

    for tensor in reader.tensors:
        name = tensor.name
        data = tensor.data
        shape = tensor.shape
        dtype = tensor.tensor_type

        match = EXPS_RE.match(name)
        if not match:
            # 原样复制
            writer.add_tensor(name, data)
            continue
        layer_idx = int(match.group(1))
        part_name = match.group(2)

        match part_name:
            case "ffn_gate_exps" | "ffn_up_exps":
                assert shape == (
                    embedding_length,
                    expert_feed_forward_length,
                    expert_used_count,
                )
                print(f"Modifying {layer_idx} {name} from {shape} -> (768, 2048, 64)")
                arr = data.reshape(768, 2048, 128)
                arr = arr[:, :, :64]
                writer.add_tensor(name, arr, dtype=dtype)
            case "ffn_down_exps":
                assert shape == (
                    expert_feed_forward_length,
                    embedding_length,
                    expert_used_count,
                )
                print(f"Modifying {layer_idx} {name} from {shape} -> (2048, 768, 64)")
                arr = data.reshape(2048, 768, 128)
                arr = arr[:, :, :64]
                writer.add_tensor(name, arr, dtype=dtype)
            case _:
                raise ValueError(f"unknown part name: {part_name}")
    return
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


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="GGUF 文件分析脚本")
    ap.add_argument("gguf_path", help="GGUF 文件路径")
    args = ap.parse_args()
    reader = GGUFReader(args.gguf_path)
    arch = detect_arch(reader)
    writer = GGUFWriter(args.gguf_path + ".pruned.gguf", arch)
    prune_gguf_file(reader, writer)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
