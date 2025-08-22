import argparse
from typing import Any

from gguf import GGUFReader


def fmt_bytes(n: int) -> str:
    """把字节数格式化成 B/KB/MB/GB/TB（十进制，便于和常见文件大小一致）"""
    units = ["B", "KB", "MB", "GB", "TB"]
    s = float(n)
    for u in units:
        if s < 1000 or u == units[-1]:
            return f"{int(s)} {u}" if u == "B" else f"{s:.2f} {u}"
        s /= 1000.0


def pretty(val: Any, max_len: int = 200) -> str:
    """把任意值转成短字符串，过长则截断"""
    s = val if isinstance(val, str) else repr(val)
    return s if len(s) <= max_len else s[:max_len] + f"... (len={len(s)})"


def decode_field_value(field) -> Any:
    """优先使用 ReaderField.contents()；不支持时回退到旧解码方式"""
    # 新接口：自动把 STRING/ARRAY 等转成可读 Python 值（含 UTF-8 解码）
    try:
        return field.contents()
    except Exception:
        pass

    # 回退：按 parts/data 取值并尽量解码
    try:
        import numpy as np  # 仅用于 dtype 判断，脚本运行环境中通常已安装

        idx = field.data[0] if field.data else -1
        part = field.parts[idx] if idx != -1 else field.parts[-1]
        # 字节数组（ASCII/UTF-8）
        if hasattr(part, "dtype") and part.dtype == np.uint8:
            return bytes(part).decode("utf-8", "replace")
        # 标量或数组
        lst = part.tolist()
        if isinstance(lst, list) and len(lst) == 1:
            return lst[0]
        return lst
    except Exception:
        return "<unreadable>"


def read_gguf_file(gguf_file_path: str):
    reader = GGUFReader(gguf_file_path)

    # ---------- 打印 KV ----------
    print("Key-Value Pairs:")
    keys = list(reader.fields.keys())
    max_key_length = max((len(k) for k in keys), default=0)

    for key in keys:
        field = reader.fields[key]
        value = decode_field_value(field)
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
        try:
            dims = [int(x) for x in tensor.shape.tolist()]
        except Exception:
            dims = list(map(int, tensor.shape))
        shape_str = "x".join(map(str, dims))

        nbytes = int(getattr(tensor, "n_bytes", tensor.n_elements))
        total_bytes += nbytes

        # 统计：名称含 exps 的张量
        if "exps" in tensor.name.lower():
            exps_bytes += nbytes
            exps_count += 1

        print(
            header.format(
                tensor.name,
                shape_str,
                f"{tensor.n_elements:,}",
                fmt_bytes(nbytes),
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


def main():
    ap = argparse.ArgumentParser(
        description="GGUF 文件分析脚本"
    )
    ap.add_argument("gguf_path", help="GGUF 文件路径")
    args = ap.parse_args()
    read_gguf_file(args.gguf_path)


if __name__ == "__main__":
    main()
