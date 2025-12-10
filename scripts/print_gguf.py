import argparse
from typing import Any
from gguf import GGUFReader


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
    header = "{:<36} | {:<14} | {:>12} | {:>10} | {}"
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


def main():
    ap = argparse.ArgumentParser(description="GGUF 文件分析脚本")
    ap.add_argument("gguf_path", help="GGUF 文件路径")
    args = ap.parse_args()
    read_gguf_file(args.gguf_path)


if __name__ == "__main__":
    main()
