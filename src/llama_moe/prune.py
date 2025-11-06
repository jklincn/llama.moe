import argparse
import re
from gguf import GGUFReader, GGUFWriter, GGUFValueType, Keys

from .utils.common import fmt_bytes, pretty
from .utils.gguf import detect_arch, get_llm_value

EXPS_RE = re.compile(r"^blk\.(\d+)\.(ffn_(?:down|gate|up)_exps)\.weight$")


def copy_all_metadata(reader: GGUFReader, writer: GGUFWriter) -> None:
    # 基本与 gguf_new_metadata.py 的 copy 逻辑一致：跳过 GGUF.* 与 ARCHITECTURE 虚拟键
    for field in reader.fields.values():
        if field.name == Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        # 直接按原类型和值写回（数组保持子类型）
        val_type = field.types[0] if field.types else None
        sub_type = field.types[-1] if field.types and field.types[0] == GGUFValueType.ARRAY else None
        if val_type is not None:
            writer.add_key_value(field.name, field.contents(), val_type, sub_type=sub_type)
        else:
            raise ValueError(f"无法处理的字段类型: {field.name}")

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
