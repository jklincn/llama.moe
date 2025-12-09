import argparse
import numpy as np
from gguf import GGUFReader, GGUFWriter, GGUFValueType, Keys

from .utils.gguf import detect_arch


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


def copy_tensors(reader: GGUFReader, writer: GGUFWriter):
    for tensor in reader.tensors:
        print(f"  - {tensor.name} | {tensor.tensor_type.name} | {tensor.shape}")
        # 如果是 uint8 类型（通常是量化数据），需要特殊处理以避免 GGUFWriter 报错或错误转换形状
        if tensor.data.dtype == np.uint8:
            writer.add_tensor(tensor.name, tensor.data, raw_dtype=tensor.tensor_type)
        else:
            # 对于非量化数据（如 F32, F16 等），直接写入
            writer.add_tensor(tensor.name, tensor.data)

# python -m llama_moe.prune /mnt/data/gguf/Qwen1.5-MoE-A2.7B-Q8_0.gguf
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="GGUF 文件复制脚本")
    ap.add_argument("gguf_path", help="GGUF 文件路径")
    args = ap.parse_args()

    print(f"Loading {args.gguf_path}...")
    reader = GGUFReader(args.gguf_path)
    arch = detect_arch(reader)
    
    output_path = args.gguf_path + ".copy"
    print(f"Output path: {output_path}")
    
    writer = GGUFWriter(output_path, arch)

    print("Copying metadata...")
    copy_all_metadata(reader, writer)

    print("Copying tensors...")
    copy_tensors(reader, writer)

    print("Writing file...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print("Done.")
