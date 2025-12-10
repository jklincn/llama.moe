import argparse
import csv
import numpy as np
import os
from typing import Dict, List
from gguf import GGUFReader, GGUFWriter, GGUFValueType, Keys

from .utils.gguf import detect_arch


def load_expert_importance(csv_path: str, keep_n: int) -> Dict[int, List[int]]:
    """
    返回: {layer_idx: [expert_idx_1, expert_idx_2, ...]} 
    列表中的顺序即为新的物理顺序（按重要性从高到低）
    """
    print(f"Loading activation report from {csv_path}...")
    layer_importance = {}
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            layer_idx = int(row['layer_index'])
            # 提取该层所有专家的激活次数
            counts = []
            for k, v in row.items():
                if k.startswith('expert_'):
                    exp_id = int(k.split('_')[1])
                    counts.append((exp_id, int(v)))
            
            # 按激活次数降序排序
            counts.sort(key=lambda x: x[1], reverse=True)
            
            # 选取前 N 个专家的索引
            keep_indices = [x[0] for x in counts[:keep_n]]
            layer_importance[layer_idx] = keep_indices
            
            print(f"  Layer {layer_idx}: keeping experts {keep_indices}")
            
    return layer_importance


def get_layer_idx_from_name(name: str) -> int:
    # 假设命名格式为 blk.N.xxx
    parts = name.split('.')
    for p in parts:
        if p.isdigit():
            return int(p)
    return -1


def reorder_and_prune_data(data: np.ndarray, indices: List[int], old_count: int, is_quantized: bool) -> np.ndarray:
    """
    根据 indices 列表重排并裁剪数据。
    indices: [2, 0, 5] 表示新数据的第0个块来自旧数据的第2个块...
    """
    # Case 1: Numpy array first dimension matches expert count (Most common for GGUF)
    if data.shape[0] == old_count:
        return data[indices]

    # Case 2: Quantized data, flat or mismatched shape -> Byte slicing
    if is_quantized:
        total_bytes = data.nbytes
        bytes_per_expert = total_bytes // old_count
        
        # Flatten data to bytes for block copying
        data_flat = data.view(np.uint8).flatten()
        new_count = len(indices)
        
        # 预分配新数组
        new_data = np.zeros(new_count * bytes_per_expert, dtype=np.uint8)
        
        for i, old_idx in enumerate(indices):
            start = old_idx * bytes_per_expert
            end = start + bytes_per_expert
            new_data[i * bytes_per_expert : (i + 1) * bytes_per_expert] = data_flat[start:end]
            
        return new_data

    # Case 3: F32 data, other dimensions? (Rare)
    if len(data.shape) > 1 and data.shape[1] == old_count:
        return data[:, indices]
            
    return data


def copy_all_metadata(reader: GGUFReader, writer: GGUFWriter, new_expert_count: int = None) -> None:
    # 基本与 gguf_new_metadata.py 的 copy 逻辑一致：跳过 GGUF.* 与 ARCHITECTURE 虚拟键
    arch = detect_arch(reader)
    for field in reader.fields.values():
        if field.name == Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        
        val = field.contents()
        val_type = field.types[0] if field.types else None
        sub_type = field.types[-1] if field.types and field.types[0] == GGUFValueType.ARRAY else None

        if new_expert_count is not None and field.name == f"{arch}.expert_count":
            print(f"  [Metadata] Updating {field.name}: {val} -> {new_expert_count}")
            val = new_expert_count
        
        # 确保 expert_used_count 不超过新的 expert_count
        if new_expert_count is not None and field.name == f"{arch}.expert_used_count":
            if int(val) > new_expert_count:
                print(f"  [Metadata] Updating {field.name}: {val} -> {new_expert_count}")
                val = new_expert_count

        if val_type is not None:
            writer.add_key_value(field.name, val, val_type, sub_type=sub_type)
        else:
            raise ValueError(f"无法处理的字段类型: {field.name}")


def copy_tensors(reader: GGUFReader, writer: GGUFWriter, old_expert_count: int, new_expert_count: int, importance_map: Dict[int, List[int]] = None):
    for tensor in reader.tensors:
        data = tensor.data
        shape = list(tensor.shape)
        name = tensor.name
        tensor_type = tensor.tensor_type

        is_expert_tensor = "exps" in name or "ffn_gate_inp" in name or "exp_probs" in name
        layer_idx = get_layer_idx_from_name(name)
        
        # 只有当我们在元数据中找到了专家数量，并且当前张量维度中包含该数量时，才尝试裁剪
        target_dim_idx = -1
        if is_expert_tensor and old_expert_count in shape:
            target_dim_idx = shape.index(old_expert_count)
        
        # 决定保留哪些索引
        keep_indices = None
        if importance_map and layer_idx in importance_map:
            keep_indices = importance_map[layer_idx]
        elif new_expert_count < old_expert_count:
            # 默认保留前 N 个
            keep_indices = list(range(new_expert_count))

        if keep_indices and target_dim_idx != -1:
            print(f"  - [Pruning] {name} | {tensor_type.name} | {shape} -> Keeping {len(keep_indices)} experts (Reordered)")
            
            # 更新逻辑形状
            shape[target_dim_idx] = len(keep_indices)
            
            is_quant = (tensor.data.dtype == np.uint8)
            # 直接调用裁剪函数，它会自动处理维度匹配
            data = reorder_and_prune_data(data, keep_indices, old_expert_count, is_quant)
        else:
            print(f"  - {name} | {tensor_type.name} | {shape}")

        # 写入数据
        if tensor.data.dtype == np.uint8:
            # 注意：不要传递 raw_shape=shape (逻辑形状)，GGUFWriter 期望的是字节形状 (byte shape)
            # 如果不传递 raw_shape，它会自动使用 data.shape，这正是我们想要的（因为 data 已经是字节数据）
            writer.add_tensor(name, data, raw_dtype=tensor_type)
        else:
            writer.add_tensor(name, data)


def prune_model_with_report(gguf_path: str, report_path: str, keep_n: int) -> str:
    print(f"Loading {gguf_path}...")
    reader = GGUFReader(gguf_path)
    arch = detect_arch(reader)
    
    expert_count_field = reader.fields.get(f"{arch}.expert_count")
    old_expert_count = int(expert_count_field.parts[-1][0]) if expert_count_field else 0
    print(f"Detected architecture: {arch}, Expert Count: {old_expert_count}")

    importance_map = load_expert_importance(report_path, keep_n)
    output_path = gguf_path + f".smart_pruned_{keep_n}"
    
    print(f"Output path: {output_path}")
    
    writer = GGUFWriter(output_path, arch)

    print("Copying metadata...")
    copy_all_metadata(reader, writer, keep_n)

    print("Copying tensors...")
    copy_tensors(reader, writer, old_expert_count, keep_n, importance_map)

    print("Writing file...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print("Done.")
    return output_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="GGUF MoE 裁剪脚本")
    ap.add_argument("gguf_path", help="GGUF 文件路径")
    ap.add_argument("--prune-experts", type=int, default=None, help="保留前 N 个专家")
    ap.add_argument("--activation-report", type=str, default=None, help="expert_activations.csv 路径")
    args = ap.parse_args()

    if args.prune_experts and args.activation_report:
        prune_model_with_report(args.gguf_path, args.activation_report, args.prune_experts)
    else:
        # 简单的复制或截断逻辑，为了兼容旧用法，这里可以保留之前的逻辑，或者简化
        print("Please provide --prune-experts and --activation-report for smart pruning.")
