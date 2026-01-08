import argparse
import csv
from pathlib import Path
import numpy as np
from typing import Dict, List
from gguf import GGUFReader, GGUFWriter, GGUFValueType, Keys

from .utils.gguf import detect_arch


def read_csv_data(csv_path: str) -> Dict[int, Dict[int, float]]:
    """读取 CSV 数据到字典结构 {layer_idx: {expert_idx: value}}"""
    print(f"Loading data from {csv_path}...")
    data = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            layer_idx = int(row["layer_index"])
            layer_data = {}
            for k, v in row.items():
                if k.startswith("expert_"):
                    exp_id = int(k.split("_")[1])
                    layer_data[exp_id] = float(v)
            data[layer_idx] = layer_data
    return data


def normalize_layer_data(layer_data: Dict[int, float]) -> Dict[int, float]:
    """归一化一层的数据，使其和为 1"""
    total = sum(layer_data.values())
    if total == 0:
        return {k: 0.0 for k in layer_data}
    return {k: v / total for k, v in layer_data.items()}


def compute_hybrid_importance(
    counts_path: str, weights_path: str, alpha: float = 0.2
) -> Dict[int, Dict[int, float]]:
    """
    计算混合重要性分数。
    alpha: 计数分数的权重 (0.0 - 1.0)。
           1.0 = 仅使用计数
           0.0 = 仅使用权重
    """
    counts_data = read_csv_data(counts_path)
    weights_data = read_csv_data(weights_path)

    # 混合计算
    hybrid_data = {}
    all_layers = set(counts_data.keys()) | set(weights_data.keys())

    for layer in all_layers:
        c_layer = counts_data.get(layer, {})
        w_layer = weights_data.get(layer, {})

        # 归一化
        c_norm = normalize_layer_data(c_layer)
        w_norm = normalize_layer_data(w_layer)

        all_experts = set(c_norm.keys()) | set(w_norm.keys())
        hybrid_layer = {}

        for exp in all_experts:
            c_val = c_norm.get(exp, 0.0)
            w_val = w_norm.get(exp, 0.0)
            # 混合公式
            score = alpha * c_val + (1.0 - alpha) * w_val
            hybrid_layer[exp] = score

        hybrid_data[layer] = hybrid_layer

    return hybrid_data


def load_expert_importance(
    counts_path: str,
    weights_path: str,
    threshold: float,
    total_experts: int = None,
    alpha: float = 0.2,
) -> Dict[int, List[int]]:
    """
    加载并计算专家重要性，返回保留的专家索引列表。
    """
    importance_data = compute_hybrid_importance(counts_path, weights_path, alpha)

    layer_importance = {}

    for layer_idx, experts_map in importance_data.items():
        # 转换为列表并排序
        counts = list(experts_map.items())

        # 如果指定了总专家数，检查数量 (仅当数据完整时)
        if total_experts is not None and len(counts) < total_experts:
            # 可能是某些专家从未被激活，补齐 0
            existing_ids = set(x[0] for x in counts)
            for i in range(total_experts):
                if i not in existing_ids:
                    counts.append((i, 0.0))

        # 按分数降序排序
        counts.sort(key=lambda x: x[1], reverse=True)

        total_score = sum(x[1] for x in counts)

        keep_indices = []

        # Coverage 逻辑
        target_coverage = threshold
        current_coverage = 0.0
        current_sum = 0.0

        if total_score == 0:
            keep_indices = [x[0] for x in counts]
        else:
            for exp_id, score in counts:
                keep_indices.append(exp_id)
                current_sum += score
                current_coverage = (current_sum / total_score) * 100
                if current_coverage >= target_coverage:
                    break

        # 计算最终覆盖率
        kept_score = sum(dict(counts)[idx] for idx in keep_indices)
        coverage = (kept_score / total_score * 100) if total_score > 0 else 0.0

        layer_importance[layer_idx] = keep_indices

        print(
            f"  Layer {layer_idx}: keeping {len(keep_indices)}/{len(counts)} experts. Score Coverage: {coverage:.2f}%"
        )

    return layer_importance


def get_layer_idx_from_name(name: str) -> int:
    # 假设命名格式为 blk.N.xxx
    parts = name.split(".")
    for p in parts:
        if p.isdigit():
            return int(p)
    return -1


def reorder_and_prune_data(
    data: np.ndarray, indices: List[int], old_count: int, is_quantized: bool
) -> np.ndarray:
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
            new_data[i * bytes_per_expert : (i + 1) * bytes_per_expert] = data_flat[
                start:end
            ]

        return new_data

    # Case 3: F32 data, other dimensions? (Rare)
    if len(data.shape) > 1 and data.shape[1] == old_count:
        return data[:, indices]

    return data


def process_metadata(
    reader: GGUFReader, writer: GGUFWriter, new_expert_count: int = None
) -> None:
    # 基本与 gguf_new_metadata.py 的 copy 逻辑一致：跳过 GGUF.* 与 ARCHITECTURE 虚拟键
    arch = detect_arch(reader)

    # Add pruning flag
    writer.add_bool("llama_moe.is_pruned", True)

    for field in reader.fields.values():
        if field.name == Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue

        val = field.contents()
        val_type = field.types[0] if field.types else None
        sub_type = (
            field.types[-1]
            if field.types and field.types[0] == GGUFValueType.ARRAY
            else None
        )

        if new_expert_count is not None and field.name == f"{arch}.expert_count":
            print(f"  [Metadata] Updating {field.name}: {val} -> {new_expert_count}")
            val = new_expert_count

        # 确保 expert_used_count 不超过新的 expert_count
        if new_expert_count is not None and field.name == f"{arch}.expert_used_count":
            if int(val) > new_expert_count:
                print(
                    f"  [Metadata] Updating {field.name}: {val} -> {new_expert_count}"
                )
                val = new_expert_count

        if val_type is not None:
            writer.add_key_value(field.name, val, val_type, sub_type=sub_type)
        else:
            raise ValueError(f"无法处理的字段类型: {field.name}")


def process_tensors(
    reader: GGUFReader,
    writer: GGUFWriter,
    old_expert_count: int,
    importance_map: Dict[int, List[int]] = None,
):
    arch = detect_arch(reader)
    leading_dense_blocks = 0
    if arch == "glm4moe":
        key = f"{arch}.leading_dense_block_count"
        if key in reader.fields:
            leading_dense_blocks = int(reader.fields[key].parts[-1][0])
            print(f"Detected {arch} with {leading_dense_blocks} leading dense blocks.")

    for tensor in reader.tensors:
        data = tensor.data
        shape = list(tensor.shape)
        name = tensor.name
        tensor_type = tensor.tensor_type

        is_expert_tensor = (
            "exps" in name or "ffn_gate_inp" in name or "exp_probs" in name
        )
        layer_idx = get_layer_idx_from_name(name)

        # Special handling for GLM4MoE dense layers
        skip_pruning = False
        if arch == "glm4moe" and layer_idx != -1 and layer_idx < leading_dense_blocks:
            skip_pruning = True

        # 只有当我们在元数据中找到了专家数量，并且当前张量维度中包含该数量时，才尝试裁剪
        target_dim_idx = -1
        if not skip_pruning and is_expert_tensor and old_expert_count in shape:
            target_dim_idx = shape.index(old_expert_count)

        # 决定保留哪些索引
        keep_indices = None
        if importance_map and layer_idx in importance_map:
            keep_indices = importance_map[layer_idx]

        if keep_indices and target_dim_idx != -1:
            old_shape = [int(x) for x in shape]

            # 更新逻辑形状
            shape[target_dim_idx] = len(keep_indices)
            new_shape = [int(x) for x in shape]

            print(
                f"[Pruning] {name:<32} | {tensor_type.name:<4} | {old_shape} -> {new_shape}"
            )

            is_quant = tensor.data.dtype == np.uint8
            # 直接调用裁剪函数，它会自动处理维度匹配
            data = reorder_and_prune_data(
                data, keep_indices, old_expert_count, is_quant
            )

        # 写入数据
        if tensor.data.dtype == np.uint8:
            # 注意：不要传递 raw_shape=shape (逻辑形状)，GGUFWriter 期望的是字节形状 (byte shape)
            # 如果不传递 raw_shape，它会自动使用 data.shape，这正是我们想要的（因为 data 已经是字节数据）
            writer.add_tensor(name, data, raw_dtype=tensor_type)
        else:
            writer.add_tensor(name, data)


def prune_model_with_report(
    gguf_path: str,
    counts_path: str,
    weights_path: str,
    threshold: float,
    alpha: float,
    output_path: str = None,
) -> str:
    print(f"Loading {gguf_path}...")
    reader = GGUFReader(gguf_path)
    arch = detect_arch(reader)

    expert_count_field = reader.fields.get(f"{arch}.expert_count")
    old_expert_count = int(expert_count_field.parts[-1][0]) if expert_count_field else 0

    block_count_field = reader.fields.get(f"{arch}.block_count")
    num_layers = int(block_count_field.parts[-1][0]) if block_count_field else 0

    print(
        f"Detected architecture: {arch}, Expert Count: {old_expert_count}, Layers: {num_layers}"
    )

    importance_map = load_expert_importance(
        counts_path=counts_path,
        weights_path=weights_path,
        threshold=threshold,
        total_experts=old_expert_count,
        alpha=alpha,
    )

    # Calculate stats
    total_original_experts = num_layers * old_expert_count
    total_kept_experts = 0
    for i in range(num_layers):
        if i in importance_map:
            total_kept_experts += len(importance_map[i])
        else:
            total_kept_experts += old_expert_count

    total_pruned_experts = total_original_experts - total_kept_experts

    print(f"Total Original Experts: {total_original_experts}")
    print(f"Total Kept Experts:     {total_kept_experts}")
    print(f"Total Pruned Experts:   {total_pruned_experts}")

    # Determine max experts kept to update metadata (though layers may vary)
    max_kept_experts = 0
    if importance_map:
        max_kept_experts = max(len(indices) for indices in importance_map.values())

    if output_path is None:
        p = Path(gguf_path)
        suffix_val = f"cov{int(threshold)}"
        output_path = str(p.with_name(f"{p.stem}-pruned_{suffix_val}{p.suffix}"))

    writer = GGUFWriter(output_path, arch)

    # Add stats to KV
    writer.add_uint64("llama_moe.pruned_experts_count", total_pruned_experts)
    writer.add_uint64("llama_moe.original_experts_count", total_original_experts)

    print("Processing metadata...")
    process_metadata(reader, writer, max_kept_experts)

    print("Processing tensors...")
    process_tensors(reader, writer, old_expert_count, importance_map)

    print("Writing file...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print("Done.")
    print(f"Output path: {output_path}")
    return output_path


# python -m llama_moe.pruner /mnt/data/gguf/GLM-4.5-Air-Q8_0.gguf --prune-coverage 90 --activation-report expert_activations.csv --activation-weights expert_weights.csv
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="GGUF MoE 裁剪脚本")
    ap.add_argument("gguf_path", help="GGUF 文件路径")
    ap.add_argument(
        "--prune-coverage",
        type=float,
        required=True,
        help="保留专家直到达到覆盖率 (0-100)",
    )

    ap.add_argument(
        "--activation-report",
        type=str,
        required=True,
        help="expert_activations.csv 路径 (计数)",
    )
    ap.add_argument(
        "--activation-weights",
        type=str,
        required=True,
        help="expert_weights.csv 路径 (权重)",
    )
    ap.add_argument(
        "--count-weight",
        type=float,
        default=0.2,
        help="混合评分中计数的权重 (0.0-1.0), 默认 0.2",
    )
    ap.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="输出文件路径",
    )

    args = ap.parse_args()

    prune_model_with_report(
        args.gguf_path,
        args.activation_report,
        args.activation_weights,
        args.prune_coverage,
        args.count_weight,
        args.output,
    )
