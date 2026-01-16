import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
from gguf import GGUFReader, GGUFValueType, GGUFWriter, Keys

from .utils.gguf import detect_arch


def resolve_report_paths(report_dir: str) -> tuple[str, str]:
    d = Path(report_dir).expanduser().resolve()
    counts_path = d / "expert_activations.csv"
    weights_path = d / "expert_weights.csv"

    if not counts_path.exists():
        raise FileNotFoundError(f"Missing {'expert_activations.csv'} in directory: {d}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing {'expert_weights.csv'} in directory: {d}")

    return str(counts_path), str(weights_path)


def read_csv_data(csv_path: str) -> Dict[int, Dict[int, float]]:
    """读取 CSV 数据到字典结构 {layer_idx: {expert_idx: value}}"""
    print(f"Loading data from {csv_path}...")
    data: Dict[int, Dict[int, float]] = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            layer_idx = int(row["layer_index"])
            layer_data: Dict[int, float] = {}
            for k, v in row.items():
                if k.startswith("expert_"):
                    exp_id = int(k.split("_")[1])
                    layer_data[exp_id] = float(v)
            data[layer_idx] = layer_data
    return data


def normalize_layer_data(layer_data: Dict[int, float]) -> Dict[int, float]:
    """归一化一层的数据，使其和为 1（若全 0 则全 0）"""
    total = sum(layer_data.values())
    if total == 0:
        return {k: 0.0 for k in layer_data}
    return {k: v / total for k, v in layer_data.items()}


def _normalized_entropy(p: Dict[int, float]) -> float:
    """
    归一化熵 H(p)/log(K)，范围 [0, 1]（K=专家数，p 已归一化但这里不强依赖）。
    - 当分布越均匀，熵越高 -> 接近 1
    - 当分布越尖锐，熵越低 -> 接近 0
    稳健处理：全 0 返回 1.0（视为“最不尖锐/最不确定”）
    """
    K = max(len(p), 1)
    vals = [v for v in p.values() if v > 0.0]
    if not vals:
        return 1.0

    H = 0.0
    for v in vals:
        H -= v * math.log(v + 1e-12)

    denom = math.log(K + 1e-12)
    if denom <= 0:
        return 1.0
    return max(0.0, min(1.0, H / denom))


def _sharpness(p: Dict[int, float]) -> float:
    """尖锐度 = 1 - 归一化熵，范围 [0, 1]，越大越尖锐。"""
    return 1.0 - _normalized_entropy(p)


def _auto_alpha_for_layer(c_norm: Dict[int, float], w_norm: Dict[int, float]) -> float:
    """
    每层自适应 alpha：
    alpha = sharpness(counts) / (sharpness(counts) + sharpness(weights) + eps)
    - counts 越尖锐 -> alpha 越大
    - weights 越尖锐 -> alpha 越小
    """
    sc = _sharpness(c_norm)
    sw = _sharpness(w_norm)
    return float(sc / (sc + sw + 1e-12))


def compute_hybrid_importance(
    counts_path: str,
    weights_path: str,
    total_experts: int | None = None,
    verbose_alpha: bool = True,
) -> Dict[int, Dict[int, float]]:
    """
    计算混合重要性分数，每层自适应 alpha
    若 total_experts 给定，会补齐缺失专家为 0，以保证熵/尖锐度估计更稳定。
    """
    counts_data = read_csv_data(counts_path)
    weights_data = read_csv_data(weights_path)

    hybrid_data: Dict[int, Dict[int, float]] = {}
    all_layers = set(counts_data.keys()) | set(weights_data.keys())

    for layer in sorted(all_layers):
        c_layer = dict(counts_data.get(layer, {}))
        w_layer = dict(weights_data.get(layer, {}))

        # 补齐缺失专家（尤其是“从未激活”导致 counts CSV 没有列/值时）
        if total_experts is not None:
            for i in range(total_experts):
                if i not in c_layer:
                    c_layer[i] = 0.0
                if i not in w_layer:
                    w_layer[i] = 0.0

        # 归一化
        c_norm = normalize_layer_data(c_layer) if c_layer else {}
        w_norm = normalize_layer_data(w_layer) if w_layer else {}

        # 统一专家集合
        all_experts = set(c_norm.keys()) | set(w_norm.keys())
        if total_experts is not None:
            all_experts |= set(range(total_experts))

        # 每层自适应 alpha
        alpha_layer = _auto_alpha_for_layer(c_norm, w_norm)

        if verbose_alpha:
            sc = _sharpness(c_norm) if c_norm else 0.0
            sw = _sharpness(w_norm) if w_norm else 0.0
            print(
                f"  Layer {layer}: auto_alpha={alpha_layer:.4f} "
                f"(sharp_counts={sc:.4f}, sharp_weights={sw:.4f})"
            )

        hybrid_layer: Dict[int, float] = {}
        for exp in all_experts:
            c_val = c_norm.get(exp, 0.0)
            w_val = w_norm.get(exp, 0.0)
            score = alpha_layer * c_val + (1.0 - alpha_layer) * w_val
            hybrid_layer[exp] = score

        hybrid_data[layer] = hybrid_layer

    return hybrid_data


def load_expert_importance(
    counts_path: str,
    weights_path: str,
    coverage: float,
    total_experts: int,
) -> Dict[int, List[int]]:
    """
    加载并计算专家重要性，返回保留的专家索引列表。
    """
    importance_data = compute_hybrid_importance(
        counts_path=counts_path,
        weights_path=weights_path,
        total_experts=total_experts,
        verbose_alpha=True,
    )

    layer_importance: Dict[int, List[int]] = {}

    for layer_idx, experts_map in importance_data.items():
        # 转换为列表并排序
        counts = list(experts_map.items())

        # 再保险：补齐缺失专家
        if total_experts is not None and len(counts) < total_experts:
            existing_ids = set(x[0] for x in counts)
            for i in range(total_experts):
                if i not in existing_ids:
                    counts.append((i, 0.0))

        # 按分数降序排序
        counts.sort(key=lambda x: x[1], reverse=True)

        total_score = sum(x[1] for x in counts)

        keep_indices: List[int] = []

        # Coverage 逻辑
        target_coverage = coverage
        current_sum = 0.0

        if total_score == 0:
            keep_indices = [x[0] for x in counts]
        else:
            for exp_id, score in counts:
                keep_indices.append(exp_id)
                current_sum += score
                current_coverage = (current_sum / total_score) * 100.0
                if current_coverage >= target_coverage:
                    break

        # 计算最终覆盖率
        score_dict = dict(counts)
        kept_score = sum(score_dict.get(idx, 0.0) for idx in keep_indices)
        final_coverage = (kept_score / total_score * 100.0) if total_score > 0 else 0.0

        layer_importance[layer_idx] = keep_indices

        print(
            f"  Layer {layer_idx}: keeping {len(keep_indices)}/{len(counts)} "
            f"({len(keep_indices) / len(counts) * 100:.2f}%) experts. "
            f"Score Coverage: {final_coverage:.2f}%"
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


def process_metadata(reader: GGUFReader, writer: GGUFWriter) -> None:
    # Add pruning flag
    writer.add_bool("llama_moe.is_pruned", True)

    # copy other metadata
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
            writer.add_tensor(name, data, raw_dtype=tensor_type)
        else:
            writer.add_tensor(name, data)


def prune_model_with_report(
    gguf_path: str,
    report_dir: str,
    coverage: float,
    output_path: str | None = None,
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

    counts_path, weights_path = resolve_report_paths(report_dir)

    importance_map = load_expert_importance(
        counts_path=counts_path,
        weights_path=weights_path,
        coverage=coverage,
        total_experts=old_expert_count,
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

    if output_path is None:
        p = Path(gguf_path)
        suffix_val = f"cov{int(coverage)}"
        output_path = str(p.with_name(f"{p.stem}-pruned_{suffix_val}{p.suffix}"))

    writer = GGUFWriter(output_path, arch)

    # Add stats to KV
    writer.add_uint64("llama_moe.pruned_experts_count", total_pruned_experts)
    writer.add_uint64("llama_moe.original_experts_count", total_original_experts)

    print("Processing metadata...")
    process_metadata(reader, writer)

    print("Processing tensors...")
    process_tensors(reader, writer, old_expert_count, importance_map)

    print("Writing file...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    try:
        before_size = Path(gguf_path).stat().st_size
        after_size = Path(output_path).stat().st_size
        if before_size > 0:
            reduced_pct = (1.0 - (after_size / before_size)) * 100.0
        else:
            reduced_pct = 0.0

        GIB = 1024**3

        print(
            "File size: "
            f"{before_size / GIB:.2f} GiB -> "
            f"{after_size / GIB:.2f} GiB, "
            f"reduced {reduced_pct:.2f}%"
        )
    except OSError as e:
        print(f"File size: unavailable ({e})")

    print("Done.")
    print(f"Output path: {output_path}")
    return output_path


# python -m llama_moe.pruner /mnt/data/gguf/Qwen3-Next-80B-A3B-Instruct-Q8_0.gguf --coverage 90 --report-dir .
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="GGUF MoE 裁剪脚本")
    ap.add_argument("gguf_path", help="GGUF 文件路径")
    ap.add_argument(
        "--coverage", type=float, required=True, help="保留专家直到达到覆盖率 (0-100)"
    )
    ap.add_argument(
        "--report-dir",
        type=str,
        default=".",
        help="目录路径，包含 expert_activations.csv 和 expert_weights.csv（默认当前目录）",
    )
    ap.add_argument("--output", "-o", type=str, default=None, help="输出文件路径")

    args = ap.parse_args()

    prune_model_with_report(
        args.gguf_path,
        args.report_dir,
        args.coverage,
        args.output,
    )
