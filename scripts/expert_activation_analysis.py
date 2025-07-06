import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def heatmap(
    csv_path: str = "expert_activations.csv",
    out_path: str = "expert_activation_heatmap.png",
) -> None:
    # ---------- 读取数据 ----------
    df = pd.read_csv(csv_path)

    # 自动判断：第 1 列为层号，其余为专家激活次数
    layer_col = df.columns[0]  # 取首列列名
    activations = df.iloc[:, 1:].to_numpy()

    num_layers = activations.shape[0]  # 行数
    num_experts = activations.shape[1]  # 列数

    # ---------- 作图 ----------
    # 动态调整画布：让单元格尽量是“接近方形”
    base = 0.25  # 单元格的基础尺寸（英寸）
    fig_w = max(8, num_experts * base)
    fig_h = max(6, num_layers * base)

    plt.figure(figsize=(fig_w, fig_h))
    im = plt.imshow(activations, aspect="auto", cmap="Blues", interpolation="nearest")

    # 颜色条
    cbar = plt.colorbar(im)
    cbar.set_label("Activation Count", fontsize=12)

    # 轴标签
    plt.xlabel(f"Expert Index (0-{num_experts - 1})", fontsize=12)
    plt.ylabel("Model Layer ID", fontsize=12)

    # Y 轴刻度——每层都标
    plt.yticks(ticks=np.arange(num_layers), labels=df[layer_col])

    # X 轴刻度——根据专家数量自适应：步长≈把刻度控制在 ≤16 个
    step = max(1, num_experts // 16)
    xticks = np.arange(0, num_experts, step)
    plt.xticks(ticks=xticks, labels=xticks)

    # 标题
    plt.title(
        "Expert Activation Heatmap\n(dark blue = many activations, light blue = few activations)",
        fontsize=14,
    )

    plt.tight_layout()

    # ---------- 保存并预览 ----------
    plt.savefig(out_path, dpi=300)
    print(f"[✓] 已保存热图 → {Path(out_path).resolve()}")


def build_k_list(num_experts: int, step: int = 4) -> list[int]:
    """生成阈值列表：0, step, 2*step, …, num_experts（含 0 与末尾）"""
    ks = list(range(0, num_experts + 1, step))
    if ks[-1] != num_experts:  # 若总专家数不是 step 的整数倍
        ks.append(num_experts)
    return ks


def topk_proportions(
    csv_path: str = "expert_activations.csv",
    out_path: str = "expert_proportions.csv",
    step: int = 4,
) -> None:
    # ---------- 读取 CSV ----------
    df = pd.read_csv(csv_path)
    layer_col = df.columns[0]  # 第一列：层号
    acts = df.iloc[:, 1:].to_numpy()  # shape = (L, N)

    num_layers, num_experts = acts.shape
    ks = build_k_list(num_experts, step)  # 0,4,8,…,N

    # ---------- 总激活 & 行内降序 ----------
    total = acts.sum(axis=1, keepdims=True).astype(float)
    total[total == 0] = 1.0  # 防止除零
    sorted_acts = -np.sort(-acts, axis=1)  # 行内降序

    # ---------- 计算占比 ----------
    result = {layer_col: df[layer_col]}
    ratios = []  # 收集所有层 × K 的占比，便于后续查 90 %
    for k in ks:
        if k == 0:
            ratio = np.zeros(num_layers)
        else:
            topk_sum = sorted_acts[:, :k].sum(axis=1)
            ratio = topk_sum / total.flatten()
        result[f"top{k}"] = ratio
        ratios.append(ratio)  # len == len(ks)

    # ---------- 计算“首个 >90 % 的 K” ----------
    ratio_matrix = np.vstack(ratios).T  # shape = (L, len(ks))
    first_k_over90 = []
    for row in ratio_matrix:
        # 找到第一个比例 > 0.9 的列索引
        idx = np.argmax(row > 0.90)
        if row[idx] > 0.90:
            first_k_over90.append(ks[idx])
        else:  # 整层都没超过 90 %
            first_k_over90.append(np.nan)

    result["first_k_over90"] = first_k_over90

    # ---------- 输出 ----------
    out_df = pd.DataFrame(result)
    out_df.to_csv(out_path, index=False)
    print(f"[✓] 已保存结果 → {Path(out_path).resolve()}")


if __name__ == "__main__":
    topk_proportions()
    heatmap()
