from rich.console import Console
from rich.table import Table
from rich import box


BENCH_COLUMNS = [
    ("model", "Model"),
    ("MMLU", "MMLU"),
    ("MMLU-Pro", "MMLU-Pro"),
    ("SuperGPQA", "SuperGPQA"),
    ("MATH", "MATH"),
    ("EvalPlus", "EvalPlus"),
]


def print_qwen3_benchmark_table(rows: list[dict]) -> None:
    if not rows:
        return

    console = Console()
    console.print()
    console.print("[bold]Qwen3 Benchmark Summary[/bold]\n")

    table = Table(
        box=box.SIMPLE_HEAVY,
        show_lines=False,
        expand=False,
        title="Model Benchmarks",
    )

    # 添加列
    for _, title in BENCH_COLUMNS:
        table.add_column(title, no_wrap=True)

    # 填充行（按你给的模型顺序）
    model_order = [
        "Qwen3-8B",
        "Qwen3-14B",
        "Qwen3-32B",
        "Qwen3-235B-A22B",
    ]
    rows_by_model = {r["model"]: r for r in rows}

    for idx, m in enumerate(model_order):
        r = rows_by_model.get(m)
        if r is None:
            continue

        cells: list[str] = []
        for key, _ in BENCH_COLUMNS:
            v = r.get(key, "")
            if key == "model":
                cell = str(v)
            else:
                if isinstance(v, (int, float)):
                    cell = f"{float(v):.2f}"
                else:
                    cell = str(v)

            cells.append(cell)

        # 最后一行加粗（所有列）
        if idx == len(model_order) - 1:
            cells = [f"[bold]{c}[/bold]" for c in cells]

        table.add_row(*cells)

    console.print(table)
    console.print()


if __name__ == "__main__":
    demo_rows = [
        {
            "model": "Qwen3-8B",
            "MMLU": 76.89,
            "MMLU-Pro": 56.73,
            "SuperGPQA": 31.64,
            "MATH": 60.80,
            "EvalPlus": 67.65,
        },
        {
            "model": "Qwen3-14B",
            "MMLU": 81.05,
            "MMLU-Pro": 61.03,
            "SuperGPQA": 34.27,
            "MATH": 62.02,
            "EvalPlus": 72.23,
        },
        {
            "model": "Qwen3-32B",
            "MMLU": 83.61,
            "MMLU-Pro": 65.54,
            "SuperGPQA": 39.78,
            "MATH": 61.62,
            "EvalPlus": 72.05,
        },
        {
            "model": "Qwen3-235B-A22B",
            "MMLU":87.81,
            "MMLU-Pro": 68.18,
            "SuperGPQA": 44.06,
            "MATH": 71.84,
            "EvalPlus": 77.60,
        },
    ]
    print_qwen3_benchmark_table(demo_rows)
