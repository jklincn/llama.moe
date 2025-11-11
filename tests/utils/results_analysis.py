import argparse
import datetime
import json
import logging
import re
from pathlib import Path

import numpy as np
from rich import box
from rich.console import Console
from rich.table import Table

from .gpu_recorder import load_gpu_info

logger = logging.getLogger("report")


class LogResult:
    version_path: Path

    model: str
    version: str

    start_time: datetime.datetime
    end_time: datetime.datetime
    duration: float

    gpu_util: tuple[int, int, int]  # % (min, max, avg)
    mem_util: tuple[int, int, int]  # % (min, max, avg)
    mem_total_bytes: int
    mem_used: tuple[int, int, int]  # Bytes (min, max, avg)

    num_requests: int  # requests
    prompt_tokens_total: int  # tokens
    eval_tokens_total: int  # tokens
    prompt_time_total: float  # ms
    eval_time_total: float  # ms

    score: float  # 综合评分

    def __init__(self, model: str, version: str, results_path: Path):
        version_path = results_path / model / version
        if not version_path.is_dir():
            raise FileNotFoundError(f"版本目录 {version_path} 不存在")

        self.version_path = version_path
        self.model = model
        self.version = version
        self.num_requests = 0
        self.prompt_tokens_total = 0
        self.eval_tokens_total = 0
        self.prompt_time_total = 0.0
        self.eval_time_total = 0.0

        self.gpu_util = (0, 0, 0)
        self.mem_util = (0, 0, 0)
        self.mem_total_bytes = 0
        self.mem_used = (0, 0, 0)

        self.score = 0.0

        self._get_performance()
        self._get_score()
        self._get_gpu_info()

    def _get_performance(self):
        log_path = self.version_path / "llama-server.log"
        if not log_path.is_file():
            raise FileNotFoundError(f"日志文件 {log_path} 不存在")

        pat_prompt = re.compile(
            r"prompt eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens"
        )
        pat_eval = re.compile(r"^\s*eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens")

        have_prompt_for_current = False  # 状态：本次请求是否已经看到 prompt 行

        try:
            with open(log_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    # 匹配prompt eval行
                    m1 = pat_prompt.search(line)
                    if m1:
                        try:
                            time_ms = float(m1.group(1))
                            tokens = int(m1.group(2))
                            self.prompt_time_total += time_ms
                            self.prompt_tokens_total += tokens
                            have_prompt_for_current = True
                        except ValueError as e:
                            print(f"警告: 第{line_num}行数据解析错误 - {e}")
                        continue

                    # 匹配eval行
                    m2 = pat_eval.search(line)
                    if m2:
                        try:
                            time_ms = float(m2.group(1))
                            tokens = int(m2.group(2))
                            self.eval_time_total += time_ms
                            self.eval_tokens_total += tokens

                            # 只有在同一个片段中先看到 prompt 再看到 eval，才算一个请求完成
                            if have_prompt_for_current:
                                self.num_requests += 1
                                have_prompt_for_current = False
                        except ValueError as e:
                            logger.warning(f"警告: 第{line_num}行数据解析错误 - {e}")
                        continue

        except Exception as e:
            logger.error(f"错误: 读取日志文件失败 - {e}")
            return None

    def _get_score(self):
        matches = list((self.version_path / "evalscope" / "reports").glob("*/*.json"))
        evalscope_report_path = matches[0]
        with open(evalscope_report_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        try:
            self.score = float(data["score"])
        except (KeyError, ValueError, TypeError):
            raise RuntimeError(
                f"Invalid or missing 'score' field in {evalscope_report_path}"
            )

    def _get_gpu_info(self):
        gpu_info_path = self.version_path / "gpu_info.npz"
        gpu_utils, mem_utils, mem_used, mem_total = load_gpu_info(gpu_info_path)

        self.mem_total_bytes = mem_total
        self.gpu_util = [
            int(np.min(gpu_utils)),
            int(np.max(gpu_utils)),
            int(np.mean(gpu_utils)),
        ]
        self.mem_util = [
            int(np.min(mem_utils)),
            int(np.max(mem_utils)),
            int(np.mean(mem_utils)),
        ]
        self.mem_used = [
            int(np.min(mem_used)),
            int(np.max(mem_used)),
            int(np.mean(mem_used)),
        ]

    def to_row(self) -> dict:
        prompt_s = self.prompt_time_total / 1000.0
        eval_s = self.eval_time_total / 1000.0

        _, gpu_util_max, gpu_util_avg = self.gpu_util
        _, mem_util_max, mem_util_avg = self.mem_util
        _, _, mem_avg = self.mem_used
        mem_total = self.mem_total_bytes or 1
        # fmt: off
        return {
            "model": self.model,
            "version": self.version,
            "tps_prompt": (self.prompt_tokens_total / prompt_s),
            "tps_eval":   (self.eval_tokens_total   / eval_s),
            "gpu_util_avg": gpu_util_avg,
            "gpu_util_max": gpu_util_max,
            "mem_util_avg": mem_util_avg,
            "mem_util_max": mem_util_max,
            "mem_avg":  mem_avg / mem_total * 100.0,
            "score": float(self.score),
        }
        # fmt: on


def fmt_float(v: float, digits: int = 1) -> str:
    return f"{float(v):.{digits}f}"


def fmt_int(v: int) -> str:
    return f"{int(v):,}"


# 对 results 目录进行分析
def analysis(
    results_path: Path,
    model_order: list[str] | None = None,
    version_order: list[str] | None = None,
) -> None:
    results: list[LogResult] = []

    for model in sorted(results_path.iterdir()):
        if not model.is_dir():
            continue
        for version in sorted(model.iterdir()):
            if not version.is_dir():
                continue
            try:
                results.append(LogResult(model.name, version.name, results_path))
            except Exception as e:
                logger.error(f"构建 LogResult 失败: {model.name}/{version.name} - {e}")

    rows: list[dict] = [r.to_row() for r in results]

    # 分组
    rows_by_model: dict[str, list[dict]] = {}
    for r in rows:
        rows_by_model.setdefault(r["model"], []).append(r)

    console = Console()
    console.print()
    console.print("[bold]Benchmark Summary[/bold]\n")

    columns = [
        ("version", "Version"),
        ("tps_prompt", "TPS(P)"),
        ("tps_eval", "TPS(E)"),
        ("score", "Score"),
        ("gpu_util", "GPU Util"),
        ("mem_util", "Mem Util"),
        ("mem_avg", "Mem Used"),
    ]

    # model 排序：优先按照给定顺序，其余按名字
    model_keys = rows_by_model.keys()
    if model_order:
        ordered_models = [m for m in model_order if m in model_keys] + [
            m for m in model_keys if m not in model_order
        ]
    else:
        ordered_models = sorted(model_keys)

    for model_name in ordered_models:
        table = Table(
            box=box.SIMPLE_HEAVY,
            show_lines=False,
            expand=False,
            title=f"Model: {model_name}",
        )
        for _, title in columns:
            table.add_column(title, no_wrap=True)

        model_rows = rows_by_model[model_name]

        # version 排序：优先按照给定顺序，其余按原始
        if version_order:
            ordered_versions = [
                v for v in version_order if any(r["version"] == v for r in model_rows)
            ]
            ordered_versions += [
                r["version"] for r in model_rows if r["version"] not in version_order
            ]
            version_map = {r["version"]: r for r in model_rows}
            model_rows = [version_map[v] for v in ordered_versions]

        for r in model_rows:
            row_cells: list[str] = []
            for key, _ in columns:
                if key == "tps_prompt":
                    row_cells.append(fmt_float(r["tps_prompt"], 1))
                elif key == "tps_eval":
                    row_cells.append(fmt_float(r["tps_eval"], 1))
                elif key == "score":
                    row_cells.append(fmt_float(r["score"], 2))
                elif key == "gpu_util":
                    row_cells.append(f"{r['gpu_util_avg']}%")
                elif key == "mem_util":
                    row_cells.append(f"{r['mem_util_avg']}%")
                elif key == "mem_avg":
                    row_cells.append(f"{r['mem_avg']:.1f}%")
                else:
                    row_cells.append(str(r[key]))
            table.add_row(*row_cells)

        console.print(table)
        console.print()


# usage:
# cd llama.moe
# python -m tests.utils.results_analysis PATH

if __name__ == "__main__":
    argparse = argparse.ArgumentParser(description="Result Analysis")
    argparse.add_argument(
        "results_path",
        type=str,
        help="Path to the results directory",
    )
    args = argparse.parse_args()
    test_models = [
        "Qwen3-30B-A3B-Q8_0",
        "GLM-4.5-Air-Q8_0",
        "Qwen3-235B-A22B-Q8_0",
        "GLM-4.5-Q8_0",
    ]
    test_versions = ["base", "llama_moe"]
    analysis(Path(args.results_path), test_models, test_versions)
