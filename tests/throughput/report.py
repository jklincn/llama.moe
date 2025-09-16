import datetime
from pathlib import Path
import argparse


class LogResult:
    model: str
    version: str

    start_time: datetime.datetime
    end_time: datetime.datetime
    duration: float

    memory_used: float  # GB
    memory_percent: float  # %
    gpu_utilization: list[float]  # GPU利用率时间序列

    num_requests: int
    num_prompt_tokens: int
    num_eval_tokens: int
    avg_tokens_per_request: float
    avg_tokens_per_eval: float

    tokens_per_second_prompt: float  # tokens/s
    tokens_per_second_eval: float  # tokens/s

    score: float  # 综合评分


class TestReport:
    result: list[LogResult] = []


def analysis(results_path: Path):
    for model in results_path.iterdir():
        if model.is_dir():
            for version in model.iterdir():
                print(f"分析 {model.name} ({version.name}) ...")
    pass


if __name__ == "__main__":
    argparse = argparse.ArgumentParser(description="Result Analysis")
    argparse.add_argument(
        "results_path",
        type=str,
        default="./results",
        help="Path to the results directory",
    )
    args = argparse.parse_args()
    analysis(Path(args.results_path))
