import json
import os
import re
import subprocess
import datetime
from statistics import mean
from prettytable import PrettyTable
from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path


@dataclass
class Config:
    gpu_layers: int
    override: Optional[str]
    test: bool
    results: List[List[float]] = field(default_factory=list)

    def add_result(self, prompt_tps: float, eval_tps: float) -> None:
        """添加一次实验结果"""
        self.results.append([prompt_tps, eval_tps])

    def get_avg_prompt_tps(self) -> Optional[float]:
        """计算平均 prompt_tps"""
        return mean([x[0] for x in self.results]) if self.results else None

    def get_avg_eval_tps(self) -> Optional[float]:
        """计算平均 eval_tps"""
        return mean([x[1] for x in self.results]) if self.results else None


def load_config(
    config_file: Path = Path(__file__).parent / "configs.json",
) -> tuple[dict, List[Config]]:
    """加载 JSON 配置文件并创建 Config 实例"""
    if not os.path.isfile(config_file):
        print(f"Config file {config_file} not found.")
        exit(1)

    with open(config_file, "r") as f:
        config_data = json.load(f)

    if (
        not isinstance(config_data, dict)
        or "settings" not in config_data
        or "configs" not in config_data
    ):
        print("Invalid JSON format: 'settings' and 'configs' fields are required.")
        exit(1)

    settings = config_data["settings"]
    if not os.path.isfile(settings["model_path"]):
        print(f"Model {settings['model_path']} not found.")
        exit(1)

    configs = [
        Config(
            gpu_layers=data["gpu_layers"], override=data["override"], test=data["test"]
        )
        for data in config_data["configs"]
    ]
    return settings, configs


def extract_tps(output: str) -> tuple[Optional[float], Optional[float]]:
    """从输出中提取 prompt 和 eval 的 TPS"""
    prompt_tps_match = re.search(
        r"prompt eval time =.*?(\d+\.\d+) tokens per second", output
    )
    eval_tps_match = re.search(
        r"eval time =.*?runs.*?(\d+\.\d+) tokens per second", output
    )

    if not prompt_tps_match or not eval_tps_match:
        print(f"Failed to extract TPS from output: {output[:200]}...")
        return None, None
    return float(prompt_tps_match.group(1)), float(eval_tps_match.group(1))


def run_experiment(settings: dict, configs: List[Config]) -> None:
    """运行实验，遍历所有配置并记录结果"""
    model_path = settings["model_path"]
    prompt = settings["prompt"]
    n_predict = settings["n_predict"]
    repeat = settings["repeat"]

    for i in range(repeat):
        for config in configs:
            print(
                f"[{i + 1}/{repeat}] GPU Layer: {config.gpu_layers}, Override: {config.override}"
            )
            if not config.test:
                print("      skip")
                continue

            # fmt: off
            cmd = [
                "llama.cpp/build/bin/llama-cli",
                "-m", model_path,
                "--prompt", prompt,
                "--seed", str(0),
                "--n-predict", str(n_predict),
                "--n-gpu-layers", str(config.gpu_layers),
                "--single-turn",
            ]
            # fmt: on

            if config.override is not None:
                cmd.extend(["-ot", config.override])

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            stdout, stderr = process.communicate()
            output = stdout + stderr

            prompt_tps, eval_tps = extract_tps(output)

            if prompt_tps is not None and eval_tps is not None:
                config.add_result(prompt_tps, eval_tps)
            else:
                print(
                    f"Skipping config {config.gpu_layers, config.override} due to TPS extraction failure."
                )


def generate_results_table(configs: List[Config]) -> PrettyTable:
    """生成并返回结果表格"""
    table = PrettyTable()
    table.field_names = ["GPU Layer", "Override", "Prefill TPS", "Decode TPS"]
    table.align = "c"
    table.padding_width = 2

    base_config = configs[1]
    base_prompt_tps = base_config.get_avg_prompt_tps()
    base_eval_tps = base_config.get_avg_eval_tps()

    for config in configs:
        avg_prompt_tps = config.get_avg_prompt_tps()
        avg_eval_tps = config.get_avg_eval_tps()

        if avg_prompt_tps is not None and avg_eval_tps is not None:
            if base_prompt_tps and base_eval_tps:
                prompt_percent = round((avg_prompt_tps / base_prompt_tps) * 100)
                eval_percent = round((avg_eval_tps / base_eval_tps) * 100)
                prompt_tps_str = f"{avg_prompt_tps:.2f}({prompt_percent}%)"
                eval_tps_str = f"{avg_eval_tps:.2f}({eval_percent}%)"
            else:
                prompt_tps_str = f"{avg_prompt_tps:.2f}(N/A)"
                eval_tps_str = f"{avg_eval_tps:.2f}(N/A)"
            table.add_row(
                [config.gpu_layers, str(config.override), prompt_tps_str, eval_tps_str]
            )
        else:
            table.add_row([config.gpu_layers, str(config.override), "N/A", "N/A"])

    return table


def main():
    """主函数，协调整个实验流程"""
    print(f"Start Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    settings, configs = load_config()

    run_experiment(settings, configs)

    table = generate_results_table(configs)
    print("All experiments completed.")
    print(f"Finish Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(table)


if __name__ == "__main__":
    main()
