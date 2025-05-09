import copy
import datetime
import re
import subprocess
from dataclasses import dataclass, field, fields
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

import tomllib
from prettytable import PrettyTable


@dataclass
class Config:
    description: Optional[str] = None
    run: bool = False
    baseline: bool = False
    output: bool = False
    args: Dict[str, Any] = field(default_factory=dict)
    results: List[List[float]] = field(default_factory=list)

    def add_result(self, prompt_tps: float, eval_tps: float) -> None:
        self.results.append([prompt_tps, eval_tps])

    def get_avg_prompt_tps(self) -> Optional[float]:
        valid_values = [x[0] for x in self.results if x[0] is not None]
        return mean(valid_values) if len(valid_values) > 0 else None

    def get_avg_eval_tps(self) -> Optional[float]:
        valid_values = [x[1] for x in self.results if x[1] is not None]
        return mean(valid_values) if len(valid_values) > 0 else None


def load_config(
    config_file: Path = Path(__file__).parent / "configs.toml",
) -> Tuple[Dict[str, Any], List[Config]]:
    if not config_file.is_file():
        print(f"Error: Config file {config_file} not found.")
        exit(1)

    print(f"Loading config file: {config_file}")

    try:
        with open(config_file, "rb") as f:
            data = tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        print(f"Error: Failed to parse TOML file {config_file}: {e}")
        exit(1)
    except Exception as e:
        print(f"Error: Unexpected error reading config file {config_file}: {e}")
        exit(1)

    settings = data.get("settings")
    if not isinstance(settings, dict):
        print("Error: [settings] section is not a valid table (dictionary).")
        exit(1)

    base_config_data = data.get("base_config")
    if not isinstance(base_config_data, dict):
        print("Error: Required [base_config] table missing.")
        exit(1)

    base_config = Config(**base_config_data)
    configs: List[Config] = []

    # 处理配置列表 [[configs]]
    configs_data = data.get("configs", [])
    if not configs_data:
        print("Warning: No override configurations found in [configs] section.")

    # 遍历覆盖配置列表
    for i, override_data in enumerate(configs_data):
        if not isinstance(override_data, dict):
            print(f"Error: [configs] section {i} is not a valid table (dictionary).")
            exit(1)

        # 创建一个新的配置对象，首先复制基础配置的所有属性
        config_dict = {}
        for field_ in fields(base_config):
            # 复制基础配置的值
            config_dict[field_.name] = copy.deepcopy(getattr(base_config, field_.name))

        # 然后应用覆盖配置
        for key, value in override_data.items():
            if key == "args" and isinstance(value, dict):
                # 对于args字段，我们需要合并而不是替换
                config_dict["args"] = {**config_dict["args"], **value}
            else:
                # 对于其他字段，直接覆盖
                config_dict[key] = value

        # 创建新的配置对象
        override_config = Config(**config_dict)
        configs.append(override_config)
        print(
            f"Found config: {override_config.description}"
            f"{' (run)' if override_config.run else ' (skip)'}"
            f"{' (baselince)' if override_config.baseline else ''}"
        )

    if not configs:
        print("Error: Failed to parse any configuration objects.")
        exit(1)

    return settings, configs


def extract_tps(output: str) -> tuple[Optional[float], Optional[float]]:
    prompt_tps_match = re.search(
        r"prompt eval time =.*?(\d+\.\d+) tokens per second", output
    )
    eval_tps_match = re.search(
        r"eval time =.*?runs.*?(\d+\.\d+) tokens per second", output
    )

    if not prompt_tps_match or not eval_tps_match:
        print(f"Failed to extract TPS from output:\n{output}")
        return None, None
    return float(prompt_tps_match.group(1)), float(eval_tps_match.group(1))


def run_experiment(settings: dict, configs: List[Config]) -> None:
    repeat = settings["repeat"]

    for i in range(repeat):
        for config in configs:
            if not config.run:
                continue
            
            print(f"[{i + 1}/{repeat}] {config.description}")

            cmd = [
                "llama.cpp/build/bin/llama-cli",
                "--single-turn",
            ]
            for arg, value in config.args.items():
                if isinstance(value, bool):
                    if value:
                        cmd.append(f"--{arg}")
                elif value is None or value == "":
                    continue
                else:
                    cmd.append(f"--{arg}")
                    cmd.append(str(value))
            # print(f"Command: {' '.join(cmd)}")

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

            if config.output:
                print(output)

            prompt_tps, eval_tps = extract_tps(output)
            config.add_result(prompt_tps, eval_tps)


def generate_results_table(configs: List[Config]) -> PrettyTable:
    table = PrettyTable()
    table.field_names = ["Description", "Prefill TPS", "Decode TPS"]
    table.align = "c"
    table.padding_width = 2

    base_prompt_tps = None
    base_eval_tps = None

    for config in configs:
        if config.baseline and config.run:
            base_prompt_tps = config.get_avg_prompt_tps()
            base_eval_tps = config.get_avg_eval_tps()

    for config in configs:
        if config.run:
            avg_prompt_tps = config.get_avg_prompt_tps()
            avg_eval_tps = config.get_avg_eval_tps()
            if base_prompt_tps and base_eval_tps:
                prompt_percent = round((avg_prompt_tps / base_prompt_tps) * 100)
                eval_percent = round((avg_eval_tps / base_eval_tps) * 100)
                prompt_tps_str = f"{avg_prompt_tps:.2f}({prompt_percent}%)"
                eval_tps_str = f"{avg_eval_tps:.2f}({eval_percent}%)"
            else:
                prompt_tps_str = f"{avg_prompt_tps:.2f}(N/A)"
                eval_tps_str = f"{avg_eval_tps:.2f}(N/A)"
            table.add_row([config.description, prompt_tps_str, eval_tps_str])

    return table


def main():
    print(f"Start Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    settings, configs = load_config()

    run_experiment(settings, configs)

    table = generate_results_table(configs)
    print("All experiments completed.")
    print(f"Finish Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(table)


if __name__ == "__main__":
    main()
