import copy
import datetime
import re
import subprocess
import sys
from dataclasses import dataclass, field, fields
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

import tomllib
from prettytable import PrettyTable

# Global log file path
OUTPUT_FILE = "output.txt"


class Tee:
    """A simple tee object that duplicates writes to multiple streams."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
        self.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


def setup_logging(log_path: str) -> None:
    """Redirects all `print` statements to both the terminal and a log file."""

    Path(OUTPUT_FILE).write_text("")
    handle = open(log_path, "a", encoding="utf-8")

    # Duplicate stdout/stderr so every print is written to both places
    sys.stdout = Tee(sys.__stdout__, handle)
    sys.stderr = Tee(sys.__stderr__, handle)


@dataclass
class Config:
    description: Optional[str] = None
    run: bool = False
    baseline: bool = False
    args: Dict[str, Any] = field(default_factory=dict)
    results: List[List[float]] = field(default_factory=list)

    def add_result(self, prompt_tps: float, eval_tps: float) -> None:
        self.results.append([prompt_tps, eval_tps])

    def get_avg_prompt_tps(self) -> Optional[float]:
        valid_values = [x[0] for x in self.results if x[0] is not None]
        return mean(valid_values) if valid_values else None

    def get_avg_eval_tps(self) -> Optional[float]:
        valid_values = [x[1] for x in self.results if x[1] is not None]
        return mean(valid_values) if valid_values else None


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------


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

    # Handle [[configs]] overrides
    configs_data = data.get("configs", [])
    if not configs_data:
        print("Warning: No override configurations found in [configs] section.")

    for i, override_data in enumerate(configs_data):
        if not isinstance(override_data, dict):
            print(f"Error: [configs] section {i} is not a valid table (dictionary).")
            exit(1)

        # Start with a deep‑copy of the base config
        config_dict: Dict[str, Any] = {
            field_.name: copy.deepcopy(getattr(base_config, field_.name))
            for field_ in fields(base_config)
        }

        # Apply overrides (merge `args` instead of replacing)
        for key, value in override_data.items():
            if key == "args" and isinstance(value, dict):
                config_dict["args"].update(value)
            else:
                config_dict[key] = value

        override_config = Config(**config_dict)
        configs.append(override_config)
        print(
            f"Found config: {override_config.description}"
            f" {'(run)' if override_config.run else '(skip)'}"
            f" {'(baseline)' if override_config.baseline else ''}"
        )

    if not configs:
        print("Error: Failed to parse any configuration objects.")
        exit(1)

    return settings, configs


# ---------------------------------------------------------------------------
# Experiment runner & helpers
# ---------------------------------------------------------------------------


def extract_tps(output: str) -> Tuple[Optional[float], Optional[float]]:
    prompt_tps_match = re.search(
        r"prompt eval time =.*?(\d+\.\d+) tokens per second", output
    )
    eval_tps_match = re.search(
        r"eval time =.*?runs.*?(\d+\.\d+) tokens per second", output
    )

    if not prompt_tps_match or not eval_tps_match:
        print(f"Failed to extract TPS from output:\n{output}")
        exit(1)

    return float(prompt_tps_match.group(1)), float(eval_tps_match.group(1))


def run_experiment(settings: dict, configs: List[Config]) -> None:
    print(f"Start Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    repeat = settings["repeat"]
    for i in range(repeat):
        for config in configs:
            if not config.run:
                continue

            header = f"[{i + 1}/{repeat}] {config.description}"
            print(header)

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

            with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                f.write(f"[Command]: {' '.join(cmd)}\n")

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )

            output_lines: List[str] = []
            # Only write *process* output to the log file; do not echo to terminal
            with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                for line in iter(process.stdout.readline, ""):
                    f.write(line)
                    output_lines.append(line)
                f.write("=" * 80 + "\n")

            process.wait()
            output_str = "".join(output_lines)

            prompt_tps, eval_tps = extract_tps(output_str)
            config.add_result(prompt_tps, eval_tps)

    print("All experiments completed.")
    print(f"Finish Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def show_results(configs: List[Config]):
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

    print(table)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    setup_logging(OUTPUT_FILE)

    settings, configs = load_config()
    run_experiment(settings, configs)
    show_results(configs)


if __name__ == "__main__":
    main()
