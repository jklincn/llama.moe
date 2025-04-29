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

# 打印当前时间
print(f"Start Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# 基本参数设置
# micro model: /mnt/data/gguf/qwen2.5-0.5b-instruct-q4_k_m.gguf
# big model: /mnt/data/gguf/DeepSeek-R1-Q4_K_M.gguf
model_path = "/mnt/data/gguf/DeepSeek-R1-Q4_K_M.gguf"

prompt = "Please help me write a paragraph introducing Beijing."
n_predict = 100
repeat = 2


# 定义 Config 类
@dataclass
class Config:
    gpu_layers: int
    override: Optional[str]
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


# 加载 JSON 配置文件
config_file = Path(__file__).parent / "configs.json"
if not os.path.isfile(config_file):
    print(f"Config file {config_file} not found.")
    exit(1)

with open(config_file, "r") as f:
    config_data = json.load(f)

# 创建 Config 实例列表
configs = [
    Config(gpu_layers=data["gpu_layers"], override=data["override"])
    for _, data in sorted(config_data.items(), key=lambda x: int(x[0]))
]

# 检查模型文件是否存在
if not os.path.isfile(model_path):
    print(f"Model {model_path} not found.")
    exit(1)

# 创建表格
table = PrettyTable()
table.field_names = ["GPU Layer", "Override", "Prefill TPS", "Decode TPS"]
table.align = "c"  # 居中对齐
table.padding_width = 2


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


# 运行实验
for i in range(repeat):
    for config in configs:
        print(f"[{i+1}/{repeat}] GPU Layer: {config.gpu_layers}, Override: {config.override}")

        # fmt: off
        # 构建命令
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

        # 运行命令并捕获输出
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout_bin, stderr_bin = process.communicate()

        # 解码输出
        try:
            stdout = stdout_bin.decode("utf-8", errors="ignore")
        except UnicodeDecodeError:
            stdout = ""
        try:
            stderr = stderr_bin.decode("utf-8", errors="ignore")
        except UnicodeDecodeError:
            stderr = ""

        output = stdout + stderr
        prompt_tps, eval_tps = extract_tps(output)

        if prompt_tps is not None and eval_tps is not None:
            config.add_result(prompt_tps, eval_tps)
        else:
            print(
                f"Skipping config {config.gpu_layers, config.override} due to TPS extraction failure."
            )

# 输出结果
base_config = configs[0]  # 第一个配置作为基准
base_prompt_tps = base_config.get_avg_prompt_tps()
base_eval_tps = base_config.get_avg_eval_tps()

# 生成表格
for config in configs:
    avg_prompt_tps = config.get_avg_prompt_tps()
    avg_eval_tps = config.get_avg_eval_tps()

    if avg_prompt_tps is not None and avg_eval_tps is not None:
        # 计算百分比
        if base_prompt_tps and base_eval_tps:
            prompt_percent = round((avg_prompt_tps / base_prompt_tps) * 100)
            eval_percent = round((avg_eval_tps / base_eval_tps) * 100)
            prompt_tps_str = f"{avg_prompt_tps:.2f}({prompt_percent}%)"
            eval_tps_str = f"{avg_eval_tps:.2f}({eval_percent}%)"
        else:
            prompt_tps_str = f"{avg_prompt_tps:.2f}(N/A)"
            eval_tps_str = f"{avg_eval_tps:.2f}(N/A)"

        table.add_row(
            [
                config.gpu_layers,
                str(config.override),
                prompt_tps_str,
                eval_tps_str,
            ]
        )
    else:
        table.add_row([config.gpu_layers, str(config.override), "N/A", "N/A"])

print("All experiments completed.")
print(f"Finish Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print({table})