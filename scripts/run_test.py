import collections
import os
import re
import subprocess
import datetime
from prettytable import PrettyTable

# 打印当前时间
print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

# 基本参数设置
# simple: /mnt/data/gguf/qwen2.5-0.5b-instruct-q4_k_m.gguf
model_path = "/mnt/data/gguf/qwen2.5-0.5b-instruct-q4_k_m.gguf"
prompt = "Please help me write a paragraph introducing Beijing."
n_predict = 100
repeat = 1

# 定义实验配置
# 格式: [gpu_layers, use_cpu_exps]
configs = [
    [2, False],  # 不使用 exps=CPU，gpu-layers=2
    [10, True],  # 使用 exps=CPU，gpu-layers=10
    [20, True],  # 使用 exps=CPU，gpu-layers=20
    [30, True],  # 使用 exps=CPU，gpu-layers=30
    [40, True],  # 使用 exps=CPU，gpu-layers=40
]

# 检查模型文件是否存在
if not os.path.isfile(model_path):
    print(f"Model {model_path} not found.")
    exit(1)

# 创建表格
table = PrettyTable()
table.field_names = ["GPU Layer", "Override", "Prefill TPS", "Decode TPS"]
table.align = "c"  # 居中对齐
table.padding_width = 2  # 单元格内边距

# 初始化结果存储，使用元组作为键
result = collections.defaultdict(list)

def extract_tps(output):
    """从输出中提取 prompt 和 eval 的 TPS"""
    prompt_tps_match = re.search(r"prompt eval time =.*?(\d+\.\d+) tokens per second", output)
    eval_tps_match = re.search(r"eval time =.*?runs.*?(\d+\.\d+) tokens per second", output)
    
    if not prompt_tps_match or not eval_tps_match:
        print(f"Failed to extract TPS from output: {output[:200]}...")  # 截断长输出
        return None, None
    return float(prompt_tps_match.group(1)), float(eval_tps_match.group(1))

for i in range(repeat):
    print(f"Repeat {i + 1}")
    for config in configs:
        gpu_layers, use_cpu_exps = config
        print(f"GPU Layer: {gpu_layers}, Override: {use_cpu_exps}")

        # 构建命令
        cmd = [
            "llama.cpp/build/bin/llama-cli",
            "-m", model_path,
            "--prompt", prompt,
            "--seed", str(0),
            "--n-predict", str(n_predict),
            "--n-gpu-layers", str(gpu_layers),
            "--single-turn",
        ]
        if use_cpu_exps:
            cmd.extend(["-ot", "exps=CPU"])

        # 运行命令并捕获输出
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout_bin, stderr_bin = process.communicate()

        # 解码输出，忽略错误
        try:
            stdout = stdout_bin.decode('utf-8', errors='ignore')
        except UnicodeDecodeError:
            stdout = ""
        try:
            stderr = stderr_bin.decode('utf-8', errors='ignore')
        except UnicodeDecodeError:
            stderr = ""

        output = stdout + stderr
        prompt_tps, eval_tps = extract_tps(output)

        if prompt_tps is not None and eval_tps is not None:
            # 使用元组作为键
            config_key = tuple(config)
            result[config_key].append([prompt_tps, eval_tps])
        else:
            print(f"Skipping config {config} due to TPS extraction failure.")

# 输出结果
base_config = tuple(configs[0])  # 第一个配置作为基准
base_prompt_tps = None
base_eval_tps = None

# 获取基准值
if base_config in result and result[base_config]:
    prompt_tps_values = [x[0] for x in result[base_config]]
    eval_tps_values = [x[1] for x in result[base_config]]
    base_prompt_tps = sum(prompt_tps_values) / len(prompt_tps_values)
    base_eval_tps = sum(eval_tps_values) / len(eval_tps_values)

for config in configs:
    config_key = tuple(config)
    gpu_layers, use_cpu_exps = config
    if config_key in result and result[config_key]:
        prompt_tps_values = [x[0] for x in result[config_key]]
        eval_tps_values = [x[1] for x in result[config_key]]
        avg_prompt_tps = sum(prompt_tps_values) / len(prompt_tps_values)
        avg_eval_tps = sum(eval_tps_values) / len(eval_tps_values)
        
        # 计算百分比
        if base_prompt_tps and base_eval_tps:
            prompt_percent = round((avg_prompt_tps / base_prompt_tps) * 100)
            eval_percent = round((avg_eval_tps / base_eval_tps) * 100)
            prompt_tps_str = f"{avg_prompt_tps:.2f} ({prompt_percent}%)"
            eval_tps_str = f"{avg_eval_tps:.2f} ({eval_percent}%)"
        else:
            # 如果没有基准值，显示 N/A
            prompt_tps_str = f"{avg_prompt_tps:.2f}(N/A)"
            eval_tps_str = f"{avg_eval_tps:.2f}(N/A)"
        
        table.add_row([gpu_layers, str(use_cpu_exps), prompt_tps_str, eval_tps_str])
    else:
        table.add_row([gpu_layers, str(use_cpu_exps), "N/A", "N/A"])

print(table)
print("\nAll experiments completed.")
print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
