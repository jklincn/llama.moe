import collections
import os
import re
import subprocess
import datetime

from prettytable import PrettyTable

print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

# 基本参数设置
model_path = "/home/lin/bs/myllama/deepseek-r1-q4_k_m.gguf"
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
result = collections.defaultdict(dict)

for i in range(repeat):
    # 循环运行所有实验
    print(f"Repeat {i + 1}")
    for i, config in enumerate(configs):
        gpu_layers, use_cpu_exps = config
        print(f"GPU Layer: {gpu_layers}, Override: {use_cpu_exps}")

        # fmt: off
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
        # fmt: on

        # 如果使用 CPU exps，添加相应参数
        if use_cpu_exps:
            cmd.extend(["-ot", "exps=CPU"])

        # 运行命令并捕获输出
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout_bin = process.communicate()

        stdout = stdout_bin.decode("utf-8", errors="ignore")

        prompt_tps_match = re.search(
            r"prompt eval time =.*?(\d+\.\d+) tokens per second", stdout
        )
        eval_tps_match = re.search(
            r"eval time =.*?runs.*?(\d+\.\d+) tokens per second", stdout
        )

        prompt_tps = prompt_tps_match.group(1)
        eval_tps = eval_tps_match.group(1)

        result[config].append([float(prompt_tps), float(eval_tps)])


# 输出结果
for i, config in enumerate(configs):
    gpu_layers, use_cpu_exps = config
    prompt_tps = sum(result[config][0]) / len(result[config][0])
    eval_tps = sum(result[config][1]) / len(result[config][1])
    table.add_row(
        [gpu_layers, "True" if use_cpu_exps else "False", prompt_tps, eval_tps]
    )
print(table)
print("\nAll experiments completed.")

print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
