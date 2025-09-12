import datetime
import sys

from gguf import GGUFReader
from task import run_eval
from log_analysis import log_analysis
from llama_moe import LlamaServerWrapper, get_override_rules

f = open("output.txt", "w", encoding="utf-8")
sys.stdout = f

model = "/mnt/data/gguf/Qwen3-30B-A3B-Q8_0.gguf"
ctx_size = 32768

# fmt: off
common_args = [
    "--model", model,
    "--seed", "0",
    "--ctx-size", str(ctx_size),
    "--threads", "8",
    "--metrics",
    "--api-key", "sk-1234",
    "--slots",
    "--log-verbosity", "0",
    "--cont-batching",
    "--parallel", "1",
    "--threads-http", "4"
]
# fmt: on

# Qwen3-30B-A3B-Q8_0: 30
arg_base = ["--n-gpu-layers", "30"]

arg_all_exps_on_cpu = ["--n-gpu-layers", "999", "--cpu-moe"]
arg_llama_moe = get_override_rules(GGUFReader(model), ctx_size)

args = {
    "base": arg_base,
    "all_exps_on_cpu": arg_all_exps_on_cpu,
    "llama_moe": arg_llama_moe,
}

wrapper = LlamaServerWrapper()

for name, arg in args.items():
    print(f"\n\n=== Running experiment: {name} ===")
    print(f"start time: {datetime.datetime.now()}")
    final = common_args + arg
    try:
        pid = wrapper.run(final)
        if pid < 0:
            print("[main] llama-server 启动失败")
            exit(1)
        run_eval()
        log_analysis()
    finally:
        wrapper.stop()

f.close()
