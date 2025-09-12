import datetime
import sys

from gguf import GGUFReader
from task import run_eval
from log_analysis import log_analysis
from llama_moe import LlamaServerWrapper, get_override_rules

f = open("output.txt", "w", encoding="utf-8")
sys.stdout = f

# fmt: off
common_args = [
    "--seed", "0",
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

model_list = {
    "Qwen3-30B-A3B-Q8_0": {
        "path": "/mnt/data/gguf/Qwen3-30B-A3B-Q8_0.gguf",
        "ctx_size": 32768,
        "base": ["--n-gpu-layers", "30"],
    }
}

versions = ["base", "all_exps_on_cpu", "llama_moe"]

wrapper = LlamaServerWrapper()

for name, model in model_list.items():
    print(f"\n\n=== Running model: {name} ===")
    path = model["path"]
    ctx_size = model["ctx_size"]
    for version in versions:
        final_arg = common_args + ["--model", path, "--ctx-size", str(ctx_size)]
        if version == "base":
            final_arg += model["base"]
        elif version == "all_exps_on_cpu":
            final_arg += ["--n-gpu-layers", "999", "--cpu-moe"]
        elif version == "llama_moe":
            final_arg += get_override_rules(GGUFReader(path), ctx_size)

        print(f"start time: {datetime.datetime.now()}")

        try:
            pid = wrapper.run(final_arg)
            if pid < 0:
                print("[main] llama-server 启动失败")
                exit(1)
            run_eval()
            log_analysis()
        finally:
            wrapper.stop()

f.close()
