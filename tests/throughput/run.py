import sys
from datetime import datetime
from pathlib import Path

import gpu_recorder
from evalscope.config import EvalType, TaskConfig
from evalscope.run import run_task
from gguf import GGUFReader
from log_analysis import log_analysis

from llama_moe import LlamaServerWrapper, get_override_rules


def run_eval(model: str, model_dir: Path, max_tokens):
    """
    运行模型评估
    """

    # 切换到模型版本目录
    work_dir = model_dir / "evalscope"
    task_config = TaskConfig(
        model=model,
        datasets=["gsm8k"],
        eval_type=EvalType.SERVICE,
        eval_batch_size=1,
        api_url="http://127.0.0.1:8080/v1/chat/completions",
        api_key="sk-1234",
        generation_config={
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": True,
        },
        limit=1,
        work_dir=str(work_dir),
    )

    start_time = datetime.now()
    print(f"[{start_time.strftime('%Y-%m-%d %H:%M:%S')}] 开始启动评估")

    run_task(task_cfg=task_config)

    end_time = datetime.now()
    print(
        f"[{end_time.strftime('%Y-%m-%d %H:%M:%S')}] 结束评估, 用时 {end_time - start_time} 秒"
    )


base_dir = Path(__file__).resolve().parent
run_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
work_dir = base_dir / f"results-{run_timestamp}"
work_dir.mkdir(exist_ok=True)

# 创建主日志文件
log_file = work_dir / "run.log"
f = open(log_file, "w", encoding="utf-8")
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
    "Qwen3/Qwen3-30B-A3B-Q8_0": {
        "path": "/mnt/gguf/Qwen3-30B-A3B-Q8_0.gguf",
        "ctx_size": 32768,
        "base": ["--n-gpu-layers", "30"],
    }
}

versions = ["base", "all_exps_on_cpu", "llama_moe"]

for name, model in model_list.items():
    path = model["path"]
    ctx_size = model["ctx_size"]

    for version in versions:
        # 创建模型版本专用目录
        model_dir = work_dir / name.replace("/", "_") / version
        model_dir.mkdir(parents=True, exist_ok=True)

        final_arg = common_args + ["--model", path, "--ctx-size", str(ctx_size)]

        match version:
            case "base":
                final_arg += model["base"]
            case "all_exps_on_cpu":
                final_arg += ["--n-gpu-layers", "999", "--cpu-moe"]
            case "llama_moe":
                final_arg += get_override_rules(GGUFReader(path), ctx_size)

        print(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 正在启动 {name} 的 {version} 版本..."
        )

        wrapper = LlamaServerWrapper(model_dir / "llama-server.log")
        try:
            pid = wrapper.run(final_arg)
            if pid < 0:
                print(f"启动 {name}-{version} 失败")
                continue

            # 开始GPU监控
            gpu_recorder.start()
            memory_used, memory_percent = gpu_recorder.get_memory_usage()
            print(f"显存使用: {memory_used:.2f} MB ({memory_percent:.2f}%)")

            # 运行评估，传入模型目录，在函数内部切换工作目录
            run_eval(name, model_dir, ctx_size)

            # 结束GPU监控并保存数据
            gpu_recorder.finish(str(model_dir / "gpu_utilization.npz"))

            # 分析服务器日志
            log_analysis(str(model_dir / "llama-server.log"))

        except Exception as e:
            print(f"运行 {name}-{version} 时出错: {e}")
        finally:
            wrapper.stop()

print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 所有评估完成")

f.close()
