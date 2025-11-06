import logging
import logging.config
import os
import time
from datetime import datetime
from pathlib import Path

import gpu_recorder
from evalscope.config import EvalType, TaskConfig
from evalscope.run import run_task
from gguf import GGUFReader
from report import analysis

from llama_moe import LlamaServerWrapper, get_override_rules

ctx_size = 4096
model_list = {
    "Qwen3-30B-A3B-Q8_0": {
        "path": "/mnt/data/gguf/Qwen3-30B-A3B-Q8_0.gguf",
        "base": ["--n-gpu-layers", "35"],
    },
    "GLM-4.5-Air-Q8_0": {
        "path": "/mnt/data/gguf/GLM-4.5-Air-Q8_0.gguf",
        "base": ["--n-gpu-layers", "10"],
    },
    "Qwen3-235B-A22B-Q8_0": {
        "path": "/mnt/data/gguf/Qwen3-235B-A22B-Q8_0.gguf",
        "base": ["--n-gpu-layers", "8"],
    },
    "GLM-4.5-Q8_0": {
        "path": "/mnt/data/gguf/GLM-4.5-Q8_0.gguf",
        "base": ["--n-gpu-layers", "6"],
    },
    # KV Cache 太大了
    "DeepSeek-R1-Q4_K_M": {
        "path": "/mnt/data/gguf/DeepSeek-R1-Q4_K_M.gguf",
        "base": ["--n-gpu-layers", "2"],
    },
}
versions_list = ["base", "all_exps_on_cpu", "llama_moe"]

test_models = [
    "Qwen3-30B-A3B-Q8_0",
    "GLM-4.5-Air-Q8_0",
    # "Qwen3-235B-A22B-Q8_0",
    # "GLM-4.5-Q8_0",
    # "DeepSeek-R1-Q4_K_M",
]
test_versions = ["base", "llama_moe"]


def setup_logging(log_file: Path, level=logging.INFO):
    LOGGING_CONFIG = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": {
            "file": {
                "class": "logging.FileHandler",
                "formatter": "default",
                "level": level,
                "filename": str(log_file),
                "mode": "w",
                "encoding": "utf-8",
            },
        },
        "root": {
            "level": level,
            "handlers": ["file"],
        },
    }
    logging.config.dictConfig(LOGGING_CONFIG)
    for noisy in [
        "openai",
        "httpcore",
    ]:
        logging.getLogger(noisy).setLevel(logging.WARNING)


def run_eval(model: str, model_dir: Path, ctx_size: int, logger: logging.Logger):
    task_config = TaskConfig(
        model=model,
        # datasets
        # - gsm8k
        # - mmlu
        datasets=["gsm8k"],
        eval_type=EvalType.SERVICE,
        eval_batch_size=1,
        api_url="http://127.0.0.1:8080/v1/chat/completions",
        api_key="sk-1234",
        generation_config={
            "max_tokens": ctx_size,
            "temperature": 0.0,
            "stream": True,
        },
        limit=1,
        use_cache=str(model_dir / "evalscope"),
    )

    start_time = time.time()
    logger.info("开始启动评估")

    run_task(task_cfg=task_config)

    end_time = time.time()
    logger.info("结束评估, 用时 %.2f 秒", end_time - start_time)


def main():
    base_dir = Path(__file__).resolve().parent
    bin_path = base_dir.parent.parent / "llama.cpp" / "build" / "bin" / "llama-server"
    run_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    work_dir = base_dir / f"results-{run_timestamp}"
    work_dir.mkdir(exist_ok=True)

    # 创建主日志文件并初始化全局日志
    log_file = work_dir / "run.log"
    if os.getenv("LLAMA_MOE_DEBUG") == "1":
        setup_logging(log_file=log_file, level=logging.DEBUG)
    else:
        setup_logging(log_file=log_file, level=logging.INFO)
    logger = logging.getLogger("run_test")
    logger.info("日志初始化完成，日志文件：%s", log_file)

    # fmt: off
    common_args = [
        "--seed", "0",
        "--api-key", "sk-1234",
        "--log-verbosity", "0",
        "--ctx-size", str(ctx_size),
    ]
    # fmt: on

    for name, model in model_list.items():
        if name not in test_models:
            continue
        for version in versions_list:
            if version not in test_versions:
                continue

            model_dir = work_dir / name.replace("/", "_") / version
            model_dir.mkdir(parents=True, exist_ok=True)
            path = model["path"]
            final_arg = common_args + ["--model", path]

            match version:
                case "base":
                    final_arg += model["base"]
                case "all_exps_on_cpu":
                    final_arg += ["--n-gpu-layers", "999", "--cpu-moe"]
                case "llama_moe":
                    final_arg += get_override_rules(GGUFReader(path), ctx_size)

            logger.info("正在启动 %s (%s) ...", name, version)

            wrapper = LlamaServerWrapper(
                str(bin_path), str(model_dir / "llama-server.log")
            )
            try:
                pid = wrapper.run(final_arg, timeout=3600)
                if pid < 0:
                    logger.error("启动 %s (%s) 失败", name, version)
                    exit(1)

                # 开始GPU监控
                gpu_recorder.start()

                # 运行评估
                run_eval(name, model_dir, ctx_size, logger)

                # 结束GPU监控并保存数据
                gpu_recorder.finish(str(model_dir / "gpu_info.npz"))

            except Exception as e:
                logger.exception("运行 %s (%s) 时出错: %s", name, version, e)
            finally:
                wrapper.stop()

    logger.info("所有评估完成")

    # 生成报告/分析
    analysis(work_dir, model_order=test_models, version_order=test_versions)


if __name__ == "__main__":
    main()
