import logging
import logging.config
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

from evalscope.config import EvalType, TaskConfig
from evalscope.run import run_task
from gguf import GGUFReader
from utils.monitor import SysMonitor, draw
from utils.results_analysis import analysis

from llama_moe import LlamaServerWrapper, get_override_rules
from llama_moe.core import check_numa

os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)

model_list = {
    "Qwen3-30B-A3B-Q8_0": {
        "unchanged": "/mnt/data/gguf/Qwen3-30B-A3B-Q8_0.gguf",
        "cov85": "/mnt/data/gguf/Qwen3-30B-A3B-Q8_0-pruned_cov85.gguf",
        "cov90": "/mnt/data/gguf/Qwen3-30B-A3B-Q8_0-pruned_cov90.gguf",
        "cov91": "/mnt/data/gguf/Qwen3-30B-A3B-Q8_0-pruned_cov91.gguf",
        "cov92": "/mnt/data/gguf/Qwen3-30B-A3B-Q8_0-pruned_cov92.gguf",
        "cov93": "/mnt/data/gguf/Qwen3-30B-A3B-Q8_0-pruned_cov93.gguf",
        "cov94": "/mnt/data/gguf/Qwen3-30B-A3B-Q8_0-pruned_cov94.gguf",
        "cov95": "/mnt/data/gguf/Qwen3-30B-A3B-Q8_0-pruned_cov95.gguf",
    },
}
versions_list = [
    "unchanged",
    "cov95",
    "cov94",
    "cov93",
    "cov92",
    "cov91",
    "cov90",
    "cov85",
]

test_models = [
    "Qwen3-30B-A3B-Q8_0",
]

test_versions = [
    "unchanged",
    "cov95",
    "cov94",
    "cov93",
    "cov92",
    "cov91",
    "cov90",
    "cov85",
]

ctx_size = 4096


def setup_logging(log_file: Path):
    if os.getenv("LLAMA_MOE_DEBUG") == "1":
        level = logging.DEBUG
    else:
        level = logging.INFO
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
        "matplotlib",
    ]:
        logging.getLogger(noisy).setLevel(logging.WARNING)


def run_evalscope(model: str, model_dir: Path, ctx_size: int, logger: logging.Logger):
    # datasets
    # - gsm8k
    # - mmlu
    datasets = ["gsm8k"]
    limit = 10
    task_config = TaskConfig(
        model=model,
        datasets=datasets,
        eval_type=EvalType.SERVICE,
        eval_batch_size=1,
        api_url="http://127.0.0.1:8080/v1/chat/completions",
        api_key="sk-1234",
        generation_config={
            "max_tokens": ctx_size,
            "temperature": 0.0,
            "stream": True,
        },
        # limit=limit,
        use_cache=str(model_dir / "evalscope"),
    )
    logger.info(f"评估数据集 {datasets}, 限制样本数: {limit}")
    start_time = time.time()
    logger.info("开始启动评估")

    run_task(task_cfg=task_config)

    end_time = time.time()
    logger.info("结束评估, 用时 %.2f 秒", end_time - start_time)


def clear_page_cache():
    command = "sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'"
    subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
    time.sleep(1)


def main():
    base_dir = Path(__file__).resolve().parent
    bin_path = base_dir.parent / "llama.cpp" / "build" / "bin" / "llama-server"
    run_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    work_dir = base_dir / f"results-{run_timestamp}"
    work_dir.mkdir(exist_ok=True)

    # 创建主日志文件并初始化全局日志
    log_file = work_dir / "run.log"
    setup_logging(log_file=log_file)
    print("日志文件: ", log_file)
    logger = logging.getLogger("throughput")
    # fmt: off
    common_args = [
        "--seed", "0",
        "--api-key", "sk-1234",
        "--log-verbosity", "0",
        "--ctx-size", str(ctx_size),
        "--metrics",
    ]
    # fmt: on

    for name, model in model_list.items():
        if name not in test_models:
            continue
        for version in versions_list:
            if version not in test_versions:
                continue

            path = model.get(version)
            if not path:
                continue

            model_dir = work_dir / name.replace("/", "_") / version
            model_dir.mkdir(parents=True, exist_ok=True)
            final_arg = common_args + ["--model", path]

            clear_page_cache()
            logger.info(f"正在启动 {name} ({version}) ...")

            numactl_cmd, numa_args = check_numa(path)
            final_arg += (
                get_override_rules(GGUFReader(path), ctx_size) + numa_args
            )
            wrapper = LlamaServerWrapper(
                str(bin_path),
                model_dir,
                numactl=numactl_cmd,
            )

            try:
                pid = wrapper.start(final_arg, timeout=3600)
                if pid < 0:
                    logger.error(f"启动 {name} ({version}) 失败")
                    exit(1)
                else:
                    logger.info(f"llama-server 启动成功, PID: {pid}")

                monitor = SysMonitor(interval=0.1)
                monitor.start()

                run_evalscope(name, model_dir, ctx_size, logger)

            except Exception as e:
                logger.exception(f"运行 {name} ({version}) 时出错: {e}")
            finally:
                wrapper.stop()
                time.sleep(1)
                results = monitor.end(save_path=model_dir / "sys_monitor.csv")

            draw(
                results,
                title=f"{name} ({version})",
                llama_server_log_path=model_dir / "llama-server.log",
                output=str(model_dir / "sys_monitor.png"),
            )

    logger.info("所有评估完成")

    # 生成报告/分析
    analysis(work_dir, model_order=test_models, version_order=test_versions)


if __name__ == "__main__":
    main()
