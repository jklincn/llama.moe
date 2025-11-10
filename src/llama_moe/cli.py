import argparse
import sys
from pathlib import Path
import time
from gguf import GGUFReader
import logging
from .override import get_override_rules
from .wrapper import LlamaServerWrapper
import os

LOG_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
DATEFMT = "%Y-%m-%d %H:%M:%S"


def _setup_cli_logging():
    logging.basicConfig(handlers=[], force=True)
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(logging.Formatter(LOG_FORMAT, DATEFMT))

    root = logging.getLogger()

    if os.getenv("LLAMA_MOE_DEBUG") == "1":
        root.setLevel(logging.DEBUG)
    else:
        root.setLevel(logging.INFO)

    root.addHandler(handler)


def main():
    _setup_cli_logging()
    logger = logging.getLogger("main")
    # fmt: off
    parser = argparse.ArgumentParser(description="llama.moe", add_help=False)
    parser.add_argument("--ctx-size", "-c", dest="ctx_size", type=int, default=None)
    parser.add_argument("--model", "-m", dest="model", type=lambda s: str(Path(s).expanduser()), default=None)
    parser.add_argument("--no-kv-offload", "-nkvo", dest="no_kv_offload", action="store_true")
    # discard the following parameters
    parser.add_argument("--n-gpu-layers", "--gpu-layers", "-ngl", dest="n-gpu-layers", type=str, default=None)
    parser.add_argument("--override-tensor", "-ot", dest="override-tensor", type=str, default=None)
    parser.add_argument("--cpu-moe", "-cmoe", dest="cpu-moe", type=str, default=None)
    parser.add_argument("--n-cpu-moe", "-ncmoe", dest="n-cpu-moe", type=str, default=None)
    parser.add_argument("--cache-type-k", "-ctk", dest="cache-type-k", type=str, default=None)
    parser.add_argument("--cache-type-v", "-ctv", dest="cache-type-v", type=str, default=None)
    parser.add_argument("--numa", dest="numa", type=str, default=None)
    # fmt: on

    args, other = parser.parse_known_args()
    model, ctx_size, kv_offload = args.model, args.ctx_size, not args.no_kv_offload

    if "00001" in model:
        logger.error(
            "当前仅支持单文件 GGUF 模型，可以参考 scripts/gguf_merge.sh 进行合并"
        )
        return 0

    logger.info("正在寻找最优配置...")
    reader = GGUFReader(model)

    ot_args = get_override_rules(reader, ctx_size, kv_offload)

    final = ["--model", model] + ["--ctx-size", str(ctx_size)] + ot_args + other

    wrapper = LlamaServerWrapper()
    logger.info("正在启动 llama-server...")
    pid = wrapper.run(final, timeout=3600)
    if pid < 0:
        logger.error("llama-server 启动失败")
        return 1
    try:
        logger.info("启动成功, 开始监听 http://127.0.0.1:8080 (key: sk-1234)")
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        logger.info("正在关闭...")
    finally:
        try:
            wrapper.stop()
        except Exception as e:
            logger.error(f"停止子进程时出错: {e}")

    return 0
