import argparse
import sys
from pathlib import Path
import time
from gguf import GGUFReader
import logging
from .override import get_override_rules
from .wrapper import LlamaServerWrapper


LOG_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
DATEFMT = "%Y-%m-%d %H:%M:%S"

class _LowerLevelFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.levelname = record.levelname.lower()
        return True
    
def _setup_cli_logging():
    # 作为 CLI 入口，直接配置统一的终端输出，不考虑“被覆盖”的情况
    logging.basicConfig(handlers=[], force=True)  # 清空并强制生效（独立进程安全）
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(logging.Formatter(LOG_FORMAT, DATEFMT))
    handler.addFilter(_LowerLevelFilter())

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(handler)

def main():
    _setup_cli_logging()
    logger = logging.getLogger("main")
    # fmt: off
    parser = argparse.ArgumentParser(description="llama.moe", add_help=False)
    parser.add_argument("--ctx-size", "-c", dest="ctx_size", type=int, default=None)
    parser.add_argument("--model", "-m", dest="model", type=lambda s: str(Path(s).expanduser()), default=None)

    # ignore the following parameters
    parser.add_argument("--n-gpu-layers", "--gpu-layers", "-ngl", dest="n-gpu-layers", type=str, default=None)
    parser.add_argument("--override-tensor", "-ot", dest="override-tensor", type=str, default=None)
    parser.add_argument("--cpu-moe", "-cmoe", dest="cpu-moe", type=str, default=None)
    parser.add_argument("--n-cpu-moe", "-ncmoe", dest="n-cpu-moe", type=str, default=None)
    parser.add_argument("--cache-type-k", "-ctk", dest="cache-type-k", type=str, default=None)
    parser.add_argument("--cache-type-v", "-ctv", dest="cache-type-v", type=str, default=None)
    # fmt: on

    args, other = parser.parse_known_args()
    model = args.model
    ctx_size = args.ctx_size

    logger.info("正在寻找最优配置...")
    reader = GGUFReader(model)
    override_rules = get_override_rules(reader, ctx_size)

    final = ["--model", model] + ["--ctx-size", str(ctx_size)] + override_rules + other

    wrapper = LlamaServerWrapper()
    logger.info("正在启动 llama-server...")
    pid = wrapper.run(final)
    if pid < 0:
        logger.error("llama-server 启动失败")
        return 1
    try:
        logger.info("开始监听 http://127.0.0.1:8080 (key: sk-1234)")
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        logger.info("正在关闭...")
    finally:
        # 确保进程被停止并等待退出
        try:
            wrapper.stop()
        except Exception as e:
            logger.error(f"停止子进程时出错: {e}")

    return 0