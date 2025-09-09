import argparse
import sys
from pathlib import Path
from gguf import GGUFReader
import uvicorn

from .override import get_override_rules
from .proxy import app
from .wrapper import LlamaServerWrapper


def main():
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

    print("[main] 正在寻找最优配置...")
    reader = GGUFReader(model)
    override_rules = get_override_rules(reader, ctx_size)

    final = ["--model", model] + ["--ctx-size", str(ctx_size)] + override_rules + other

    wrapper = LlamaServerWrapper()
    print("[main] 正在启动 llama-server...")
    pid = wrapper.run(final)
    if pid < 0:
        print("[main] llama-server 启动失败")
        return 1

    try:
        uvicorn.run(app, log_level="info")
    except KeyboardInterrupt:
        print("[main] 捕获到 Ctrl+C, 正在停止子进程...")
    finally:
        # 确保进程被停止并等待退出
        try:
            wrapper.stop()
        except Exception as e:
            print(f"[main] 停止子进程时出错: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
