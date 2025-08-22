import argparse
from pathlib import Path
import sys

from .wrapper import LlamaServerWrapper
from .override import get_override_rules


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
    final = ["--model", model] + ["--ctx-size", str(ctx_size)] + get_override_rules(model, ctx_size) + other

    wrapper = LlamaServerWrapper()
    return wrapper.run(final)


if __name__ == "__main__":
    sys.exit(main())
