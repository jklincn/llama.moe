import argparse
import logging
import os
import sys
from pathlib import Path

from .core import run


def setup_logging():
    LOG_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
    DATEFMT = "%Y-%m-%d %H:%M:%S"
    logging.basicConfig(handlers=[], force=True)
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(logging.Formatter(LOG_FORMAT, DATEFMT))
    root = logging.getLogger()
    if os.getenv("LLAMA_MOE_DEBUG") == "1":
        root.setLevel(logging.DEBUG)
    else:
        root.setLevel(logging.INFO)
    root.addHandler(handler)
    for noisy in ["urllib3"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)


def build_parser():
    # fmt: off
    parser = argparse.ArgumentParser(description="llama.moe", add_help=False)
    parser.add_argument("--ctx-size", "-c", dest="ctx_size", type=int, default=4096)
    parser.add_argument("--model", "-m", dest="model", type=lambda s: str(Path(s).expanduser()), default=None)
    parser.add_argument("--no-kv-offload", "-nkvo", dest="no_kv_offload", action="store_true")
    # llama_moe pruning parameters
    parser.add_argument("--prune-threshold", "-pt", dest="prune_threshold", type=int, default=sys.maxsize, help="Token threshold for pruning")
    parser.add_argument("--prune-coverage", "-pc", dest="prune_coverage", type=float, default=90.0, help="Coverage percentage for pruning (0-100)")
    # discard the following parameters, llama_moe will add them automatically
    parser.add_argument("--n-gpu-layers", "--gpu-layers", "-ngl", dest="n-gpu-layers", type=str, default=None)
    parser.add_argument("--override-tensor", "-ot", dest="override-tensor", type=str, default=None)
    parser.add_argument("--cpu-moe", "-cmoe", dest="cpu-moe", type=str, default=None)
    parser.add_argument("--n-cpu-moe", "-ncmoe", dest="n-cpu-moe", type=str, default=None)
    parser.add_argument("--cache-type-k", "-ctk", dest="cache-type-k", type=str, default=None)
    parser.add_argument("--cache-type-v", "-ctv", dest="cache-type-v", type=str, default=None)
    parser.add_argument("--numa", dest="numa", type=str, default=None)
    parser.add_argument("--metrics", dest="metrics", action="store_true")
    # fmt: on
    return parser


def main():
    setup_logging()
    parser = build_parser()
    args, other = parser.parse_known_args()
    run(args, other)
