import time
from gguf import GGUFReader
import logging
from .override import get_override_rules
from .wrapper import LlamaServerWrapper
import os

logger = logging.getLogger("main")

# sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
def drop_file_cache(path: str):
    fd = os.open(path, os.O_RDONLY)
    try:
        size = os.path.getsize(path)
        os.posix_fadvise(fd, 0, size, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)

def check_numa(model) -> None | tuple[list[str], list[str]]:
    path = "/sys/devices/system/node/has_cpu"
    try:
        with open(path, "r") as f:
            content = f.read().strip()
    except FileNotFoundError:
        logger.debug("未检测到 NUMA 信息，跳过 numactl 配置")
        return None

    if content == "0":
        return None
    elif content == "0-1":
        # todo: 更准确的确认物理核index
        cpu_count = os.cpu_count() or 1
        half = max(cpu_count // 2, 1)
        numactl_cmd = [
            "numactl",
            f"--physcpubind=0-{half - 1}",
            "--interleave=all",
        ]
        numa_args = ["--numa", "numactl", "-t", str(half)]
        drop_file_cache(model)
        return numactl_cmd, numa_args
    else:
        raise ValueError("暂不支持大于2个NUMA节点的系统")


def check_model(model_path: str):
    if not os.path.isfile(model_path):
        msg = f"模型文件 {model_path} 不存在"
        logger.error(msg)
        raise FileNotFoundError(msg)
    if "00001" in model_path:
        msg = "当前仅支持单文件 GGUF 模型，可以参考 scripts/gguf_merge.sh 进行合并"
        logger.error(msg)
        raise ValueError(msg)


def run(args, other):
    model, ctx_size, kv_offload = args.model, args.ctx_size, not args.no_kv_offload

    check_model(model)

    logger.info("正在寻找最优配置...")
    reader = GGUFReader(model)
    ot_args = get_override_rules(reader, ctx_size, kv_offload)

    numa_result = check_numa(model)
    if numa_result is None:
        numactl_cmd = None
        numa_args = []
    else:
        numactl_cmd, numa_args = numa_result

    final_args = (
        ["--model", model] + ["--ctx-size", str(ctx_size)] + ot_args + other + numa_args
    )
    wrapper = LlamaServerWrapper(numactl=numactl_cmd)
    try:
        logger.info("正在启动 llama-server...")
        pid = wrapper.start(final_args, timeout=3600)
        if pid < 0:
            logger.error("llama-server 启动失败")
            raise RuntimeError("llama-server 启动失败")
        logger.info("启动成功, 开始监听 http://127.0.0.1:8080 (key: sk-1234)")
        while wrapper.is_running():
            time.sleep(10)
    except KeyboardInterrupt:
        logger.info("正在关闭...")
    finally:
        try:
            wrapper.stop()
        except Exception:
            logger.exception("停止子进程时出错")
