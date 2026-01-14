import threading
import time
from gguf import GGUFReader
import logging

from llama_moe.tracker import MetricsTracker
from .override import get_override_rules
from .wrapper import LlamaServerWrapper
from .pruner import prune_model_with_report
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
    logger.info(f"使用模型: {model_path}")


def run(args, other):
    model, ctx_size, kv_offload = args.model, args.ctx_size, not args.no_kv_offload

    check_model(model)

    logger.info("正在寻找最优配置...")
    reader = GGUFReader(model)
    ot_args = get_override_rules(reader, ctx_size, kv_offload)

    numa_result = check_numa(model)
    numactl_cmd, numa_args = numa_result if numa_result else (None, [])

    if "--metrics" not in other:
        other.append("--metrics")
    if os.getenv("LLAMA_MOE_DEBUG") == "1":
        other.append("-v")

    current_model = model
    pruning_done = False
    threshold = args.prune_threshold
    while True:
        final_args = (
            ["--model", current_model]
            + ["--ctx-size", str(ctx_size)]
            + ot_args
            + other
            + numa_args
        )
        wrapper = LlamaServerWrapper(numactl=numactl_cmd)
        tracker = None
        try:
            logger.info("正在启动 llama-server...")

            pid = wrapper.start(final_args, timeout=3600)
            if pid < 0:
                logger.error("llama-server 启动失败")
                raise RuntimeError("llama-server 启动失败")

            logger.info("启动成功, 开始监听 http://127.0.0.1:8080 (key: sk-1234)")
            if not pruning_done and threshold > 0:
                stop_event = threading.Event()

                def on_threshold(tokens):
                    logger.info(f"触发剪枝阈值 ({tokens} tokens). 正在停止服务器...")
                    stop_event.set()

                tracker = MetricsTracker(
                    threshold=threshold, on_threshold_reached=on_threshold
                )
                tracker.start()

                # 主循环等待
                while wrapper.is_running() and not stop_event.is_set():
                    time.sleep(5)

                if stop_event.is_set():
                    # 阈值触发，停止服务器
                    wrapper.stop()
                    tracker.stop()
                    time.sleep(3)  # 等待激活文件保存
                    # 执行剪枝
                    logger.info("开始执行专家裁剪...")
                    # 假设 expert_activations.csv 在当前目录
                    csv_path = "expert_activations.csv"
                    if os.path.exists(csv_path):
                        try:
                            # 使用 coverage 模式进行剪枝
                            new_model_path = prune_model_with_report(
                                current_model, 
                                csv_path, 
                                method='coverage', 
                                threshold=args.prune_coverage
                            )
                            current_model = new_model_path
                            pruning_done = True
                            logger.info(f"剪枝完成，新模型路径: {current_model}")
                        except Exception as e:
                            logger.error(f"剪枝失败: {e}")
                            break
                    else:
                        logger.error("未找到激活报告，无法剪枝")
                        break

                    continue  # 重启循环，使用新模型
            else:
                # 普通运行模式
                while wrapper.is_running():
                    time.sleep(10)

            break  # 正常退出循环
        except KeyboardInterrupt:
            logger.info("正在关闭...")
            break
        finally:
            if tracker:
                tracker.stop()
            try:
                wrapper.stop()
            except Exception:
                logger.exception("停止子进程时出错")
