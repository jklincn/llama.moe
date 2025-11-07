import logging
import threading
import time
from pathlib import Path

import numpy as np
import pynvml as nvml

logger = logging.getLogger("gpu_recorder")

# 全局状态变量
_records = []
_thread = None
_stop_event = threading.Event()
_gpu_handle = None
_nvml_initialized = False


def start(interval: float = 0.1) -> None:
    """开始GPU监控

    Args:
        interval: 采样间隔（秒），默认0.1秒

    Raises:
        RuntimeError: 当GPU监控已在运行或无GPU设备时
    """
    global _thread, _stop_event, _records

    if _thread is not None:
        logger.warning("GPU监控已在运行中")
        return

    # 初始化
    _init_nvml()
    _records = []
    _stop_event = threading.Event()

    # 启动监控线程
    _thread = threading.Thread(
        target=_monitor_loop, args=(interval,), name="GPURecorder", daemon=True
    )
    _thread.start()


def finish(filepath: str):
    global _thread, _stop_event, _records

    if _thread is None:
        logger.warning("GPU监控未启动")
        return 0.0

    # 停止监控线程
    _stop_event.set()
    _thread.join(timeout=5.0)  # 添加超时防止无限等待
    _thread = None

    if not _records:
        logger.warning("未记录到任何数据")
        return 0.0

    # 过滤无效数据
    filtered = [row for row in _records if not (row[0] == 0 and row[1] == 0)]

    # 确保目录存在
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    # 保存为NumPy文件
    gpu_utils = np.array([row[0] for row in filtered], dtype=np.int64)
    mem_utils = np.array([row[1] for row in filtered], dtype=np.int64)
    mem_used = np.array([row[2] for row in filtered], dtype=np.int64)

    np.savez_compressed(
        filepath,
        gpu_utils=gpu_utils,
        mem_utils=mem_utils,
        mem_used=mem_used,
        mem_total=_get_gpu_total_memory(),
    )

    _shutdown_nvml()


def load_gpu_info(filepath: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    返回 [计算核心利用率, 显存带宽利用率, 已用显存(Bytes), 总显存(Bytes)]
    """

    if not Path(filepath).exists():
        raise FileNotFoundError(f"文件不存在: {filepath}")

    data = np.load(filepath)

    for key in ("gpu_utils", "mem_utils", "mem_used", "mem_total"):
        if key not in data:
            raise KeyError(f"文件格式不正确，缺少必要的字段: {key}")

    return (
        data["gpu_utils"],
        data["mem_utils"],
        data["mem_used"],
        data["mem_total"],
    )


# 内部函数
def _init_nvml() -> None:
    global _gpu_handle, _nvml_initialized

    if _nvml_initialized:
        return

    try:
        nvml.nvmlInit()
        device_count = nvml.nvmlDeviceGetCount()

        if device_count < 1:
            raise RuntimeError("未检测到GPU设备")

        _gpu_handle = nvml.nvmlDeviceGetHandleByIndex(0)
        _nvml_initialized = True

    except Exception as e:
        raise RuntimeError(f"初始化NVML失败: {e}")


def _shutdown_nvml() -> None:
    """关闭NVML"""
    global _gpu_handle, _nvml_initialized

    if _nvml_initialized:
        nvml.nvmlShutdown()
        _nvml_initialized = False
        _gpu_handle = None


def _get_gpu_info() -> tuple[int, int, int]:
    """
    返回 [计算核心利用率, 显存带宽利用率, 已用显存(Bytes)]
    """
    global _gpu_handle

    try:
        utilization = nvml.nvmlDeviceGetUtilizationRates(_gpu_handle)
        memory_info = nvml.nvmlDeviceGetMemoryInfo(_gpu_handle)
        return utilization.gpu, utilization.memory, memory_info.used

    except Exception as e:
        raise RuntimeError(f"获取GPU实时性能信息失败: {e}")


def _get_gpu_total_memory() -> int:
    global _gpu_handle

    try:
        memory_info = nvml.nvmlDeviceGetMemoryInfo(_gpu_handle)
        return memory_info.total

    except Exception as e:
        raise RuntimeError(f"获取GPU总内存失败: {e}")


def _monitor_loop(interval: float) -> None:
    """监控循环

    Args:
        interval: 采样间隔
    """
    global _records, _stop_event

    last_time = time.perf_counter()
    while not _stop_event.is_set():
        # 高精度控制采样间隔
        current_time = time.perf_counter()
        elapsed = current_time - last_time
        if elapsed < interval:
            remaining = interval - elapsed
            if _stop_event.wait(remaining):
                break
        last_time = time.perf_counter()
        # 采样
        _records.append(_get_gpu_info())


# 使用示例
if __name__ == "__main__":
    print("开始GPU监控测试...")
    start()
    print("模拟5秒GPU工作...")
    time.sleep(5)
    finish("test_gpu_info.npz")
    utils, mems, total = load_gpu_info("test_gpu_info.npz")
    print(f"数据点: {len(utils)}，最高利用率: {np.max(utils)}%，总显存: {total} Bytes")
