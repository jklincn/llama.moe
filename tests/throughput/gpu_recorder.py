import threading
import time
from pathlib import Path

import numpy as np
import pynvml as nvml

# 全局状态变量
_records = []
_thread = None
_stop_event = threading.Event()
_start_time = None
_started_recording = False
_gpu_handle = None
_nvml_initialized = False


def start(interval: float = 0.1) -> None:
    """开始GPU监控

    Args:
        interval: 采样间隔（秒），默认0.1秒

    Raises:
        RuntimeError: 当GPU监控已在运行或无GPU设备时
    """
    global _thread, _stop_event, _records, _start_time, _started_recording

    if _thread is not None:
        print("GPU监控已在运行中")
        return

    # 初始化
    _init_nvml()
    _records = []
    _stop_event = threading.Event()
    _start_time = None
    _started_recording = False

    # 启动监控线程
    _thread = threading.Thread(
        target=_monitor_loop, args=(interval,), name="GPURecorder", daemon=True
    )
    _thread.start()
    print(f"GPU监控已启动，采样间隔: {interval}s")


def finish(filepath: str) -> float:
    """停止监控并保存数据到NumPy文件

    Args:
        filepath: 保存文件路径（.npz格式）

    Returns:
        平均GPU利用率，如果没有数据则返回0.0
    """
    global _thread, _stop_event, _records

    if _thread is None:
        print("GPU监控未启动")
        return 0.0

    # 停止监控线程
    _stop_event.set()
    _thread.join(timeout=5.0)  # 添加超时防止无限等待
    _thread = None

    _shutdown_nvml()

    if not _records:
        print("未记录到任何数据")
        return 0.0

    # 确保目录存在
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    # 保存为NumPy文件
    data_array = np.array(_records)

    np.savez_compressed(
        filepath, time_seconds=data_array[:, 0], gpu_utilization=data_array[:, 1]
    )

    # 计算统计信息
    avg_util = np.mean(data_array[:, 1])
    max_util = np.max(data_array[:, 1])
    min_util = np.min(data_array[:, 1])

    print(f"GPU监控已停止，数据已保存到: {filepath}")
    print(
        f"监控统计 - 平均: {avg_util:.2f}%, 最高: {max_util:.2f}%, 最低: {min_util:.2f}%"
    )

    return avg_util


def load_data(filepath: str) -> tuple[np.ndarray, np.ndarray]:
    """加载GPU监控数据

    Args:
        filepath: .npz文件路径

    Returns:
        (time_seconds, gpu_utilization) 两个NumPy数组

    Raises:
        FileNotFoundError: 文件不存在时
        KeyError: 文件格式不正确时
    """
    if not Path(filepath).exists():
        raise FileNotFoundError(f"文件不存在: {filepath}")

    data = np.load(filepath)

    if "time_seconds" not in data or "gpu_utilization" not in data:
        raise KeyError("文件格式不正确，缺少必要的数据字段")

    return data["time_seconds"], data["gpu_utilization"]


def get_memory_usage() -> tuple[float, float]:
    """获取GPU显存使用信息

    Returns:
        (memory_used_mb, memory_usage_percentage) 元组
        - memory_used_mb: 已使用显存大小（MB）
        - memory_usage_percentage: 显存使用百分比

    Raises:
        RuntimeError: 当NVML未初始化或获取失败时
    """
    global _gpu_handle, _nvml_initialized

    if not _nvml_initialized:
        # 如果NVML未初始化，先初始化
        _init_nvml()

    try:
        memory_info = nvml.nvmlDeviceGetMemoryInfo(_gpu_handle)

        # 转换为MB
        memory_used_mb = memory_info.used / (1024 * 1024)

        # 计算使用百分比
        memory_usage_percentage = (memory_info.used / memory_info.total) * 100

        return memory_used_mb, memory_usage_percentage

    except Exception as e:
        raise RuntimeError(f"获取GPU显存信息失败: {e}")


# 内部函数
def _init_nvml() -> None:
    """初始化NVML

    Raises:
        RuntimeError: 当未检测到GPU设备时
    """
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
        try:
            nvml.nvmlShutdown()
        except Exception:
            pass  # 忽略关闭时的错误
        finally:
            _nvml_initialized = False
            _gpu_handle = None


def _get_gpu_utilization() -> int:
    """获取GPU利用率

    Returns:
        GPU利用率百分比

    Raises:
        RuntimeError: 获取失败时
    """
    global _gpu_handle

    try:
        utilization = nvml.nvmlDeviceGetUtilizationRates(_gpu_handle)
        return utilization.gpu
    except Exception as e:
        raise RuntimeError(f"获取GPU利用率失败: {e}")


def _monitor_loop(interval: float) -> None:
    """监控循环

    Args:
        interval: 采样间隔
    """
    global _records, _stop_event, _start_time, _started_recording

    last_time = time.perf_counter()

    while not _stop_event.is_set():
        # 控制采样间隔
        current_time = time.perf_counter()
        elapsed = current_time - last_time

        if elapsed < interval:
            remaining = interval - elapsed
            if _stop_event.wait(remaining):
                break

        last_time = time.perf_counter()

        try:
            # 获取GPU利用率
            gpu_util = _get_gpu_utilization()

            # 如果还没开始记录，等待GPU开始工作
            if not _started_recording:
                if gpu_util > 0:
                    _started_recording = True
                    _start_time = time.time()  # 设置开始时间为第一次GPU使用时
                else:
                    continue

            # 记录数据：[相对时间(秒), GPU利用率(%)]
            relative_time = time.time() - _start_time
            _records.append([relative_time, gpu_util])

        except Exception as e:
            print(f"监控循环出错: {e}")
            break


# 使用示例
if __name__ == "__main__":
    print("开始GPU监控测试...")
    start()  # 开始监控

    # 测试显存使用情况
    try:
        memory_used, memory_percent = get_memory_usage()
        print(f"当前显存使用: {memory_used:.2f} MiB ({memory_percent:.2f}%)")
    except Exception as e:
        print(f"获取显存信息失败: {e}")
    # 模拟GPU工作
    print("模拟5秒GPU工作...")
    time.sleep(5)

    avg_util = finish("test_gpu_utilization.npz")  # 停止监控并保存

    # 加载数据进行分析
    if avg_util > 0:
        try:
            times, utilizations = load_data("test_gpu_utilization.npz")
            print(f"总数据点: {len(times)}")
            print(f"监控时长: {times[-1] - times[0]:.2f}秒")
            print(f"最高利用率: {np.max(utilizations):.2f}%")
            print(f"最低利用率: {np.min(utilizations):.2f}%")

            # 再次检查显存使用情况
            try:
                memory_used, memory_percent = get_memory_usage()
                print(
                    f"监控结束后显存使用: {memory_used:.2f} MB ({memory_percent:.2f}%)"
                )
            except Exception as e:
                print(f"获取显存信息失败: {e}")

        except Exception as e:
            print(f"加载数据失败: {e}")
    else:
        print("测试完成，但未获取到有效数据")
