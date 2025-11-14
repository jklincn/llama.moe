import csv
from datetime import datetime
import threading
import time
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt

from .bw_info import get_memory_bandwidth, get_pcie_bandwidth
from .cpu import CPUMonitor
from .gpu import GPUMonitor
from .memory import MemoryMonitor
from .pcie import PCIeMonitor


@lru_cache(maxsize=1)
def _cached_memory_bw():
    return get_memory_bandwidth()


@lru_cache(maxsize=1)
def _cached_pcie_bw():
    return get_pcie_bandwidth()


class SysMonitor:
    def __init__(self, interval=0.1):
        self.interval = interval

        self.cpu_monitor = CPUMonitor(interval=interval)
        self.memory_monitor = MemoryMonitor(
            interval=interval,
            pcm_binary="/home/lin/bs/llama.moe/build/pcm/bin/pcm-memory",
        )
        self.pcie_monitor = PCIeMonitor(
            interval=interval,
            pcm_binary="/home/lin/bs/llama.moe/build/pcm/bin/pcm-pcie",
        )
        self.gpu_monitor = GPUMonitor(interval=interval)

    def _parallel_call(self, funcs):
        """
        并行执行一组无参函数，阻塞到全部完成。
        funcs: [(name, callable), ...]
        """
        results = {}
        threads = []

        def wrap(name, fn):
            def _runner():
                results[name] = fn()

            return _runner

        for name, fn in funcs:
            t = threading.Thread(target=wrap(name, fn))
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

        return results

    def _save_results(self, results, filepath: Path):
        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["type", "ts", "value"])
            for key, series in results.items():
                for ts, val in series:
                    writer.writerow([key, ts.isoformat(), val])

    def start(self):
        self._parallel_call(
            [
                ("cpu_start", self.cpu_monitor.start),
                ("mem_start", self.memory_monitor.start),
                ("pcie_start", self.pcie_monitor.start),
                ("gpu_start", self.gpu_monitor.start),
            ]
        )
        time.sleep(3) # pcm 需要一些启动时间

    def end(self, save_path: Path | None = None):
        results = self._parallel_call(
            [
                ("cpu_util", self.cpu_monitor.end),
                ("mem_bw", self.memory_monitor.end),
                ("pcie_bw", self.pcie_monitor.end),
                ("gpu_data", self.gpu_monitor.end),
            ]
        )
        # 重新组合字典，将 GPU 数据拆分为三项
        gpu_records = results.pop("gpu_data", [])
        gpu_core_util = []
        gpu_mem_util = []
        gpu_vram_used = []

        for ts, core_util, mem_util, vram_used in gpu_records:
            gpu_core_util.append((ts, core_util))
            gpu_mem_util.append((ts, mem_util))
            gpu_vram_used.append((ts, vram_used))

        results.update(
            {
                "gpu_core_util": gpu_core_util,
                "gpu_mem_util": gpu_mem_util,
                "gpu_vram_used": gpu_vram_used,
            }
        )
        if save_path:
            self._save_results(results, save_path)
        return results


def draw(
    data, title="System Monitor", llama_server_log_path=None, output="sys_monitor.png"
):
    """
    绘制系统监控数据图表，横坐标从所有监控项都有数据的时间点开始。
    """
    # --- 1. 数据准备 ---
    max_memory_bw = _cached_memory_bw()
    max_pcie_bw = _cached_pcie_bw()
    
    # 原始数据
    cpu_util = data.get("cpu_util", [])
    mem_bw = data.get("mem_bw", [])
    pcie_bw = data.get("pcie_bw", [])
    gpu_core_util = data.get("gpu_core_util", [])
    gpu_mem_util = data.get("gpu_mem_util", [])

    # 转换为百分比
    mem_bw_percent = [(ts, (x / max_memory_bw) * 100) for ts, x in mem_bw]
    pcie_bw_percent = [(ts, (x / max_pcie_bw) * 100) for ts, x in pcie_bw]
    # CPU 由于没有使用超线程，因此认为 50% 就是满载
    cpu_util = [(ts, min(x * 2, 100)) for ts, x in cpu_util]

    # --- 关键修正步骤 1: 找到所有数据系列中最晚的起始时间 ---
    all_series = [cpu_util, mem_bw_percent, pcie_bw_percent, gpu_core_util, gpu_mem_util]
    
    # 收集所有非空数据系列的第一个时间戳
    start_times = [series[0][0] for series in all_series if series]

    # 如果没有任何数据，则直接返回
    if not start_times:
        print("警告：所有监控数据均为空，无法生成图表。")
        return
        
    # global_start_time 就是我们需要的“有效起点”
    global_start_time = max(start_times)

    # --- 关键修正步骤 2 & 3: 过滤数据并转换为相对秒数 ---
    # 定义一个辅助函数，用于过滤和转换
    def filter_and_rebase(series, start_dt):
        # 步骤2: 过滤掉早于 global_start_time 的数据
        filtered_series = [(ts, val) for ts, val in series if ts >= start_dt]
        # 步骤3: 时间戳重新归零
        rebased_series = [((ts - start_dt).total_seconds(), val) for ts, val in filtered_series]
        return rebased_series

    cpu_util_rel = filter_and_rebase(cpu_util, global_start_time)
    mem_bw_rel = filter_and_rebase(mem_bw_percent, global_start_time)
    pcie_bw_rel = filter_and_rebase(pcie_bw_percent, global_start_time)
    gpu_core_rel = filter_and_rebase(gpu_core_util, global_start_time)
    gpu_mem_rel = filter_and_rebase(gpu_mem_util, global_start_time)

    # --- 2. 开始绘图 ---
    plt.figure(figsize=(15, 8))

    plt.plot(*zip(*cpu_util_rel), label="CPU Utilization")
    plt.plot(*zip(*mem_bw_rel), label="Memory Bandwidth Utilization")
    plt.plot(*zip(*pcie_bw_rel), label="PCIe Bandwidth Utilization")
    plt.plot(*zip(*gpu_core_rel), label="GPU Core Utilization")
    plt.plot(*zip(*gpu_mem_rel), label="GPU Memory Utilization")

    # --- 3. 日志解析和标记逻辑 ---
    # 日志标记的时间计算也需要基于 global_start_time，以保证对齐
    if llama_server_log_path:
        prompt_ts_strings = []
        decode_ts_strings = []
        with open(llama_server_log_path, "r") as f:
            for line in f:
                if "prompt start:" in line:
                    timestamp_str = line.split("prompt start:")[1].strip()
                    prompt_ts_strings.append(timestamp_str)
                elif "decode start:" in line:
                    timestamp_str = line.split("decode start:")[1].strip()
                    decode_ts_strings.append(timestamp_str)
        
        time_format = "%Y-%m-%d %H:%M:%S.%f"
        prompt_times = [(datetime.strptime(ts, time_format) - global_start_time).total_seconds() for ts in prompt_ts_strings]
        decode_times = [(datetime.strptime(ts, time_format) - global_start_time).total_seconds() for ts in decode_ts_strings]
            
        for i, t in enumerate(prompt_times):
            plt.axvline(x=t, color="red", linestyle="--", label="Prompt Start" if i == 0 else None)
        for i, t in enumerate(decode_times):
            plt.axvline(x=t, color="green", linestyle="--", label="Decode Start" if i == 0 else None)

    # --- 4. 设置图表属性 ---
    plt.title(title, fontsize=16)
    plt.xlabel("Time (s)")
    plt.ylabel("Utilization (%)")
    plt.ylim(0, 105)
    plt.xlim(left=0)
    plt.legend(loc="upper left")
    plt.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(output)
    plt.close()


if __name__ == "__main__":
    monitor = SysMonitor(interval=0.1)
    monitor.start()
    time.sleep(20)
    results = monitor.end()
    for ts, util in results["cpu_util"]:
        print(f"{ts:%Y-%m-%d %H:%M:%S.%f} CPU {util:.2f} %")
    for ts, mem in results["mem_bw"]:
        print(f"{ts:%Y-%m-%d %H:%M:%S.%f} Memory {mem / 1e3:.2f} GB/s")
    for ts, pcie in results["pcie_bw"]:
        print(f"{ts:%Y-%m-%d %H:%M:%S.%f} PCIe {pcie:.2f} GB/s")
    for ts, util in results["gpu_core_util"]:
        print(f"{ts:%Y-%m-%d %H:%M:%S.%f} GPU Core {util:.2f} %")
    for ts, util in results["gpu_mem_util"]:
        print(f"{ts:%Y-%m-%d %H:%M:%S.%f} GPU Memory {util:.2f} %")
    for ts, used in results["gpu_vram_used"]:
        print(f"{ts:%Y-%m-%d %H:%M:%S.%f} GPU VRAM Used {(used / 1e9):.2f} GiB")
    draw(results)
