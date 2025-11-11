from typing import List, Tuple
from datetime import datetime
import threading
import time

import pynvml as nvml


class GPUMonitor:
    """
    采样 GPU 三项指标：
      1) 计算核心利用率（utilization.gpu，%）
      2) 内存控制器利用率（utilization.memory，%）
      3) 已用显存（bytes）
    返回序列：List[Tuple[datetime, int, int, int]]
    """

    def __init__(self, interval: float = 0.1, device_index: int = 0):
        self.interval = float(interval)
        self.device_index = int(device_index)

        self.gpu_data = []
        self._running = False
        self._thread = None

        self._handle = None

    def _nvml_init(self):
        nvml.nvmlInit()
        count = nvml.nvmlDeviceGetCount()
        if count <= self.device_index:
            nvml.nvmlShutdown()
            raise RuntimeError(
                f"GPU index {self.device_index} out of range (found {count} device(s))"
            )
        self._handle = nvml.nvmlDeviceGetHandleByIndex(self.device_index)
        self._nvml_inited = True

    def _nvml_shutdown(self):
        nvml.nvmlShutdown()
        self._nvml_inited = False
        self._handle = None

    def _loop(self):
        next_time = time.time()
        while self._running:
            ts = datetime.now()
            util = nvml.nvmlDeviceGetUtilizationRates(self._handle)
            meminfo = nvml.nvmlDeviceGetMemoryInfo(self._handle)
            self.gpu_data.append((ts, util.gpu, util.memory, meminfo.used))
            next_time += self.interval
            time.sleep(max(0.0, next_time - time.time()))

    def start(self):
        self._nvml_init()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def end(self) -> List[Tuple[datetime, int, int, int]]:
        self._running = False
        self._thread.join()
        self._nvml_shutdown()
        return self.gpu_data
