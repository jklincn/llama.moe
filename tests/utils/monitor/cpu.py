from typing import List, Tuple
import psutil
import time
import threading
from datetime import datetime


class CPUMonitor:
    """
    采样 CPU 利用率（%）
    返回序列：List[Tuple[datetime, float]]
    """

    def __init__(self, interval=0.1):
        self.interval = interval
        self.cpu_data = []
        self._running = False
        self._thread = None

    def _loop(self):
        psutil.cpu_percent(interval=None)  # 预热一次
        next_time = time.time()
        while self._running:
            ts = datetime.now()
            usage = psutil.cpu_percent(interval=self.interval)
            self.cpu_data.append((ts, usage))
            next_time += self.interval
            time.sleep(max(0, next_time - time.time()))

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def end(self) -> List[Tuple[datetime, float]]:
        self._running = False
        self._thread.join()
        return self.cpu_data
