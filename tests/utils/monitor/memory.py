import csv
import os
import signal
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple


class MemoryMonitor:
    def __init__(
        self,
        interval: float = 0.1,
        csv_path: str = "pcm_memory.csv",
        pcm_binary: str = "pcm-memory",
        use_sudo: bool = True,
    ):
        self.csv_path = Path(csv_path)
        self.interval = interval
        self.pcm_binary = pcm_binary
        self.use_sudo = use_sudo

        self._proc: Optional[subprocess.Popen] = None
        self.data: List[Tuple[datetime, float]] = []

    def start(self):
        cmd = []
        if self.use_sudo:
            cmd.append("sudo")
        cmd.extend(
            [
                self.pcm_binary,
                str(self.interval),
                f"-csv={str(self.csv_path)}",
            ]
        )

        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )

    def _stop_process(self):
        pgid = os.getpgid(self._proc.pid)
        os.killpg(pgid, signal.SIGINT)
        self._proc.wait(timeout=3)

    def _parse_csv(self) -> List[Tuple[datetime, float]]:
        data: List[Tuple[datetime, float]] = []

        with self.csv_path.open("r", newline="") as f:
            reader = csv.reader(f)
            _ = next(reader)  # 跳过第一行
            header = next(reader)  # 字段名
            for row in reader:
                dt_str = f"{row[0].strip()} {row[1].strip()}"
                mem_str = row[len(header) - 1].strip()
                ts = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S.%f")
                mem_val = float(mem_str)
                data.append((ts, mem_val))

        return data

    def end(self) -> List[Tuple[datetime, float]]:
        self._stop_process()
        time.sleep(0.1)  # 确保内核 flush 写完文件
        self.data = self._parse_csv()
        return self.data
