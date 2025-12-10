import csv
import os
import signal
import subprocess
import time
import io
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple


class PCIeMonitor:
    """
    采样 PCIe 带宽（GB/s）
    返回序列：List[Tuple[datetime, float]]
    """
    def __init__(
        self,
        interval: float = 0.1,
        csv_path: str = "pcm_pcie.csv",
        pcm_binary: str = "pcm-pcie",
        use_sudo: bool = True,
    ):
        self.csv_path = Path(csv_path)
        self.interval = interval
        self.pcm_binary = pcm_binary
        self.use_sudo = use_sudo

        self._proc: Optional[subprocess.Popen] = None
        self.data: List[Tuple[datetime, float]] = []

    def start(self):
        if self.csv_path.exists():
            try:
                self.csv_path.unlink()
            except OSError:
                pass

        cmd = []
        if self.use_sudo:
            cmd.append("sudo")
        cmd.extend(
            [self.pcm_binary, str(self.interval), f"-csv={str(self.csv_path)}", "-B"]
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

        if not self.csv_path.exists():
            return data

        try:
            with self.csv_path.open("r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        except Exception:
            return data

        content = content.replace('\x00', '')
        f_obj = io.StringIO(content)
        reader = csv.reader(f_obj)

        try:
            for row in reader:
                if not row:
                    continue
                try:
                    if row[0].strip() == "Date":
                        continue
                    
                    # 确保行有足够的数据
                    if len(row) < 2:
                        continue

                    dt_str = f"{row[0].strip()} {row[1].strip()}"
                    ts = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S.%f")
                    
                    pcie_rd = float(row[-2])
                    pcie_wr = float(row[-1])
                    # pcm-pcie -B outputs Bytes/sec, so we just convert to GB/s
                    data.append((ts, max(pcie_rd, pcie_wr) / 1e9))
                except (ValueError, IndexError):
                    continue
        except csv.Error:
            pass

        return data

    def end(self) -> List[Tuple[datetime, float]]:
        self._stop_process()
        time.sleep(0.1)  # 确保内核 flush 写完文件
        self.data = self._parse_csv()
        return self.data
