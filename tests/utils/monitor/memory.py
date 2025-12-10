import csv
import os
import signal
import subprocess
import time
import io
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

        if not self.csv_path.exists():
            return data

        # 读取文件内容并清洗 NUL 字节，防止 pcm 异常退出导致文件末尾包含大量 NUL
        try:
            with self.csv_path.open("r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        except Exception:
            return data

        content = content.replace('\x00', '')
        
        f_obj = io.StringIO(content)
        reader = csv.reader(f_obj)

        try:
            iterator = iter(reader)
            try:
                _ = next(iterator)  # 跳过第一行
                header = next(iterator)  # 字段名
            except StopIteration:
                return data

            for row in iterator:
                if not row:
                    continue
                
                try:
                    # 简单的完整性检查
                    if len(row) < 2: 
                        continue

                    dt_str = f"{row[0].strip()} {row[1].strip()}"
                    
                    # 原始逻辑：取最后一列作为 Memory Bandwidth
                    target_idx = len(header) - 1
                    if target_idx >= len(row):
                        continue
                    
                    mem_str = row[target_idx].strip()
                    
                    ts = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S.%f")
                    mem_val = float(mem_str)
                    data.append((ts, mem_val))
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
