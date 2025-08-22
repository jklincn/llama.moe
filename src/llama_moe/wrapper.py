import os
import subprocess
import signal
import threading
import time
from pathlib import Path
from typing import Optional


class LlamaServerWrapper:
    def __init__(self):
        self.process: Optional[subprocess.Popen] = None
        self.shutdown_requested = False
        self.signal_forwarded = False
        self.output_thread: Optional[threading.Thread] = None

    # 仅将收到的信号转发给子进程组（POSIX）
    def _send_to_child_group(self, signum: int):
        if not self.process or self.process.poll() is not None:
            return
        try:
            pgid = os.getpgid(self.process.pid)
            os.killpg(pgid, signum)
        except ProcessLookupError:
            pass
        except Exception as e:
            print(f"转发信号失败：{e}")

    # 信号处理器（只转发一次）
    def signal_handler(self, signum, frame):
        self.shutdown_requested = True

        if self.signal_forwarded:
            return

        if self.process and self.process.poll() is None:
            self._send_to_child_group(signum)
            self.signal_forwarded = True
        else:
            print("子进程不存在或已退出。")

    # 持续读取输出（不做“退出后剩余输出”清空）
    def read_output_continuously(self, process: subprocess.Popen, log_file):
        try:
            while True:
                if process.poll() is not None:
                    break  # 子进程结束就退出线程（不再额外清空）
                line = process.stdout.readline()
                if line:
                    print(line.rstrip())
                    log_file.write(line)
                    log_file.flush()
                else:
                    time.sleep(0.01)
        except Exception as e:
            print(f"读取输出时出错: {e}")

    # 主运行逻辑
    def run(self, argv: list[str]) -> int:
        script_dir = Path(__file__).parent.absolute()
        root_dir = script_dir.parent.parent
        server_path = root_dir / "llama.cpp/build/bin/llama-server"
        log_path = root_dir / "llama-server.log"

        cmd = [str(server_path)] + argv

        print(f"启动命令: {' '.join(cmd)}")
        print(f"日志文件: {log_path}")

        # 注册信号处理器
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        if hasattr(signal, "SIGHUP"):
            try:
                signal.signal(signal.SIGHUP, self.signal_handler)
            except Exception:
                pass

        try:
            with open(log_path, "w", buffering=1) as log_file:
                popen_kwargs = dict(
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    bufsize=1,
                    start_new_session=True,  # 新会话/新进程组
                )

                # 启动子进程
                self.process = subprocess.Popen(cmd, **popen_kwargs)

                # 启动输出读取线程
                self.output_thread = threading.Thread(
                    target=self.read_output_continuously,
                    args=(self.process, log_file),
                    daemon=True,
                )
                self.output_thread.start()

                # 主循环：直到子进程退出或收到关闭请求
                while self.process.poll() is None and not self.shutdown_requested:
                    time.sleep(0.1)

                # 收到关闭请求后，等待其自行清理
                if self.shutdown_requested and self.process.poll() is None:
                    self.process.wait()

                # 记录返回码
                return_code = self.process.returncode

                # 关闭 stdout 以便读取线程尽快退出，然后静默 join
                try:
                    if self.process.stdout:
                        self.process.stdout.close()
                except Exception:
                    pass
                if self.output_thread:
                    self.output_thread.join(timeout=5)

                return return_code if return_code is not None else 0

        except FileNotFoundError:
            print(f"错误: 找不到 llama-server 程序: {server_path}")
            return 1
        except Exception as e:
            print(f"运行时错误: {e}")
            return 1
