import os
import subprocess
import threading
import time
import signal
from pathlib import Path
from typing import Optional


class LlamaServerWrapper:
    def __init__(self, log_filename: str = "llama-server.log"):
        self.process: Optional[subprocess.Popen] = None
        self.output_thread: Optional[threading.Thread] = None
        self._log_file: Optional[object] = None
        self.log_path = Path.cwd() / log_filename

    def _read_output_continuously(self):
        try:
            while True:
                if self.process is None or self.process.poll() is not None:
                    break
                line = self.process.stdout.readline()
                if line:
                    try:
                        self._log_file.write(line)
                        self._log_file.flush()
                    except Exception:
                        pass
                else:
                    time.sleep(0.01)
        except Exception as e:
            print(f"[wrapper] 读取输出时出错: {e}")
        finally:
            try:
                if self.process and self.process.stdout:
                    self.process.stdout.close()
            except Exception:
                pass

    def run(self, argv: list[str]) -> int:
        if self.process and self.process.poll() is None:
            raise RuntimeError("llama-server 已在运行，请先 stop().")

        default_path = Path.cwd() / "llama.cpp" / "build" / "bin" / "llama-server"
        cmd = [str(default_path)] + argv

        print(f"[wrapper] 启动命令: {' '.join(cmd)}")
        print(f"[wrapper] 日志文件: {self.log_path}")

        # 打开日志文件（保持句柄，用于持续写入）
        self._log_file = open(self.log_path, "w", buffering=1, encoding="utf-8")

        popen_kwargs = dict(
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
            start_new_session=True,
        )

        try:
            self.process = subprocess.Popen(cmd, **popen_kwargs)
        except FileNotFoundError:
            print(f"[wrapper] 错误: 找不到可执行文件: {cmd[0]}")
            self._log_file.close()
            self._log_file = None
            return -1
        except Exception as e:
            print(f"[wrapper] 启动失败: {e}")
            self._log_file.close()
            self._log_file = None
            return -1

        # 后台线程持续写日志
        self.output_thread = threading.Thread(
            target=self._read_output_continuously,
            daemon=True,
        )
        self.output_thread.start()

        # 等待启动成功标志，超时时间60秒
        success = False
        timeout = 60
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                # 检查进程是否提前退出
                if self.process.poll() is not None:
                    print("[wrapper] 错误: 进程提前退出，请检查日志。")
                    break

                # 检查日志文件是否包含启动成功标志
                with open(self.log_path, "r", encoding="utf-8") as log:
                    for line in log:
                        if "server is listening on" in line:
                            success = True
                            break
                if success:
                    break
                time.sleep(0.5)
            except Exception as e:
                print(f"[wrapper] 检查启动状态时出错: {e}")
                break

        if not success:
            print(f"[wrapper] 启动失败: {timeout} 秒内未检测到启动成功标志")
            self.stop()
            return -1

        print("[wrapper] llama-server 启动成功")
        return self.process.pid

    def stop(self, signum: int = signal.SIGTERM) -> None:
        """
        发送信号给子进程组，并等待其退出
        """
        if not self.process:
            print("[wrapper] stop: 未发现子进程。")
            return
        if self.process.poll() is not None:
            print("[wrapper] stop: 子进程已退出。")
            return

        try:
            pgid = os.getpgid(self.process.pid)
            os.killpg(pgid, signum)
        except ProcessLookupError:
            print("[wrapper] stop: 进程不存在或已退出。")
        except Exception as e:
            print(f"[wrapper] stop: 发送信号失败：{e}")

        # 等待退出
        try:
            rc = self.process.wait(timeout=10)
            print(f"[wrapper] stop: llama-server 已退出，返回码 {rc}")
        except TimeoutError:
            print("[wrapper] stop: 等待子进程退出超时")
        finally:
            if self.output_thread and self.output_thread.is_alive():
                self.output_thread.join(timeout=5)
            if self._log_file:
                self._log_file.close()
                self._log_file = None
            self.process = None
