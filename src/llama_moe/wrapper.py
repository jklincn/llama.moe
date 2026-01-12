import logging
import os
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional, Sequence

logger = logging.getLogger("wrapper")


class LlamaServerWrapper:
    def __init__(
        self,
        bin_path: str = "llama.cpp/build/bin/llama-server",
        work_dir: Path = Path.cwd(),
        numactl: Optional[Sequence[str]] = None,
    ):
        if not Path(bin_path).is_file():
            raise FileNotFoundError("找不到 llama-server 可执行文件, 请先进行编译")
        self.bin_path = bin_path
        self.work_dir = work_dir
        self.numactl: Optional[list[str]] = (
            list(numactl) if numactl is not None else None
        )
        self.moe_counter = True if os.getenv("LLAMA_MOE_COUNTER") == "1" else False
        self.process: Optional[subprocess.Popen] = None
        self.output_thread: Optional[threading.Thread] = None
        self._log_file: Optional[object] = None
        self.log_path = work_dir / "llama-server.log"

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
            logger.error(f"读取输出时出错: {e}")
        finally:
            try:
                if self.process and self.process.stdout:
                    self.process.stdout.close()
            except Exception:
                pass

    def start(self, argv: list[str], timeout: int = 60) -> int:
        if self.process and self.process.poll() is None:
            raise RuntimeError("llama-server 已在运行，请先 stop().")

        cmd = [self.bin_path] + argv

        if self.numactl:
            cmd = self.numactl + cmd

        logger.debug(f"启动命令: {' '.join(cmd)}")
        logger.debug(f"日志文件: {self.log_path}")

        # 打开日志文件（保持句柄，用于持续写入）
        self._log_file = open(self.log_path, "w", buffering=1, encoding="utf-8")

        env = {"WORK_DIR": str(self.work_dir)}

        # 检查参数和当前环境变量
        if self.moe_counter:
            env["LLAMA_MOE_COUNTER"] = "1"
        else:
            env["LLAMA_MOE_COUNTER"] = "0"

        popen_kwargs = dict(
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            start_new_session=True,
            env=env,
        )

        try:
            self.process = subprocess.Popen(cmd, **popen_kwargs)
        except FileNotFoundError:
            logger.error(f"找不到可执行文件: {cmd[0]}")
            self._log_file.close()
            self._log_file = None
            return -1
        except Exception as e:
            logger.error(f"启动失败: {e}")
            self._log_file.close()
            self._log_file = None
            return -1

        # 后台线程持续写日志
        self.output_thread = threading.Thread(
            target=self._read_output_continuously,
            daemon=True,
        )
        self.output_thread.start()

        # 等待启动成功标志
        success = False
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                # 检查进程是否提前退出
                if self.process.poll() is not None:
                    logger.error("进程提前退出，请检查日志。")
                    break

                # 检查日志文件是否包含启动成功标志
                with open(self.log_path, "r", encoding="utf-8", errors="ignore") as log:
                    for line in log:
                        if "server is listening on" in line:
                            success = True
                            break
                if success:
                    break
                time.sleep(0.5)
            except Exception as e:
                logger.error(f"检查启动状态时出错: {e}")
                break

        if not success:
            logger.error(f"启动失败: {timeout} 秒内未检测到启动成功标志")
            self.stop()
            return -1

        return self.process.pid

    def stop(self) -> None:
        if not self.process:
            logger.debug("stop: 未发现子进程。")
            return

        already_exited = self.process.poll() is not None

        if already_exited:
            logger.debug("stop: 子进程已退出。")
        else:
            try:
                pgid = os.getpgid(self.process.pid)
                os.killpg(pgid, signal.SIGTERM)
            except ProcessLookupError:
                logger.warning("stop: 进程不存在或已退出。")
            except Exception as e:
                logger.error(f"stop: 发送信号失败：{e}")

            try:
                return_code = self.process.wait(timeout=60)
                if return_code != 0:
                    logger.warning(f"llama-server 退出异常, 子返回码 {return_code}")
            except subprocess.TimeoutExpired:
                logger.error("llama-server 退出超时")

        if self.output_thread and self.output_thread.is_alive():
            self.output_thread.join(timeout=5)
        if self._log_file:
            self._log_file.close()
            self._log_file = None
        self.process = None

    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None
