import logging
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Optional, Sequence
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

logger = logging.getLogger("wrapper")


class LlamaServerWrapper:
    def __init__(
        self,
        bin_path: str,
        work_dir: Path,
        numactl: Optional[Sequence[str]],
        host: str,
        port: int,
        log_to_file: bool,
    ):
        if not Path(bin_path).is_file():
            raise FileNotFoundError("找不到 llama-server 可执行文件, 请先进行编译")

        self.bin_path = str(bin_path)
        self.work_dir = work_dir
        self.numactl: Optional[list[str]] = (
            list(numactl) if numactl is not None else None
        )

        self.host = host
        self.port = port
        self.log_to_file = log_to_file

        self.moe_counter = os.getenv("LLAMA_MOE_COUNTER") == "1"
        self.process: Optional[subprocess.Popen] = None
        self._log_file = None
        self.log_path = self.work_dir / "llama-server.log"

    def start(self, argv: list[str], timeout: int = 60) -> int:
        if self.process and self.process.poll() is None:
            raise RuntimeError("llama-server 已在运行，请先 stop().")

        cmd = [self.bin_path] + argv
        if self.numactl:
            cmd = self.numactl + cmd

        self.work_dir.mkdir(parents=True, exist_ok=True)

        env = os.environ.copy()
        env["WORK_DIR"] = str(self.work_dir)
        env["LLAMA_MOE_COUNTER"] = "1" if self.moe_counter else "0"

        popen_kwargs = dict(
            text=True,
            bufsize=1,
            start_new_session=True,
            env=env,
            cwd=str(self.work_dir),
            close_fds=True,
        )

        # 是否重定向日志
        if self.log_to_file:
            self._log_file = open(
                self.log_path, "w", buffering=1, encoding="utf-8", errors="replace"
            )
            popen_kwargs["stdout"] = self._log_file
            popen_kwargs["stderr"] = self._log_file
        else:
            self._log_file = None

        try:
            self.process = subprocess.Popen(cmd, **popen_kwargs)
        except FileNotFoundError:
            logger.error(f"找不到可执行文件: {cmd[0]}")
            self._cleanup_after_start_failure()
            return -1
        except Exception as e:
            logger.error(f"启动失败: {e}")
            self._cleanup_after_start_failure()
            return -1

        if not self._wait_for_ready_http(timeout=timeout):
            logger.error(f"启动失败: {timeout} 秒内未检测到服务 ready")
            self.stop()
            return -1

        return self.process.pid

    def _wait_for_ready_http(
        self,
        timeout: int,
        interval_sec: float = 2,
        per_request_timeout: float = 2.0,
    ) -> bool:
        assert self.process is not None

        url = f"http://{self.host}:{self.port}/v1/models"
        deadline = time.time() + timeout

        while time.time() < deadline:
            if self.process.poll() is not None:
                logger.error(f"进程提前退出，returncode={self.process.returncode}")
                return False

            try:
                req = Request(url, method="GET")
                with urlopen(req, timeout=per_request_timeout) as resp:
                    if resp.status == 200:
                        return True
            except HTTPError:
                pass
            except URLError:
                pass
            except Exception:
                pass

            time.sleep(interval_sec)

        return False

    def _cleanup_after_start_failure(self) -> None:
        try:
            if self._log_file:
                self._log_file.close()
        except Exception:
            pass
        self._log_file = None
        self.process = None

    def stop(self) -> None:
        if not self.process:
            logger.debug("stop: 未发现子进程。")
            return

        if self.process.poll() is None:
            try:
                pgid = os.getpgid(self.process.pid)
                os.killpg(pgid, signal.SIGTERM)
            except ProcessLookupError:
                logger.warning("stop: 进程不存在或已退出。")
            except Exception as e:
                logger.error(f"stop: 发送信号失败：{e}")

            try:
                return_code = self.process.wait(timeout=30)
                if return_code != 0:
                    logger.warning(f"llama-server 退出异常, 子返回码 {return_code}")
            except subprocess.TimeoutExpired:
                logger.error("llama-server 退出超时，尝试 SIGKILL")
                try:
                    pgid = os.getpgid(self.process.pid)
                    os.killpg(pgid, signal.SIGKILL)
                except Exception:
                    pass
                try:
                    self.process.wait(timeout=5)
                except Exception:
                    pass
        else:
            logger.debug("stop: 子进程已退出。")

        if self._log_file:
            try:
                self._log_file.close()
            except Exception:
                pass
            self._log_file = None

        self.process = None

    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None
