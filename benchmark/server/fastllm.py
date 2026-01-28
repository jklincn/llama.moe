import os
import re
import time
import psutil
from datetime import datetime
from subprocess import Popen

import requests
from modelscope.hub.snapshot_download import snapshot_download
from .server_handler import ServerHandler
from ..utils.port import kill_process_on_port

model_list = {
    "Qwen3-Next-80B-A3B-Instruct": {
        "path": "/mnt/data/safetensors/Qwen3-Next-80B-A3B-Instruct-FP8",
        "hf_path": None,
        "moe_device": "{'cuda':13,'cpu':35}",
    },
    "GLM-4.5-Air": {
        "path": "/mnt/data/gguf/GLM-4.5-Air-Q4_K_M.gguf",
        "hf_path": "zai-org/GLM-4.5-Air",
        "moe_device": "{'cuda':11,'cpu':35}",
    },
    "Qwen3-235B-A22B": {
        "path": "/mnt/data/gguf/Qwen3-235B-A22B-Q4_K_M.gguf",
        "hf_path": "Qwen/Qwen3-235B-A22B",
        "moe_device": "{'cuda':10,'cpu':84}",
    },
}


class FastLLMServerHandler(ServerHandler):
    def __init__(
        self, model_name: str, log_dir: str = "./logs", args: list[str] = None
    ):
        if model_name not in model_list:
            raise ValueError(f"Model {model_name} not found in model_list")
        self.model_name = model_name
        self.model_info = model_list[model_name]
        self.log_dir = log_dir
        self.port = 8080
        self.args = args if args is not None else []
        self.process = None
        self.log_f = None
        self.log_path = None

        # Statistics
        self.success_count = 0
        self.total_completion_tokens = 0
        self.total_duration = 0.0

    def get_server_name(self) -> str:
        return "FastLLM"

    def _get_ori(self, hf_path: str) -> str:
        model_dir = snapshot_download(
            hf_path,
            ignore_patterns="*safetensors*",
        )
        return model_dir

    def _wait_for_ready(
        self,
        url: str = "http://127.0.0.1:8080/v1/models",
        timeout_sec: int = 600,
        interval_sec: float = 2.0,
    ) -> bool:
        deadline = time.time() + timeout_sec

        while time.time() < deadline:
            if self.process is not None and self.process.poll() is not None:
                return False

            try:
                resp = requests.get(url, timeout=5)
                if resp.status_code == 200:
                    data = resp.json()
                    if "data" in data:
                        return True
            except requests.RequestException:
                pass

            time.sleep(interval_sec)

        return False

    def start_server(self):
        kill_process_on_port(self.port)
        time.sleep(1)  # 等待端口释放

        cmd = ["ftllm", "server", self.model_info["path"]]

        hf_path = self.model_info["hf_path"]
        cmd += ["--ori", self._get_ori(hf_path)] if hf_path is not None else []

        cmd += ["--device", "cuda", "--moe_device", self.model_info["moe_device"]]

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = self.model_name.replace("/", "_")
        log_path = os.path.join(self.log_dir, f"ftllm_{safe_name}_{ts}.log")
        self.log_path = log_path

        cmd += self.args

        print("Starting FastLLM server with command:")
        print(" ".join(cmd))
        print(f"Redirecting logs to: {log_path}")

        os.makedirs(self.log_dir, exist_ok=True)
        self.log_f = open(log_path, "a", buffering=1, encoding="utf-8")

        self.process = Popen(
            cmd,
            stdout=self.log_f,
            stderr=self.log_f,
            text=True,
            bufsize=1,
            close_fds=True,
            start_new_session=True,
        )

        ok = self._wait_for_ready(timeout_sec=600)

        if ok:
            print("FastLLM server is up!")
            print(f"Logs: {log_path}")
        else:
            print("FastLLM server failed to become healthy within 600s.")
            print(f"Logs: {log_path}")
            self.stop_server()
            raise RuntimeError(f"FastLLM server failed to start for {self.model_name}")

    def _cleanup_log(self):
        if not self.log_path or not os.path.exists(self.log_path):
            return

        tmp_path = f"{self.log_path}.tmp"
        try:
            with open(self.log_path, "r", encoding="utf-8", errors="ignore") as src, open(
                tmp_path, "w", encoding="utf-8"
            ) as dst:
                for line in src:
                    if not line.startswith("Loading"):
                        dst.write(line)
            os.replace(tmp_path, self.log_path)
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    def stop_server(self):
        """停止服务器进程"""
        if self.process and self.process.poll() is None:
            print(f"Stoping FastLLM server (PID: {self.process.pid})...")
            try:
                process = psutil.Process(self.process.pid)
                process.terminate()
                try:
                    process.wait(timeout=20)
                    print("Server exited normally")
                except psutil.TimeoutExpired:
                    print(
                        f"Process {self.process.pid} did not exit within 20 seconds, using SIGKILL"
                    )
                    process.kill()
                    process.wait(timeout=5)
                    print("Process was forcibly terminated")
            except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                print(f"Error terminating process: {e}")
            except Exception as e:
                print(f"Error stopping server: {e}")

        self.process = None

        if self.log_f:
            try:
                self.log_f.close()
            except Exception:
                pass
            self.log_f = None

        self._cleanup_log()

    def handle_result(self, data, duration):
        """
        Accumulates statistics from a completion result.

        Args:
            data: The dictionary returned by completion.model_dump()
            duration: The time taken for the request in seconds
        """

        try:
            usage = data.get("usage", {})
            completion_tokens = usage.get("completion_tokens")
            self.total_completion_tokens += completion_tokens
            self.total_duration += duration
            self.success_count += 1
        except Exception as e:
            print(f"Error handling result data: {e}")

    def get_result(self):
        """
        Calculates the decoder throughput based on accumulated stats.

        Returns:
            float: Tokens per second (Total Tokens / Total Duration)
        """
        speeds = []
        if self.log_path and os.path.exists(self.log_path):
            pattern = re.compile(r"Speed:\s*([0-9]*\.?[0-9]+)\s*tokens\s*/\s*s\.")
            try:
                with open(self.log_path, "r", encoding="utf-8", errors="ignore") as log_f:
                    for line in log_f:
                        match = pattern.search(line)
                        if match:
                            speeds.append(float(match.group(1)))
            except Exception:
                speeds = []

        if speeds:
            return sum(speeds) / len(speeds)

        if self.total_duration > 0:
            return self.total_completion_tokens / self.total_duration
        return 0.0


if __name__ == "__main__":
    for model in model_list.keys():
        print(f"=== Testing ServerHandler for model: {model} ===")
        handler = FastLLMServerHandler(model, log_dir="./logs")
        try:
            handler.start_server()
            print("Server running, waiting 5 seconds...")
            time.sleep(5)
        except Exception as e:
            print(f"Server start failed: {e}")
        finally:
            handler.stop_server()
