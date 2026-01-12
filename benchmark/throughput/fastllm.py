import os
import signal
import time
from datetime import datetime
from subprocess import Popen

import requests
from modelscope.hub.snapshot_download import snapshot_download
from .server_handler import ServerHandler

model_list = {
    # "Qwen3-Next-80B-A3B-Instruct": {
    #     "path": "/mnt/data/gguf/Qwen3-Next-80B-A3B-Instruct-Q4_K_M.gguf",
    #     "hf_path": "Qwen/Qwen3-Next-80B-A3B-Instruct",
    #     "moe_device": "{'cuda':13,'cpu':35}",
    # },
    "GLM-4.5-Air": {
        "path": "/mnt/data/gguf/GLM-4.5-Air-Q4_K_M.gguf",
        "hf_path": "zai-org/GLM-4.5-Air",
        "moe_device": "{'cuda':10,'cpu':34}",
    },
    # "MiniMax-M2": {
    #     "path": "/mnt/data/gguf/MiniMax-M2-Q4_K_M.gguf",
    #     "hf_path": "MiniMax/MiniMax-M2",
    #     "moe_device": "{'cuda':7,'cpu':39}",
    # },
    "Qwen3-235B-A22B": {
        "path": "/mnt/data/gguf/Qwen3-235B-A22B-Q4_K_M.gguf",
        "hf_path": "Qwen/Qwen3-235B-A22B",
        "moe_device": "{'cuda':10,'cpu':84}",
    },
}


class FastLLMServerHandler(ServerHandler):
    def __init__(self, model_name: str, log_dir: str = "./logs"):
        if model_name not in model_list:
            raise ValueError(f"Model {model_name} not found in model_list")
        self.model_name = model_name
        self.model_info = model_list[model_name]
        self.log_dir = log_dir

        self.process = None
        self.log_f = None

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
        model_path = self.model_info["path"]
        hf_path = self.model_info["hf_path"]
        moe_device = self.model_info["moe_device"]

        ori_path = self._get_ori(hf_path)

        cmd = [
            "ftllm",
            "server",
            model_path,
            "--ori",
            ori_path,
            "--device",
            "cuda",
            "--moe_device",
            moe_device,
        ]

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = self.model_name.replace("/", "_")
        log_path = os.path.join(self.log_dir, f"ftllm_{safe_name}_{ts}.log")

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

    def stop_server(self):
        if self.process and self.process.poll() is None:
            print("Stopping FastLLM server...")
            try:
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                self.process.wait(timeout=10)
            except Exception:
                print("Force killing server process group...")
                try:
                    os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                except Exception:
                    pass
        self.process = None

        if self.log_f:
            self.log_f.close()
            self.log_f = None

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
