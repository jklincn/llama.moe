import os
import signal
import time
from datetime import datetime
from pathlib import Path
from subprocess import Popen

import psutil
import requests
from .server_handler import ServerHandler

model_list = {
    "Qwen3-Next-80B-A3B-Instruct": {
        "path": "/mnt/data/gguf/Qwen3-Next-80B-A3B-Instruct-Q8_0.gguf",
        "n-gpu-layers": 14,
    },
    "GLM-4.5-Air": {
        "path": "/mnt/data/gguf/GLM-4.5-Air-Q4_K_M.gguf",
        "n-gpu-layers": 16,
    },
    "Qwen3-235B-A22B": {
        "path": "/mnt/data/gguf/Qwen3-235B-A22B-Q4_K_M.gguf",
        "n-gpu-layers": 15,
    },
}


class LlamaCppServerHandler(ServerHandler):
    def __init__(self, model_name: str, log_dir: str = "./logs"):
        if model_name not in model_list:
            raise ValueError(f"Model {model_name} not found in model_list")
        self.model_name = model_name
        self.model_info = model_list[model_name]
        self.log_dir = log_dir
        self.process = None
        self.log_f = None
        self.port = 8080

        # Statistics
        self.success_count = 0
        self.sum_predicted_n = 0
        self.sum_predicted_ms = 0

    def get_server_name(self) -> str:
        return "llama.cpp"

    def _kill_process_on_port(self, port: int):
        """杀死占用指定端口的进程"""
        try:
            for conn in psutil.net_connections():
                if conn.laddr.port == port and conn.status == "LISTEN":
                    try:
                        process = psutil.Process(conn.pid)
                        print(
                            f"Found process occupying port {port}: PID={conn.pid}, name={process.name()}"
                        )
                        process.terminate()
                        try:
                            process.wait(timeout=20)
                            print(
                                f"Successfully terminated process {conn.pid} occupying port {port}"
                            )
                        except psutil.TimeoutExpired:
                            print(
                                f"Process {conn.pid} did not exit within 20 seconds, using SIGKILL"
                            )
                            process.kill()
                            process.wait(timeout=5)
                            print(f"Process {conn.pid} was forcibly terminated")
                    except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                        print(f"Error terminating process {conn.pid}: {e}")
        except Exception as e:
            print(f"Error cleaning up port {port}: {e}")

    def _wait_for_ready(
        self,
        url: str = "http://127.0.0.1:8080/health",
        timeout_sec: int = 600,
        interval_sec: float = 2.0,
    ) -> bool:
        # llama-server has /health endpoint usually, or check /v1/models
        # Trying /v1/models is safer for openai compatibility
        url = "http://127.0.0.1:8080/v1/models"

        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            if self.process is not None and self.process.poll() is not None:
                print(f"Server process died with code {self.process.returncode}")
                return False

            try:
                resp = requests.get(url, timeout=5)
                if resp.status_code == 200:
                    return True
            except requests.RequestException:
                pass

            time.sleep(interval_sec)
        return False

    def start_server(self):
        self._kill_process_on_port(self.port)
        time.sleep(1)

        # Locate llama-server binary
        current_dir = Path(__file__).parent.resolve()
        project_root = current_dir.parent.parent
        server_bin = project_root / "llama.cpp" / "build" / "bin" / "llama-server"

        if not server_bin.exists():
            raise FileNotFoundError(f"llama-server binary not found at {server_bin}")

        model_path = self.model_info["path"]
        n_gpu_layers = self.model_info["n-gpu-layers"]
        # fmt: off
        cmd = [
            str(server_bin),
            "--model", model_path,
            "--api-key", "sk-1234",
            "--ctx-size", "4096",
            "--n-gpu-layers", str(n_gpu_layers),
            "--seed", "0",
        ]
        # fmt: on

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = self.model_name.replace("/", "_")
        log_path = os.path.join(self.log_dir, f"llama_cpp_{safe_name}_{ts}.log")

        print("Starting llama-server with command:")
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
            print("llama-server is up!")
        else:
            print("llama-server failed to become healthy.")
            self.stop_server()
            raise RuntimeError(f"llama-server failed to start for {self.model_name}")

    def stop_server(self):
        """停止服务器进程（Linux：仅发送 Ctrl+C / SIGINT）"""
        if self.process and self.process.poll() is None:
            pid = self.process.pid
            print(f"Stopping llama-server (PID: {pid})...")

            try:
                pgid = os.getpgid(pid)

                print(f"Sending SIGINT (Ctrl+C) to process group {pgid}...")
                try:
                    os.killpg(pgid, signal.SIGINT)
                except ProcessLookupError:
                    print("Process group not found (already exited).")
                except Exception as e:
                    print(f"Failed to send SIGINT: {e}")

            except Exception as e:
                print(f"Error stopping server: {e}")

        self.process = None

        if self.log_f:
            try:
                self.log_f.close()
            except Exception:
                pass
            self.log_f = None

    def handle_result(self, data, duration):
        """
        Extract 'timings' from the response.
        """
        timings = data.get("timings")
        if timings:
            pred_n = int(timings.get("predicted_n", 0) or 0)
            pred_ms = float(timings.get("predicted_ms", 0) or 0)

            if pred_n > 0 and pred_ms > 0:
                self.sum_predicted_n += pred_n
                self.sum_predicted_ms += pred_ms
                self.success_count += 1

    def get_result(self):
        """
        Return aggregated tokens/sec based on server timings.
        """
        if self.sum_predicted_ms > 0:
            return (self.sum_predicted_n / self.sum_predicted_ms) * 1000.0
        return 0.0


if __name__ == "__main__":
    handler = LlamaCppServerHandler("GLM-4.5-Air")
    try:
        handler.start_server()
        time.sleep(5)
    finally:
        handler.stop_server()
