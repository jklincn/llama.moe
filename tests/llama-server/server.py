import os
import signal
import subprocess
import sys
import time

import requests


class LlamaServer:
    def __init__(
        self,
        model_path="/mnt/data/gguf/DeepSeek-R1-Q4_K_M.gguf",
        server_path=None,
        port=8088,
        seed=0,
        ctx_size=1024,
        override_tensor="exps=CPU",
        n_gpu_layers=999,
        enable_metrics=True,
        enable_slots=True,
    ):
        """
        初始化 LlamaServer 实例

        :param model_path: 模型文件路径
        :param server_path: llama-server 可执行文件路径，默认使用相对路径推导
        :param port: 启动端口
        :param seed: 随机种子
        :param ctx_size: 上下文大小
        :param override_tensor: tensor 配置
        :param n_gpu_layers: 使用 GPU 的层数
        :param enable_metrics: 是否启用 metrics
        :param enable_slots: 是否启用 slots
        """
        self.model_path = model_path
        self.port = str(port)
        self.seed = str(seed)
        self.ctx_size = str(ctx_size)
        self.override_tensor = override_tensor
        self.n_gpu_layers = str(n_gpu_layers)
        self.enable_metrics = enable_metrics
        self.enable_slots = enable_slots

        if server_path is None:
            dir_path = os.path.dirname(os.path.realpath(__file__))
            self.server_path = os.path.join(
                dir_path, "../../llama.cpp/build/bin/llama-server"
            )
        else:
            self.server_path = server_path

        self.process = None

    def build_command(self):
        # fmt: off
        cmd = [
            self.server_path,
            "--model", self.model_path,
            "--port", self.port,
            "--seed", self.seed,
            "--override-tensor", self.override_tensor,
            "--n-gpu-layers", self.n_gpu_layers,
            "--ctx-size", self.ctx_size,
        ]
        # fmt: on
        if self.enable_metrics:
            cmd.append("--metrics")
        if self.enable_slots:
            cmd.append("--slots")
        return cmd

    def start(self, log_file="llama-server.log"):
        """
        启动 llama-server 进程
        :param log_file: 日志文件路径，可选
        """
        if self.process:
            raise RuntimeError("LlamaServer 已在运行。")

        cmd = self.build_command()

        logfile = open(log_file, "w")
        self.process = subprocess.Popen(cmd, stdout=logfile, stderr=logfile)

        print(f"[LlamaServer] Starting, PID: {self.process.pid}")

    def is_running(self):
        """检查 llama-server 是否仍在运行"""
        return self.process is not None and self.process.poll() is None

    def stop(self):
        """终止 llama-server"""
        if self.process and self.is_running():
            self.process.terminate()
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.kill()
            print("[LlamaServer] terminated.")
        else:
            print("[LlamaServer] not running.")
        self.process = None

    def __enter__(self):
        self.start(log_file="llama-server.log")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


if __name__ == "__main__":
    # 创建服务器实例
    server = LlamaServer()

    # 设置信号处理函数，用于优雅关闭
    def signal_handler(signum, frame):
        print(f"\n[LlamaServer] 收到信号 {signum}，正在关闭服务器...")
        server.stop()
        sys.exit(0)

    def check_health():
        try:
            response = requests.get(f"http://localhost:{server.port}/health", timeout=5)
            if response.status_code == 200:
                print("[LlamaServer] 服务器健康状态正常")
                return True
        except requests.exceptions.RequestException as e:
            print("[LlamaServer] 健康检查失败")
            return False

    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # 终止信号

    try:
        # 启动服务器
        print(f"[LlamaServer] 正在启动服务器，端口: {server.port}")
        print(f"[LlamaServer] 模型路径: {server.model_path}")
        print("[LlamaServer] 日志文件: llama-server.log")
        print("[LlamaServer] 按 Ctrl+C 停止服务器")

        server.start()

        while True:
            health = check_health()
            if health:
                time.sleep(60)
            else:
                time.sleep(5)

        print("[LlamaServer] 服务器进程已退出")

    except KeyboardInterrupt:
        print("\n[LlamaServer] 用户中断，正在关闭服务器...")
        server.stop()
    except Exception as e:
        print(f"[LlamaServer] 发生错误: {e}")
        server.stop()
        sys.exit(1)
