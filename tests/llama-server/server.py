import os
import signal
import subprocess
import sys
import time

import requests

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
)


class LlamaServer:
    def __init__(
        self,
        model_path="/mnt/data/gguf/Qwen3-30B-A3B-Q8_0.gguf",
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
        self.process = subprocess.Popen(
            cmd, stdout=logfile, stderr=logfile, preexec_fn=os.setsid
        )

        logger.info(f"正在启动, PID: {self.process.pid}")

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
            logger.info("terminated.")
        else:
            logger.info("not running.")
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
        logger.info(f"\n收到信号 {signum}，正在关闭服务器...")
        server.stop()
        sys.exit(0)

    def check_health():
        try:
            response = requests.get(f"http://localhost:{server.port}/health", timeout=5)
            if response.status_code == 200:
                return True
        except requests.exceptions.RequestException as e:
            return False

    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # 终止信号

    try:
        # 启动服务器
        logger.info(f"正在启动服务器，端口: {server.port}")
        logger.info(f"模型路径: {server.model_path}")
        logger.info("日志文件: llama-server.log")
        logger.info("按 Ctrl+C 停止服务器")

        server.start()

        while True:
            if check_health():
                logger.info("服务器启动完成")
                break
            else:
                time.sleep(2)

        while True:
            if check_health():
                logger.info("服务器健康状态正常")
            else:
                logger.info("服务器健康状态异常")
            time.sleep(60)
    except Exception as e:
        logger.info(f"发生错误: {e}")
        server.stop()
        sys.exit(1)
