import threading
import time
from datetime import datetime
import pynvml as nvml
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


class GPURecorder:
    """
    单卡 GPU=0 监控器：
    - start(): 启动后台线程采样；当GPU利用率大于0时开始记录
    - finish(): 停止采样并返回所有记录(list[dict])，同时生成监控图表
    """

    _MB_FACTOR = 1024 * 1024  # 字节转 MB 的因子

    def __init__(self, interval: float = 0.1):
        self.interval = float(interval)  # 采样间隔秒

        self._records = []  # 采样结果 list[dict]
        self._thread = None
        self._stop_event = threading.Event()
        self._started_recording = False  # 是否已开始真正记录

        self._nvml_initialized = False
        self._gpu_handle = None  # 缓存 GPU 句柄

    # ========== 公有 API ==========

    def start(self):
        """在你的训练开始前调用"""
        if self._thread is not None:
            return  # 已启动

        self._init_nvml()

        self._thread = threading.Thread(
            target=self._run_loop, name="GPURecorder", daemon=True
        )
        self._thread.start()

    def finish(self):
        """在你的训练结束后调用，返回采样数组；同时释放 NVML"""
        self._stop_event.set()

        if self._thread is not None:
            self._thread.join()
            self._thread = None

        self._shutdown_nvml()

        # 如果有记录数据，则生成图表
        if self._records:
            self._plot_gpu_metrics()

        return self._records

    def get_average_metrics(self):
        """计算并返回平均GPU利用率和显存占用率"""
        if not self._records:
            return {"avg_gpu_util_percent": 0.0, "avg_memory_percent": 0.0}

        total_gpu_util = sum(record["gpu_util_percent"] for record in self._records)
        record_count = len(self._records)

        return {
            "avg_gpu_util_percent": round(total_gpu_util / record_count, 2),
        }

    # ========== 内部实现 ==========

    def _init_nvml(self):
        """初始化 NVML 并获取 GPU 句柄"""
        if self._nvml_initialized:
            return

        nvml.nvmlInit()
        device_count = nvml.nvmlDeviceGetCount()

        if device_count < 1:
            raise RuntimeError("未检测到任何 GPU（NVML 设备数为 0）")

        self._gpu_handle = nvml.nvmlDeviceGetHandleByIndex(0)  # 缓存句柄
        self._nvml_initialized = True

    def _shutdown_nvml(self):
        """关闭 NVML"""
        if self._nvml_initialized:
            try:
                nvml.nvmlShutdown()
            finally:
                self._nvml_initialized = False
                self._gpu_handle = None

    def _create_sample_record(self):
        """创建一条包含 GPU0 指标的记录"""
        utilization = nvml.nvmlDeviceGetUtilizationRates(self._gpu_handle)

        timestamp = time.time()

        return {
            "ts_iso": datetime.fromtimestamp(timestamp).isoformat(
                timespec="milliseconds"
            ),
            "ts_ms": int(timestamp * 1000),
            "interval_s": self.interval,
            "gpu_index": 0,
            "gpu_util_percent": int(getattr(utilization, "gpu", 0)),
        }

    def _run_loop(self):
        """主采样循环"""
        last_time = time.perf_counter()

        while not self._stop_event.is_set():
            # 控制采样间隔
            current_time = time.perf_counter()
            elapsed = current_time - last_time

            if elapsed < self.interval:
                remaining_time = self.interval - elapsed
                if self._stop_event.wait(remaining_time):
                    break  # 收到停止信号

            last_time = time.perf_counter()

            # 采样并创建记录
            record = self._create_sample_record()

            # 如果尚未开始记录，检查是否达到条件
            if not self._started_recording:
                if record["gpu_util_percent"] > 0:
                    self._started_recording = True
                else:
                    continue

            # 如果已开始记录，则保存数据
            if self._started_recording:
                self._records.append(record)

    def _plot_gpu_metrics(self):
        """绘制 GPU 利用率图表并保存到本地"""
        if not self._records:
            return

        # 提取数据
        timestamps = [
            datetime.fromtimestamp(record["ts_ms"] / 1000) for record in self._records
        ]
        gpu_util_percent = [record["gpu_util_percent"] for record in self._records]

        # 创建图表
        fig, ax = plt.subplots(figsize=(12, 6))

        ax.set_xlabel("Time (H:M:S)")
        ax.set_ylabel("GPU Utilization (%)", color="tab:red")
        ax.plot(
            timestamps,
            gpu_util_percent,
            color="tab:red",
            linewidth=2,
            label="GPU Utilization",
        )
        ax.tick_params(axis="y", labelcolor="tab:red")
        ax.set_ylim(0, 100)

        # 设置时间轴格式
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        ax.xaxis.set_major_locator(
            mdates.SecondLocator(interval=max(1, len(timestamps) // 10))
        )
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        # 图例
        ax.legend(loc="upper left")

        # 标题和布局
        plt.title("GPU Utilization Metrics", fontsize=14, pad=20)
        plt.tight_layout()
        plt.grid(True, alpha=0.3)

        # 生成文件名并保存
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"gpu_utilization_{timestamp_str}.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)  # 关闭图表释放内存

        print(f"GPU 利用率图表已保存至: {filename}")