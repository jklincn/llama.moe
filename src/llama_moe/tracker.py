import time
import requests
import threading
import logging

logger = logging.getLogger("tracker")

class MetricsTracker(threading.Thread):
    def __init__(
        self, 
        host: str = "127.0.0.1", 
        port: int = 8080, 
        api_key: str = "sk-1234", 
        threshold: int = 10000,
        interval: float = 5.0,
        on_threshold_reached: callable = None
    ):
        super().__init__(daemon=True)
        self.host = host
        self.port = port
        self.api_key = api_key
        self.threshold = threshold
        self.interval = interval
        self.on_threshold_reached = on_threshold_reached
        self.running = True
        self.triggered = False

    def stop(self):
        self.running = False

    def run(self):
        metrics_url = f"http://{self.host}:{self.port}/metrics"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        
        logger.info(f"Tracker started. Monitoring for threshold: {self.threshold}")

        while self.running and not self.triggered:
            try:
                resp = requests.get(metrics_url, headers=headers, timeout=0.5)
                if resp.status_code == 200:
                    lines = resp.text.split('\n')
                    total_tokens = 0
                    
                    for line in lines:
                        if line.startswith("llamacpp:tokens_predicted_total"):
                            total_tokens = int(float(line.split()[-1]))
                            break
                    
                    if total_tokens >= self.threshold:
                        logger.info(f"Threshold reached: {total_tokens} >= {self.threshold}")
                        self.triggered = True
                        if self.on_threshold_reached:
                            self.on_threshold_reached(total_tokens)
                        break

            except Exception:
                # 忽略连接错误（服务可能正在重启）
                pass

            time.sleep(self.interval)