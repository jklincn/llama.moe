import requests
from server import LlamaServer
import time


class LlamaServerAPIBase:
    def __init__(self, url):
        self.url = url

    def get(self, endpoint) -> requests.Response:
        response = requests.get(f"{self.url}{endpoint}")
        if response.status_code != 200:
            raise Exception(f"Error: {response.status_code} - {response.text}")
        return response

    def post(self, endpoint, data=None) -> requests.Response:
        response = requests.post(f"{self.url}{endpoint}", json=data)
        if response.status_code != 200:
            raise Exception(f"Error: {response.status_code} - {response.text}")
        return response


class LlamaServerAPI(LlamaServerAPIBase):
    def __init__(self, url):
        super().__init__(url)
        self.wait_until_healthy(timeout=60)

    def health(self) -> bool:
        """Check the health of the Llama server."""
        try:
            requests.get(f"{self.url}/health", timeout=5)
            return True
        except (
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            requests.exceptions.RequestException,
        ):
            return False

    def wait_until_healthy(self, timeout=60, interval=3):
        """Wait until the server becomes healthy or timeout occurs."""
        start = time.time()
        while time.time() - start < timeout:
            if self.health():
                print("[LlamaServerAPI] Server is healthy.")
                return True
            print("[LlamaServerAPI] Waiting for server health...")
            time.sleep(interval)
        raise TimeoutError(
            f"[LlamaServerAPI] Server not healthy after {timeout} seconds."
        )


def main():
    api = LlamaServerAPI("http://127.0.0.1:8088")
    print(api.health())


if __name__ == "__main__":
    model = "/mnt/data/gguf/DeepSeek-R1-Q4_K_M.gguf"
    with LlamaServer(model_path=model) as server:
        main()
