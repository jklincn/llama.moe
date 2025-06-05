import requests


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
        self.get("/health")

    def completions(
        self,
        prompt: str,
        n_predict: int = -1,
        temperature: float = 0.8,
        stop: list = None,
        stream: bool = False,
        **kwargs,
    ) -> dict:
        """
        Perform a single-turn completion.

        Args:
            prompt: The prompt to send to the model.
            n_predict: Number of tokens to predict. -1 for model default.
            temperature: Sampling temperature.
            stop: A list of strings to stop generation at.
            stream: Whether to stream the response. Defaults to False for single-round.
            **kwargs: Additional parameters to pass to the server,
                      e.g., top_k, top_p, repeat_penalty, grammar.

        Returns:
            A dictionary containing the server's JSON response.
        """
        data = {
            "prompt": prompt,
            "n_predict": n_predict,
            "temperature": temperature,
            "stream": stream,
            **kwargs,
        }
        if stop is not None:
            data["stop"] = stop

        response = self.post("/completions", data=data)
        return response.json()

    def get_metrics(self) -> str:
        """
        Retrieves metrics from the server.
        Requires the server to be started with the --metrics flag.

        Returns:
            A string containing the metrics in Prometheus format.
        """
        response = self.get("/metrics")
        return response.text

    def get_slots(self) -> list:
        """
        Retrieves the state of all processing slots from the server.
        Requires the server to be started with the --slots flag.

        Returns:
            A list of dictionaries, where each dictionary represents a slot's state.
        """
        response = self.get("/slots")
        return response.json()


def main():
    api = LlamaServerAPI("http://127.0.0.1:8088")
    completion_data = {
        "prompt": "你好，世界！请给我写一个关于猫的小故事。",
        "n_predict": 200,
    }
    response_json = api.completions(**completion_data)
    print("\nCompletion Response:")
    print(response_json)
    if "content" in response_json:
        print(f"\nGenerated content: {response_json['content']}")


if __name__ == "__main__":
    main()
