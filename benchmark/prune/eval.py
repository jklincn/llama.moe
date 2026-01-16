import time
from pathlib import Path

import openai
from evalscope.config import EvalType, TaskConfig
from evalscope.run import run_task

from benchmark.server import LlamaMoeServerHandler


def run_eval(model: str, dataset: list[str], limit: int | None = None):
    handler = LlamaMoeServerHandler(model, args=["--enable-counter"])

    print("=" * 80)
    print(f"模型: {model}")
    print(f"数据集: {dataset}")
    print(f"limit: {limit if limit is not None else '不限'}")

    try:
        print("启动 llama-moe server...")
        handler.start_server()
    except Exception as e:
        print(f"Failed to start server: {e}")
        return None

    # Initialize OpenAI Client
    base_url = "http://127.0.0.1:8080/v1"
    api_key = "sk-1234"
    client = openai.OpenAI(base_url=base_url, api_key=api_key)

    try:
        print("检查 OpenAI-compatible 接口连通性...")
        models_list = client.models.list()
        if not models_list.data:
            raise ValueError("No models found on the server.")
        target_model_id = models_list.data[0].id
        print(f"服务端模型 ID: {target_model_id}")

    except Exception as e:
        print(f"Connection error: {e}")
        handler.stop_server()
        return None

    output_dir = (Path("outputs") / "_".join(dataset)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    task_config = TaskConfig(
        model=target_model_id,
        datasets=dataset,
        eval_type=EvalType.SERVICE,
        eval_batch_size=1,
        api_url="http://127.0.0.1:8080/v1/chat/completions",
        api_key="sk-1234",
        generation_config={
            "max_tokens": 4096,
            "temperature": 0.0,
            "stream": True,
        },
        limit=limit,
        use_cache=str(output_dir),
    )

    print(f"输出目录(use_cache): {output_dir}")
    print("开始运行评估...")

    start_t = time.time()
    try:
        result = run_task(task_cfg=task_config)
        elapsed = time.time() - start_t
        print(f"评估完成，用时: {elapsed:.2f}s")
        return result

    except KeyboardInterrupt:
        print("\nBenchmark interrupted.")
    finally:
        handler.stop_server()


# python -m benchmark.accuracy.eval
if __name__ == "__main__":
    models = [
        "Qwen3-Next-80B-A3B-Instruct",
        # "GLM-4.5-Air",
        # "Qwen3-235B-A22B",
    ]

    dataset = [
        # Math
        "gsm8k",
        "math_500",
        # Code
        # "humaneval",
        # "mbpp",
    ]

    limit = None

    for model in models:
        run_eval(model=model, dataset=dataset, limit=limit)
