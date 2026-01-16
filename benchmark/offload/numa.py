import time

import openai

from benchmark.dataset.deepseek import get_prompts
from benchmark.server import LlamaMoeServerHandler


def run_benchmark(
    server_name: str, model_name: str, count=100, server_args=None, server_label=None
):
    server_args = server_args or []
    framework_label = server_label or server_name

    # 1. Prepare Dataset
    print(f"Preparing {count} prompts from dataset...")
    try:
        prompts = get_prompts(count=count)
    except Exception as e:
        print(f"Failed to get prompts: {e}")
        return None

    if not prompts:
        print("No prompts loaded.")
        return None

    print(f"Loaded {len(prompts)} prompts for benchmarking.")

    # 2. Initialize Handler
    print(f"Initializing handler for {framework_label} with model {model_name}...")

    handler = LlamaMoeServerHandler(model_name, log_dir="./logs", args=server_args)

    # 3. Start Server
    try:
        handler.start_server()
    except Exception as e:
        print(f"Failed to start server: {e}")
        return None

    # 4. Initialize OpenAI Client
    base_url = "http://127.0.0.1:8080/v1"
    api_key = "sk-1234"
    client = openai.OpenAI(base_url=base_url, api_key=api_key)

    try:
        models_list = client.models.list()
        if not models_list.data:
            raise ValueError("No models found on the server.")
        target_model_id = models_list.data[0].id

    except Exception as e:
        print(f"Connection error: {e}")
        handler.stop_server()
        return None

    temperature = 0.0
    top_p = 1.0
    top_k = 1
    max_tokens = 128

    print("Running benchmark...")

    failed_requests = 0

    try:
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]

            try:
                req_start = time.time()

                if server_name == "fastllm":
                    completion = client.chat.completions.create(
                        model=target_model_id,
                        messages=messages,
                        temperature=temperature,
                        top_p=top_p,
                        extra_body={"top_k": top_k},
                        stream=True,
                        stream_options={"include_usage": True},
                        max_tokens=max_tokens,
                    )

                    first_token_time = None
                    last_token_time = None
                    completion_tokens = None

                    for chunk in completion:
                        # 1) 计时：只要有内容就更新 first/last
                        if chunk.choices:
                            delta = chunk.choices[0].delta
                            if getattr(delta, "content", None):
                                now = time.time()
                                if first_token_time is None:
                                    first_token_time = now
                                last_token_time = now

                        # 2) 取 token：最后一个 chunk 会带 usage
                        u = getattr(chunk, "usage", None)
                        if u is not None:
                            ct = getattr(u, "completion_tokens", None)
                            if ct is not None:
                                completion_tokens = int(ct)

                    # 3) 汇总
                    if (
                        completion_tokens
                        and first_token_time
                        and last_token_time
                        and last_token_time > first_token_time
                    ):
                        decode_duration = last_token_time - first_token_time
                        data = {
                            "usage": {"completion_tokens": max(0, completion_tokens)}
                        }
                        handler.handle_result(data, decode_duration)

                else:
                    # For others (llama-cpp or llama-moe), use server-supplied timings
                    completion = client.chat.completions.create(
                        model=target_model_id,
                        messages=messages,
                        temperature=temperature,
                        top_p=top_p,
                        extra_body={"top_k": top_k},
                        stream=False,
                        max_tokens=max_tokens,
                    )

                    req_duration = time.time() - req_start
                    data = completion.model_dump()

                    # Delegate to handler for stats accumulation
                    handler.handle_result(data, req_duration)

            except Exception as e:
                print(f"Request failed: {e}")
                failed_requests += 1

    except KeyboardInterrupt:
        print("\nBenchmark interrupted.")
    finally:
        tps = handler.get_result()

        result = {
            "framework": framework_label,
            "model": model_name,
            "throughput_tps": tps,
        }

        handler.stop_server()

        return result


def print_results_table(results):
    """
    打印结果汇总表格 - 以模型为分组，显示不同框架的 TPS

    Args:
        results: 结果列表，每个元素是一个包含测试结果的字典
    """
    if not results:
        print("No results to display.")
        return

    print("\n" + "=" * 60)
    print("THROUGHPUT BENCHMARK RESULTS (tokens/s)")
    print("=" * 60)

    # 获取所有模型和框架
    models = []
    frameworks = []
    for r in results:
        if r["model"] not in models:
            models.append(r["model"])
        if r["framework"] not in frameworks:
            frameworks.append(r["framework"])

    # 按模型分组显示
    for model in models:
        print(f"\n{model}:")
        print("-" * 60)

        model_results = [r for r in results if r["model"] == model]

        for result in model_results:
            framework = result["framework"]
            throughput = result["throughput_tps"]
            print(f"  {framework:<20} {throughput:>10.2f} tokens/s")

    print("\n" + "=" * 60)


# python -m benchmark.prune.overhead
if __name__ == "__main__":
    all_results = []

    models = [
        "Qwen3-Next-80B-A3B-Instruct",
        "GLM-4.5-Air",
        "Qwen3-235B-A22B",
    ]

    # fmt: off
    servers = [
        {"name": "llama-moe", "label": "llama-moe", "args": []},
        {"name": "llama-moe", "label": "llama-moe+counter", "args": ["--enable-counter"]},
    ]
    # fmt: on

    count = 20

    total_tests = len(models) * len(servers)
    current_test = 0

    for model in models:
        for s in servers:
            current_test += 1
            print("\n" + "=" * 60)
            print(f"TEST {current_test}/{total_tests}: {s['label']} - {model}")
            print("=" * 60 + "\n")

            result = run_benchmark(
                s["name"],
                model,
                count,
                server_args=s["args"],
                server_label=s["label"],
            )

            if result is not None:
                all_results.append(result)
                print("Test completed successfully")
            else:
                print("Test failed")

            time.sleep(1)

    print_results_table(all_results)
