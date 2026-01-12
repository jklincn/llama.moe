import time
import subprocess
import openai

from .dataset import get_prompts
from .fastllm import FastLLMServerHandler
from .llama_cpp import LlamaCppServerHandler

def wait_port_free(port: int, timeout: int = 30) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        out = subprocess.run(
            ["bash", "-lc", f"ss -lntp | grep ':{port} ' || true"],
            capture_output=True, text=True
        ).stdout.strip()
        if out == "":
            return True
        time.sleep(0.5)
    return False

def run_benchmark(server_name: str, model_name: str, count=100):
    # 1. Prepare Dataset
    print(f"Preparing {count} prompts from dataset...")
    try:
        prompts = get_prompts(count=count)
    except Exception as e:
        print(f"Failed to get prompts: {e}")
        return

    if not prompts:
        print("No prompts loaded.")
        return

    print(f"Loaded {len(prompts)} prompts for benchmarking.")

    # 2. Initialize Handler
    print(f"Initializing handler for {server_name} with model {model_name}...")
    if server_name == "fastllm":
        try:
            handler = FastLLMServerHandler(model_name, log_dir="./logs")
        except ValueError as e:
            print(f"Error: {e}")
            return
    elif server_name == "llama-cpp":
        try:
            handler = LlamaCppServerHandler(model_name, log_dir="./logs")
        except ValueError as e:
            print(f"Error: {e}")
            return
    else:
        print(f"Unknown framework: {server_name}")
        return

    # 3. Start Server
    try:
        handler.start_server()
        # Give a small buffer after start, though start_server waits for ready
        time.sleep(2)
    except Exception as e:
        print(f"Failed to start server: {e}")
        return

    # 4. Initialize OpenAI Client
    # FastLLM and llama-cpp usually default to port 8080
    base_url = "http://127.0.0.1:8080/v1"
    api_key = "sk-1234"
    client = openai.OpenAI(base_url=base_url, api_key=api_key)

    try:
        print(f"Connecting to server at {base_url}...")
        models_list = client.models.list()
        if not models_list.data:
            raise ValueError("No models found on the server.")

        # We might want to use the model_name passed in, or the one reported by server
        # Usually they match or are aliases.
        target_model_id = models_list.data[0].id
        print(f"Server reported model ID: {target_model_id}")

    except Exception as e:
        print(f"Connection error: {e}")
        handler.stop_server()
        return

    # 5. Configure Benchmark Parameters
    temperature = 0.0
    top_p = 1.0
    top_k = 1
    max_tokens = 128

    print("\nBenchmark Configuration:")
    print(f"  Framework:   {server_name}")
    print(f"  Model:       {model_name}")
    print(f"  Temperature: {temperature}")
    print(f"  Top-P:       {top_p}")
    print(f"  Top-K:       {top_k}")
    print(f"  Max Tokens:  {max_tokens}")
    print("-" * 80)
    print("Running benchmark...")
    print("-" * 80)

    total_requests = len(prompts)
    failed_requests = 0

    benchmark_start_time = time.time()

    try:
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]

            try:
                # Timing request duration (wall clock)
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
                    # For others (llama-cpp), use non-streaming or server-supplied timings
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
        wall_time_s = time.time() - benchmark_start_time

        # 6. Report Results
        print("Benchmark Results Summary:")
        print(f"Total Requests:           {total_requests}")
        print(f"Failed:                   {failed_requests}")
        print(f"Wall Time Elapsed:        {wall_time_s:.2f} s")
        print("-" * 80)

        tps = handler.get_result()
        print(f"Average Decode Throughput: {tps:.2f} tokens/s")
        print("-" * 80)

        handler.stop_server()
        wait_port_free(8080, timeout=30)


# python -m benchmark.throughput.main
if __name__ == "__main__":
    for model in [
        "GLM-4.5-Air",
        # "Qwen3-235B-A22B",
    ]:
        for server in [
            "llama-cpp",
            # "fastllm",
        ]:
            run_benchmark(server, model, 3)
