import re

results = []

# 匹配 "xxxx ms / yyy tokens"
pattern = re.compile(r"=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens")

# 累加统计
prompt_time_total = 0.0
prompt_tokens_total = 0
eval_time_total = 0.0
eval_tokens_total = 0

with open("llama-server.log", "r", encoding="utf-8") as f:
    buffer = []
    for line in f:
        match = pattern.search(line)
        if match:
            time_val = float(match.group(1))  # 毫秒
            token_val = int(match.group(2))
            buffer.extend([time_val, token_val])

            if "prompt eval" in line:
                prompt_time_total += time_val
                prompt_tokens_total += token_val
            elif "eval time" in line and "prompt" not in line:
                eval_time_total += time_val
                eval_tokens_total += token_val

            # 每个片段两行 → 收集四个数
            if len(buffer) == 4:
                results.append(buffer)
                buffer = []
# fmt: off
prompt_perf = (prompt_tokens_total / (prompt_time_total / 1000)) if prompt_time_total > 0 else 0.0
eval_perf = (eval_tokens_total / (eval_time_total / 1000)) if eval_time_total > 0 else 0.0

n_reqs = len(results)

print("\n=== 累计统计结果 ===")
print(f"请求数: {n_reqs}")
print(f"Prompt eval: {prompt_tokens_total} tokens, {prompt_time_total / 1000:.2f} s, 吞吐率 {prompt_perf:.2f} tokens/s")
print(f"Eval:        {eval_tokens_total} tokens, {eval_time_total / 1000:.2f} s, 吞吐率 {eval_perf:.2f} tokens/s")
# fmt: on

if n_reqs > 0:
    avg_prompt_tokens = prompt_tokens_total / n_reqs
    avg_eval_tokens = eval_tokens_total / n_reqs
    avg_prompt_time_s = prompt_time_total / n_reqs / 1000
    avg_eval_time_s = eval_time_total / n_reqs / 1000

    print("\n=== 平均每个请求 ===")
    print(f"平均输入长度: {avg_prompt_tokens:.2f} tokens/req")
    print(f"平均生成长度: {avg_eval_tokens:.2f} tokens/req")
    print(f"平均输入处理时间: {avg_prompt_time_s:.2f} s/req")
    print(f"平均生成处理时间: {avg_eval_time_s:.2f} s/req")
else:
    print("\n未提取到任何请求片段。")
