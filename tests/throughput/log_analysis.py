import re


def log_analysis():
    # 只匹配目标两行
    pat_prompt = re.compile(
        r"prompt eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens"
    )
    pat_eval = re.compile(r"^\s*eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens")

    prompt_time_total = 0.0  # ms
    prompt_tokens_total = 0
    eval_time_total = 0.0  # ms
    eval_tokens_total = 0

    n_reqs = 0
    have_prompt_for_current = False  # 状态：本次请求是否已经看到 prompt 行

    with open("llama-server.log", "r", encoding="utf-8") as f:
        for line in f:
            m1 = pat_prompt.search(line)
            if m1:
                time_ms = float(m1.group(1))
                toks = int(m1.group(2))
                prompt_time_total += time_ms
                prompt_tokens_total += toks
                have_prompt_for_current = True
                continue

            m2 = pat_eval.search(line)
            if m2:
                time_ms = float(m2.group(1))
                toks = int(m2.group(2))
                eval_time_total += time_ms
                eval_tokens_total += toks

                # 只有在同一个片段中先看到 prompt 再看到 eval，才算一个请求完成
                if have_prompt_for_current:
                    n_reqs += 1
                    have_prompt_for_current = False
                continue

    # 吞吐率（tokens/s）
    prompt_perf = (
        (prompt_tokens_total / (prompt_time_total / 1000))
        if prompt_time_total > 0
        else 0.0
    )
    eval_perf = (
        (eval_tokens_total / (eval_time_total / 1000)) if eval_time_total > 0 else 0.0
    )

    print("=== 累计统计结果 ===")
    print(f"请求数: {n_reqs}")
    print(
        f"Prompt eval: {prompt_tokens_total} tokens, {prompt_time_total / 1000:.2f} s, 吞吐率 {prompt_perf:.2f} tokens/s"
    )
    print(
        f"Eval:        {eval_tokens_total} tokens, {eval_time_total / 1000:.2f} s, 吞吐率 {eval_perf:.2f} tokens/s"
    )

    if n_reqs > 0:
        print("=== 平均每个请求 ===")
        print(f"平均输入长度: {prompt_tokens_total / n_reqs:.2f} tokens/req")
        print(f"平均生成长度: {eval_tokens_total / n_reqs:.2f} tokens/req")
        print(f"平均输入处理时间: {prompt_time_total / n_reqs / 1000:.2f} s/req")
        print(f"平均生成处理时间: {eval_time_total / n_reqs / 1000:.2f} s/req")
    else:
        print("未提取到任何请求片段。")
