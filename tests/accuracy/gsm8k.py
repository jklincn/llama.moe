from evalscope.run import run_task
from evalscope.config import TaskConfig, EvalType

output_dir = "results/gsm8k"
model_name = "Qwen/Qwen3-30B-A3B"

task_config = TaskConfig(
    model=model_name,
    datasets=["gsm8k"],
    eval_type=EvalType.SERVICE,
    eval_batch_size=1,
    api_url="http://127.0.0.1:8080/v1/chat/completions",
    api_key="sk-1234",
    generation_config={
        "max_tokens": 32768,
        "temperature": 0.0,
    },
    use_cache=output_dir,
    stream=True,
)

run_task(task_cfg=task_config)

print(f"评估完成，使用 python tests/eval/utils/analysis.py {output_dir} 统计结果。")
