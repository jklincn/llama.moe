from evalscope.run import run_task
from evalscope.config import TaskConfig, EvalType

output_dir = "results"
model_name = "Qwen/Qwen3-30B-A3B"

task_config = TaskConfig(
    model=model_name,
    datasets=["aime24"],
    eval_type=EvalType.SERVICE,
    limit=10,
    eval_batch_size=1,
    api_url="http://127.0.0.1:8088/v1/chat/completions",
    api_key="sk-1234",
    use_cache=output_dir,
    generation_config={
        "max_tokens": 16384,
        "temperature": 0.0,
    },
)

run_task(task_cfg=task_config)

print(f"评估完成，使用 python tests/eval/analysis.py {output_dir} 统计结果。")
