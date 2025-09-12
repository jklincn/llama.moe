from evalscope.run import run_task
from evalscope.config import TaskConfig, EvalType

output_dir = "outputs/Qwen3-30B-A3B-origin"
model_name = "Qwen/Qwen3-30B-A3B"

task_config = TaskConfig(
    model=model_name,
    datasets=["mmlu", "gsm8k"],
    dataset_args={
        "mmlu": {
            "few_shot_num": 5,
        },
    },
    eval_type=EvalType.SERVICE,
    # limit=10,
    eval_batch_size=1,
    api_url="http://127.0.0.1:8088/v1/chat/completions",
    api_key="sk-1234",
    use_cache=output_dir,
    generation_config={
        "max_tokens": 32768,
        "temperature": 0.0,
        "stream": True,
    },
)

run_task(task_cfg=task_config)

print(f"评估完成，使用 python tests/eval/utils/analysis.py {output_dir} 统计结果。")
