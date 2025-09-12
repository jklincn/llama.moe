from evalscope.config import EvalType, TaskConfig
from evalscope.run import run_task

def run_eval():
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
            "stream": True,
        },
    )

    run_task(task_cfg=task_config)
