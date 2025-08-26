from evalscope.run import run_task
from evalscope.config import TaskConfig, EvalType
from gpu_recorder import GPURecorder

output_dir = "results/gsm8k"
model_name = "Qwen/Qwen3-30B-A3B"

task_config = TaskConfig(
    model=model_name,
    datasets=["gsm8k"],
    eval_type=EvalType.SERVICE,
    limit=5,
    eval_batch_size=1,
    api_url="http://127.0.0.1:8088/v1/chat/completions",
    api_key="sk-1234",
    generation_config={
        "max_tokens": 32768,
        "temperature": 0.0,
    },
    stream=True,
)

recorder = GPURecorder()
recorder.start()
run_task(task_cfg=task_config)
recorder.finish()
print(recorder.get_average_metrics())
print(f"评估完成，使用 python tests/eval/analysis.py {output_dir} 统计结果。")
