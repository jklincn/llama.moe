from mmengine.config import read_base
from opencompass.models import OpenAISDK
from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLEvalTask, OpenICLInferTask

with read_base():
    from opencompass.configs.datasets.mmlu_cf.mmlu_cf_gen_040615 import mmlu_cf_datasets
    from opencompass.configs.summarizers.mmlu_cf import summarizer


models = [
    dict(
        type=OpenAISDK,
        path="/mnt/data/gguf/Qwen3-30B-A3B-Q8_0.gguf",
        key="sk-1234",
        openai_api_base="http://127.0.0.1:8088/v1",
        abbr="Qwen3-30B-A3B-Q8_0-on-mmlu-cf",
        max_seq_len=4096,
        max_out_len=1024,
        batch_size=4,
        retry=3,
    )
]

datasets = mmlu_cf_datasets


infer = dict(
    partitioner=dict(type=NumWorkerPartitioner, num_worker=8),
    runner=dict(type=LocalRunner, max_num_workers=8, task=dict(type=OpenICLInferTask)),
)

eval = dict(
    partitioner=dict(type=NaivePartitioner, n=10),
    runner=dict(type=LocalRunner, max_num_workers=256, task=dict(type=OpenICLEvalTask)),
)

work_dir = "outputs/my_model_on_mmlu_cf"
