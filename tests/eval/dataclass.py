from pydantic import BaseModel, field_validator, ValidationError
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path


class Usage(BaseModel):
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int


class ModelSpec(BaseModel):
    api_url: str
    model_id: str
    api_key: str


class ReviewerSpec(BaseModel):
    metric: List[str]
    reviewer: List[str]
    revision: List[str]


class Message(BaseModel):
    content: str
    role: str


class Review(BaseModel):
    gold: str
    pred: str
    result: Union[int, float]


class RawInputMMLU(BaseModel):
    input: str
    A: str
    B: str
    C: str
    D: str
    target: str


class RawInputGsm8k(BaseModel):
    question: str
    answer: str


RAW_INPUT_TYPES = (RawInputMMLU, RawInputGsm8k)
RawInput = Union[*RAW_INPUT_TYPES]


class Choice(BaseModel):
    finish_reason: str
    index: int
    message: Message
    review: Review


class ReviewResult(BaseModel):
    id: str
    choices: List[Choice]
    created: int
    model: str
    object: str
    usage: Usage
    model_spec: ModelSpec
    answer_id: str
    subset_name: str
    raw_input: RawInput
    index: int
    reviewed: bool
    review_id: str
    reviewer_spec: ReviewerSpec
    review_time: float

    # 可选字段
    system_fingerprint: Optional[str] = None
    timings: Optional[Dict[str, Any]] = None

    @field_validator("raw_input", mode="before")
    @classmethod
    def validate_raw_input(cls, v: Any):
        if not isinstance(v, dict):
            raise ValueError("raw_input 必须是一个字典")

        for model_cls in RAW_INPUT_TYPES:
            try:
                return model_cls(**v)
            except ValidationError:
                continue

        known_types = ", ".join([m.__name__ for m in RAW_INPUT_TYPES])
        raise ValueError(
            f"未知的 raw_input 结构: 无法匹配任何已知的类型 ({known_types})"
        )


def load_from_jsonl_pydantic(jsonl_path: str) -> List[ReviewResult]:
    """
    从 JSONL 文件中加载数据，并使用 Pydantic V2 模型进行解析和验证。
    """
    data_items = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                item = ReviewResult.model_validate_json(line)
                data_items.append(item)
            except ValidationError as e:
                print(f"警告：第 {line_num} 行数据验证失败，已跳过。\n错误详情: {e}")
            except Exception as e:
                print(f"警告：处理第 {line_num} 行时发生未知错误: {e}")
    return data_items


def load_jsonl_files(directory_path: str) -> List[ReviewResult]:
    """
    搜索指定目录及其所有子目录下的 .jsonl 文件，将它们的内容合并到一个列表中。

    :param directory_path: 要搜索的目录路径。
    :return: 一个包含所有 DataItem 对象的列表。
    """

    target_directory = Path(directory_path)

    # 检查目录是否存在
    if not target_directory.is_dir():
        print(f"错误：目录 '{directory_path}' 不存在或不是一个目录。")
        return []

    jsonl_files = list(target_directory.rglob("*.jsonl"))

    if not jsonl_files:
        print(f"提示：在目录 '{directory_path}' 及其子目录中未找到任何 .jsonl 文件。")
        return []

    print(
        f"在 '{directory_path}' 中找到 {len(jsonl_files)} 个 .jsonl 文件，开始合并..."
    )

    all_data_items = []
    for file_path in jsonl_files:
        print(f"  -> 正在处理文件: {file_path.relative_to(target_directory)}")
        # 调用单文件处理函数
        data_from_file = load_from_jsonl_pydantic(str(file_path))
        # 使用 extend() 将当前文件的数据合并到总列表中
        all_data_items.extend(data_from_file)

    print(f"\n合并完成！总共收集了 {len(all_data_items)} 条数据。")
    return all_data_items
