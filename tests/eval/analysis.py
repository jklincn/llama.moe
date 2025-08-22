import json
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import argparse
import sys


@dataclass
class Message:
    content: str
    role: str


@dataclass
class Choice:
    finish_reason: str
    index: int
    message: Message
    review: Dict[str, Any]


@dataclass
class Usage:
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int


@dataclass
class DataItem:
    id: str
    object: str
    model: str
    created: int
    system_fingerprint: str
    choices: List[Choice]
    usage: Usage
    timings: Dict[str, Any]
    model_spec: Dict[str, Any]
    answer_id: str
    subset_name: str
    raw_input: Dict[str, Any]
    index: int
    reviewed: bool
    review_id: str
    reviewer_spec: Dict[str, Any]
    review_time: float


def find_jsonl_files(query: str) -> List[Path]:
    """
    根据 query 查找 .jsonl 文件：
    - 若 query 是存在的目录：递归寻找该目录下所有 .jsonl
    - 若 query 是存在的文件且以 .jsonl 结尾：返回该文件
    - 否则，将 query 视为子字符串，从当前目录起递归查找路径中包含该子串的 .jsonl
    """
    p = Path(query) / "reviews"
    results: List[Path] = []

    if p.exists():
        if p.is_dir():
            results = sorted(p.rglob("*.jsonl"))
        elif p.is_file() and p.suffix.lower() == ".jsonl":
            results = [p.resolve()]
        else:
            # 存在但不是jsonl文件：不返回
            results = []
    else:
        # 子字符串匹配：从cwd递归找
        cwd = Path.cwd()
        for fp in cwd.rglob("*.jsonl"):
            # 在完整路径字符串中做子串判断（大小写敏感）
            if query in str(fp):
                results.append(fp.resolve())
        results.sort()

    return results


def load_jsonl(filepath: Path) -> List[DataItem]:
    data_items: List[DataItem] = []
    with filepath.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue

            try:
                raw = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"第{line_num}行JSON解析失败: {e}")

            try:
                # 处理 choices
                choices: List[Choice] = []
                for i, c in enumerate(raw["choices"]):
                    try:
                        msg_dict = c["message"]
                        message = Message(
                            content=msg_dict["content"],
                            role=msg_dict["role"],
                        )
                        choice = Choice(
                            finish_reason=c["finish_reason"],
                            index=c["index"],
                            message=message,
                            review=c["review"],
                        )
                        choices.append(choice)
                    except KeyError as e:
                        raise ValueError(f"第{line_num}行choices[{i}]中缺少字段: {e}")

                # 处理 usage
                try:
                    u = raw["usage"]
                    usage_obj = Usage(
                        completion_tokens=u["completion_tokens"],
                        prompt_tokens=u["prompt_tokens"],
                        total_tokens=u["total_tokens"],
                    )
                except KeyError as e:
                    raise ValueError(f"第{line_num}行usage中缺少字段: {e}")

                # 创建主要对象
                try:
                    item = DataItem(
                        id=raw["id"],
                        object=raw["object"],
                        model=raw["model"],
                        created=raw["created"],
                        system_fingerprint=raw["system_fingerprint"],
                        choices=choices,
                        usage=usage_obj,
                        timings=raw["timings"],
                        model_spec=raw["model_spec"],
                        answer_id=raw["answer_id"],
                        subset_name=raw["subset_name"],
                        raw_input=raw["raw_input"],
                        index=raw["index"],
                        reviewed=raw["reviewed"],
                        review_id=raw["review_id"],
                        reviewer_spec=raw["reviewer_spec"],
                        review_time=raw["review_time"],
                    )
                    data_items.append(item)
                except KeyError as e:
                    raise ValueError(f"第{line_num}行根级别缺少字段: {e}")
                except TypeError as e:
                    raise ValueError(f"第{line_num}行数据类型不匹配: {e}")

            except ValueError:
                raise  # 重新抛出我们自定义的异常
            except Exception as e:
                raise ValueError(f"第{line_num}行解析失败: {e}")

    return data_items


def load_multiple(files: List[Path]) -> List[Tuple[Path, List[DataItem]]]:
    """
    依次解析多个 jsonl 文件，返回 (路径, 解析后的数组) 的列表。
    """
    results: List[Tuple[Path, List[DataItem]]] = []
    for fp in files:
        try:
            dataset = load_jsonl(fp)
            results.append((fp, dataset))
        except Exception as e:
            print(f"[ERROR] 解析失败: {fp} -> {e}", file=sys.stderr)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="根据子字符串/路径查找并解析 JSONL 文件"
    )
    parser.add_argument(
        "query",
        help="可以是目录/具体jsonl文件路径，或用于匹配路径的子字符串（例如：outputs/eval）",
    )
    args = parser.parse_args()

    files = find_jsonl_files(args.query)

    if not files:
        print(f"未找到匹配的 .jsonl 文件：{args.query}")
        sys.exit(1)

    print("找到以下 .jsonl 文件：")
    for fp in files:
        print(f"- {fp}")

    parsed = load_multiple(files)

    total = sum(len(arr) for _, arr in parsed)
    print(f"\n共解析 {len(parsed)} 个文件，总计 {total} 条数据")

    for fp, arr in parsed:
        print(f"\n文件: {fp}")
        if arr:
            print(f"  该文件共 {len(arr)} 条，示例第一条：")
            print(f"  {arr[0]}")
        else:
            print("  该文件为空或未能解析出条目。")
