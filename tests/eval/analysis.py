from dataclass import load_jsonl_files, ReviewResult


def perform_data_validation(data_list: list[ReviewResult]):
    """
    对一组 DataItem 对象执行自定义数据验证，并输出警告。

    主要检查项：
    1. choices 列表中的 finish_reason 是否都为 "stop"。
    2. choices 中 review 的 result 是否都为 1 或 1.0。

    :param data_list: 一个包含 DataItem 对象的列表。
    """
    if not data_list:
        print("警告：传入的数据列表为空，没有数据可供验证。")
        return

    print("\n--- 开始执行自定义数据验证 ---")

    # 统计异常条目数
    finish_reason_warnings = 0
    review_result_warnings = 0

    for item in data_list:
        # 遍历 choices 列表，对每个 choice 进行检查
        for choice in item.choices:
            # 检查 finish_reason 字段
            if choice.finish_reason != "stop":
                print(
                    f"{item.subset_name} {item.index} 结束原因为: '{choice.finish_reason}'"
                )
                finish_reason_warnings += 1

            # 检查 review.result 字段
            if choice.review.result not in (1, 1.0):
                print(
                    f"{item.subset_name} {item.index} 回答错误, 做出答案 {choice.review.pred}，正确答案 {choice.review.gold}"
                )
                review_result_warnings += 1

    print("\n--- 验证完成 ---")
    print(f"总计发现 {finish_reason_warnings} 条 finish_reason 异常。")
    print(f"总计发现 {review_result_warnings} 条 review.result 异常。")


def main():
    data_list = load_jsonl_files("results/Qwen3-30B-A3B-origin/reviews/Qwen3-30B-A3B")

    completion_tokens_list = [item.usage.completion_tokens for item in data_list]
    total_completion_tokens = sum(completion_tokens_list)
    total_items = len(data_list)

    # 计算平均值
    average_completion_tokens = total_completion_tokens / total_items

    print(f"所有数据的 completion_tokens 平均值为: {average_completion_tokens:.2f}")

    perform_data_validation(data_list)


if __name__ == "__main__":
    main()
