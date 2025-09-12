from dataclass import load_jsonl_files, ReviewResult


def perform_data_validation(data_list: list[ReviewResult]):
    total_items = len(data_list)
    incorrect = 0
    over_length = 0

    for item in data_list:
        for choice in item.choices:
            if choice.finish_reason != "stop":
                # print(
                #     f"{item.subset_name} {item.index} 结束原因异常: '{choice.finish_reason}'"
                # )
                over_length += 1
                incorrect += 1

    for item in data_list:
        for choice in item.choices:
            if choice.review.result not in (1, 1.0):
                # print(
                #     f"{item.subset_name} {item.index} 回答错误, 做出答案 {choice.review.pred}，正确答案 {choice.review.gold}"
                # )
                incorrect += 1

    correct = total_items - incorrect

    # 计算平均 completion_token
    completion_tokens_list = [item.usage.completion_tokens for item in data_list]
    total_completion_tokens = sum(completion_tokens_list)
    total_items = len(data_list)
    average_completion_tokens = total_completion_tokens / total_items

    print("\n--- 验证完成 ---")
    print(
        "总计 {} 条数据，正确 {} 条，错误 {} 条 (其中 {} 条超出上下文长度), 正确率 {:.2f}%".format(
            total_items, correct, incorrect, over_length, (correct) / total_items * 100
        )
    )
    print(f"所有数据的 completion_tokens 平均值为: {average_completion_tokens:.2f}")


def main():
    data_list = load_jsonl_files("results/Qwen3-30B-A3B-origin/reviews/Qwen3-30B-A3B")

    perform_data_validation(data_list)


if __name__ == "__main__":
    main()
