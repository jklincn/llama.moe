import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger("log_analysis")


@dataclass
class PerformanceStats:
    """性能统计数据类"""

    n_requests: int = 0
    prompt_tokens_total: int = 0
    prompt_time_total: float = 0.0  # ms
    eval_tokens_total: int = 0
    eval_time_total: float = 0.0  # ms

    @property
    def prompt_throughput(self) -> float:
        """Prompt处理吞吐率 (tokens/s)"""
        if self.prompt_time_total > 0:
            return self.prompt_tokens_total / (self.prompt_time_total / 1000)
        return 0.0

    @property
    def eval_throughput(self) -> float:
        """生成吞吐率 (tokens/s)"""
        if self.eval_time_total > 0:
            return self.eval_tokens_total / (self.eval_time_total / 1000)
        return 0.0

    @property
    def avg_prompt_tokens_per_request(self) -> float:
        """平均每请求输入长度"""
        return (
            self.prompt_tokens_total / self.n_requests if self.n_requests > 0 else 0.0
        )

    @property
    def avg_eval_tokens_per_request(self) -> float:
        """平均每请求生成长度"""
        return self.eval_tokens_total / self.n_requests if self.n_requests > 0 else 0.0

    @property
    def avg_prompt_time_per_request(self) -> float:
        """平均每请求输入处理时间 (s)"""
        return (
            (self.prompt_time_total / self.n_requests / 1000)
            if self.n_requests > 0
            else 0.0
        )

    @property
    def avg_eval_time_per_request(self) -> float:
        """平均每请求生成时间 (s)"""
        return (
            (self.eval_time_total / self.n_requests / 1000)
            if self.n_requests > 0
            else 0.0
        )


def log_analysis(log_path: str) -> Optional[PerformanceStats]:
    """分析llama-server日志文件

    Args:
        log_path: 日志文件路径

    Returns:
        性能统计对象，如果分析失败返回None
    """
    log_file = Path(log_path)
    if not log_file.exists():
        print(f"错误: 日志文件不存在 - {log_path}")
        return None

    # 编译正则表达式模式
    pat_prompt = re.compile(
        r"prompt eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens"
    )
    pat_eval = re.compile(r"^\s*eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens")

    stats = PerformanceStats()
    have_prompt_for_current = False  # 状态：本次请求是否已经看到 prompt 行

    try:
        with open(log_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                # 匹配prompt eval行
                m1 = pat_prompt.search(line)
                if m1:
                    try:
                        time_ms = float(m1.group(1))
                        tokens = int(m1.group(2))
                        stats.prompt_time_total += time_ms
                        stats.prompt_tokens_total += tokens
                        have_prompt_for_current = True
                    except ValueError as e:
                        print(f"警告: 第{line_num}行数据解析错误 - {e}")
                    continue

                # 匹配eval行
                m2 = pat_eval.search(line)
                if m2:
                    try:
                        time_ms = float(m2.group(1))
                        tokens = int(m2.group(2))
                        stats.eval_time_total += time_ms
                        stats.eval_tokens_total += tokens

                        # 只有在同一个片段中先看到 prompt 再看到 eval，才算一个请求完成
                        if have_prompt_for_current:
                            stats.n_requests += 1
                            have_prompt_for_current = False
                    except ValueError as e:
                        logger.warning(f"警告: 第{line_num}行数据解析错误 - {e}")
                    continue

    except Exception as e:
        logger.error(f"错误: 读取日志文件失败 - {e}")
        return None

    # 打印统计结果
    _print_stats(stats)

    return stats


def _print_stats(stats: PerformanceStats) -> None:
    """打印性能统计结果

    Args:
        stats: 性能统计对象
    """
    logging.info("===================== 性能统计 =====================")
    if stats.n_requests == 0:
        logging.warning("未提取到任何完整的请求数据")
        return

    logging.info(f"总请求数:        {stats.n_requests}")
    logging.info(f"总输入tokens:    {stats.prompt_tokens_total}")
    logging.info(f"总生成tokens:    {stats.eval_tokens_total}")

    logging.info("--- 吞吐率统计 ---")
    logging.info(f"输入处理吞吐率:  {stats.prompt_throughput:.2f} tokens/s")
    logging.info(f"生成吞吐率:      {stats.eval_throughput:.2f} tokens/s")

    logging.info("--- 时间统计 ---")
    logging.info(f"总输入处理时间:  {stats.prompt_time_total / 1000:.2f} s")
    logging.info(f"总生成时间:      {stats.eval_time_total / 1000:.2f} s")

    logging.info("--- 平均每请求统计 ---")
    logging.info(
        f"平均输入长度:    {stats.avg_prompt_tokens_per_request:.2f} tokens/req"
    )
    logging.info(f"平均生成长度:    {stats.avg_eval_tokens_per_request:.2f} tokens/req")
    logging.info(f"平均输入处理时间: {stats.avg_prompt_time_per_request:.3f} s/req")
    logging.info(f"平均生成时间:    {stats.avg_eval_time_per_request:.3f} s/req")
    logging.info("================================================")


# 分析单个日志文件
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        log_file = sys.argv[1]
    else:
        log_file = "llama-server.log"

    print(f"正在分析日志文件: {log_file}")
    stats = log_analysis(log_file)

    if stats is None:
        print("日志分析失败")
        sys.exit(1)
    elif stats.n_requests == 0:
        print("未找到有效的性能数据")
        sys.exit(1)
    else:
        print("日志分析完成")
