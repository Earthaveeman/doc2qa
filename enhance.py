"""语义增强流程：读取问答对 JSONL -> 逐条调用百炼语义增强应用 -> 校验拼接 -> 输出 fivefold 文件。

与原有"文本分块产问答"流程完全隔离，仅由 main.py --enhance 参数进入。
"""
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Callable, List

from bailian import BailianCallError, BailianClient
from chunker import EncodingError, read_text_file

logger = logging.getLogger("doc2qa")

# 每条输入预期产出的增强问答对数量（接口约定返回 5 个 JSON 对象）
EXPECTED_LINES = 5


class EnhanceInputError(Exception):
    """输入 JSONL 单行解析失败。"""


@dataclass
class EnhanceSummary:
    total: int = 0                      # 输入总条数
    ok: int = 0                         # 处理成功条数
    failed: int = 0                     # 处理失败条数（解析失败 + 调用失败）
    out_lines: int = 0                  # 产出输出总行数
    failures: List[str] = field(default_factory=list)


def fivefold_path(input_path: str) -> str:
    """输出文件名 = 原文件名 + _fivefold 后缀（保留原扩展名）。"""
    root, ext = os.path.splitext(input_path)
    return root + "_fivefold" + ext


def parse_qa_line(line: str) -> dict:
    """解析单条问答对 JSON 行，非法时抛 EnhanceInputError。"""
    try:
        obj = json.loads(line)
    except json.JSONDecodeError as e:
        raise EnhanceInputError(f"不是合法 JSON: {e.msg}; 片段: {line[:200]!r}") from e
    if not isinstance(obj, dict):
        raise EnhanceInputError(f"不是 JSON 对象: {line[:200]!r}")
    return obj


def run_enhance(input_path: str, client: BailianClient,
                console: Callable[[str], None]) -> int:
    """语义增强主流程。

    单条失败（解析失败 / 调用重试耗尽）跳过不中断；返回 0 全部成功，
    1 存在失败，2 输入文件级错误。
    """
    summary = EnhanceSummary()

    try:
        text = read_text_file(input_path)
    except EncodingError as e:
        console(f"[错误] 输入文件编码无法识别: {e}")
        return 2
    except OSError as e:
        console(f"[错误] 读取输入文件失败: {e}")
        return 2

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    summary.total = len(lines)
    out_path = fivefold_path(input_path)
    if summary.total == 0:
        console(f"[错误] 输入文件无有效内容: {input_path}")
        return 2
    console(f"输入 {input_path}: {summary.total} 条问答对; 输出: {out_path}")

    all_lines: List[dict] = []
    for idx, line in enumerate(lines, start=1):
        started = time.monotonic()
        # 1) 解析单条 JSON（失败跳过）
        try:
            obj = parse_qa_line(line)
        except EnhanceInputError as e:
            summary.failed += 1
            summary.failures.append(f"第 {idx} 条输入解析失败: {e}")
            console(f"[{idx}/{summary.total}] ✗ 输入解析失败，跳过: {e}")
            logger.error("第 %d 条输入解析失败: %s", idx, e)
            continue

        # 2) 调用语义增强应用（内部含重试；重试耗尽跳过）
        try:
            results = client.call(
                json.dumps(obj, ensure_ascii=False),
                on_retry=lambda attempt, total_r, reason, idx=idx: console(
                    f"    第 {idx} 条 ✗ 重试 {attempt}/{total_r} ({reason})"
                ),
            )
        except BailianCallError as e:
            summary.failed += 1
            summary.failures.append(f"第 {idx} 条调用失败: {e}")
            console(f"[{idx}/{summary.total}] ✗ 调用失败，跳过: {e}")
            logger.error("第 %d 条调用失败: %s", idx, e)
            continue

        # 3) 结果已在 client.call 内按原有逻辑校验（合法 JSONL 且含 prompt/response）
        if len(results) != EXPECTED_LINES:
            logger.warning("第 %d 条返回 %d 行（预期 %d）", idx, len(results), EXPECTED_LINES)
            console(f"    第 {idx} 条返回 {len(results)} 行（预期 {EXPECTED_LINES}）")
        all_lines.extend(results)
        summary.ok += 1
        summary.out_lines += len(results)
        console(f"[{idx}/{summary.total}] ✓ ({time.monotonic() - started:.1f}s, {len(results)} 行)")

    if not all_lines:
        console("全部条目处理失败，未生成输出文件")
        _print_summary(summary, out_path=None, console=console)
        return 1

    # 原子写入：先写 .tmp 再改名，保证不出现半截输出
    tmp_path = out_path + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8", newline="\n") as f:
            for obj in all_lines:
                f.write(json.dumps(obj, ensure_ascii=False))
                f.write("\n")
        os.replace(tmp_path, out_path)
    except OSError as e:
        logger.error("写入失败 %s: %s", out_path, e)
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        console(f"[错误] 写入输出文件失败: {e}")
        _print_summary(summary, out_path=None, console=console)
        return 1

    logging.info("语义增强完成: 输入 %d 条, 成功 %d, 失败 %d, 输出 %d 行 -> %s",
                 summary.total, summary.ok, summary.failed, summary.out_lines, out_path)
    _print_summary(summary, out_path=out_path, console=console)
    return 0 if summary.failed == 0 else 1


def _print_summary(summary: EnhanceSummary, out_path, console: Callable[[str], None]) -> None:
    print()
    print("=" * 60)
    print(f"语义增强完成: 总计 {summary.total} 条 | 成功 {summary.ok} | 失败 {summary.failed} | "
          f"产出 {summary.out_lines} 行")
    if out_path:
        print(f"输出文件: {out_path}")
    if summary.failed:
        print("失败条目:")
        for msg in summary.failures:
            print(f"  - {msg}")
    print("=" * 60)
