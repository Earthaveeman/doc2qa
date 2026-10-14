"""doc2qa 入口：批量扫描文本文件 -> 段落分块 -> 逐块调用百炼智能体 -> 原子写入同名 .jsonl。

用法:
    python main.py                    # 使用 .env 中的 INPUT_DIR
    python main.py --input ./docs     # 覆盖输入目录
"""
import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from typing import List

from bailian import BailianCallError, BailianClient
from chunker import EncodingError, chunk_text, read_text_file
from config import ConfigError, load_config

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(BASE_DIR, "logs")


@dataclass
class FileResult:
    status: str            # "ok" | "skip" | "fail" | "empty"
    rel_path: str
    detail: str = ""


@dataclass
class RunSummary:
    results: List[FileResult] = field(default_factory=list)

    def count(self, status: str) -> int:
        return sum(1 for r in self.results if r.status == status)


def setup_logging() -> str:
    os.makedirs(LOGS_DIR, exist_ok=True)
    log_file = os.path.join(LOGS_DIR, time.strftime("run_%Y%m%d_%H%M%S.log"))
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
        force=True,
    )
    return log_file


def scan_text_files(input_dir: str, extensions: List[str]) -> List[str]:
    """递归扫描输入目录，返回待处理文本文件绝对路径（排除隐藏目录、已有同名 .jsonl 的文件）。"""
    files = []
    for root, dirs, names in os.walk(input_dir):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for name in names:
            if name.startswith("."):
                continue
            ext = os.path.splitext(name)[1].lower()
            if ext not in extensions:
                continue
            files.append(os.path.join(root, name))
    return sorted(files)


def process_file(path: str, rel_path: str, client: BailianClient,
                 max_chars: int, console) -> FileResult:
    """处理单个文件：读取 -> 分块 -> 逐块调用 -> 原子写入 .jsonl。"""
    try:
        text = read_text_file(path)
    except EncodingError as e:
        logging.error("读取失败 %s: %s", rel_path, e)
        return FileResult("fail", rel_path, f"编码错误: {e}")
    except OSError as e:
        logging.error("读取失败 %s: %s", rel_path, e)
        return FileResult("fail", rel_path, f"读取错误: {e}")

    chunks = chunk_text(text, max_chars)
    if not chunks:
        logging.info("空文件跳过 %s", rel_path)
        return FileResult("empty", rel_path, "无有效内容（0 块）")

    console(f"  {rel_path} → {len(chunks)} 块")
    all_lines: List[dict] = []
    for idx, chunk in enumerate(chunks, start=1):
        started = time.monotonic()
        try:
            lines = client.call(
                chunk,
                on_retry=lambda attempt, total, reason, idx=idx: console(
                    f"    块 {idx}/{len(chunks)} ✗ 重试 {attempt}/{total} ({reason})"
                ),
            )
        except BailianCallError as e:
            logging.error("块调用失败 %s 块 %d/%d: %s", rel_path, idx, len(chunks), e)
            return FileResult("fail", rel_path, f"块 {idx}/{len(chunks)} 调用失败: {e}")
        all_lines.extend(lines)
        console(f"    块 {idx}/{len(chunks)} ✓ ({time.monotonic() - started:.1f}s, {len(lines)} 行)")

    # 原子写入：先写 .tmp 再改名，保证不出现半截 .jsonl 被误判为已完成
    out_path = os.path.splitext(path)[0] + ".jsonl"
    tmp_path = out_path + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8", newline="\n") as f:
            for obj in all_lines:
                f.write(json.dumps(obj, ensure_ascii=False))
                f.write("\n")
        os.replace(tmp_path, out_path)
    except OSError as e:
        logging.error("写入失败 %s: %s", rel_path, e)
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        return FileResult("fail", rel_path, f"写入错误: {e}")

    rel_out = os.path.splitext(rel_path)[0] + ".jsonl"
    logging.info("完成 %s: %d 块, %d 行 -> %s", rel_path, len(chunks), len(all_lines), rel_out)
    return FileResult("ok", rel_path, f"{len(chunks)} 块 {len(all_lines)} 行 → {rel_out}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="批量文本分块并调用阿里云百炼智能体，结果写入同名 .jsonl"
    )
    parser.add_argument("--input", help="输入目录（默认取 .env 的 INPUT_DIR）")
    args = parser.parse_args(argv)

    try:
        config = load_config()
    except ConfigError as e:
        print(f"[配置错误] {e}", file=sys.stderr)
        return 2

    input_dir = args.input or config.input_dir
    if not os.path.isdir(input_dir):
        print(f"[错误] 输入目录不存在: {input_dir}", file=sys.stderr)
        return 2

    log_file = setup_logging()

    def console(msg: str) -> None:
        print(msg, flush=True)
        logging.info("%s", msg.rstrip())

    client = BailianClient(config)
    summary = RunSummary()
    files = scan_text_files(input_dir, config.text_extensions)
    total = len(files)
    console(f"扫描 {input_dir}: {total} 个文本文件; 日志: {os.path.relpath(log_file, BASE_DIR)}")

    for i, path in enumerate(files, start=1):
        rel_path = os.path.relpath(path, input_dir)
        out_path = os.path.splitext(path)[0] + ".jsonl"
        if os.path.exists(out_path):
            console(f"[{i}/{total}] {rel_path} SKIP（已存在 {os.path.basename(out_path)}）")
            summary.results.append(FileResult("skip", rel_path, "已存在同名 .jsonl"))
            continue
        console(f"[{i}/{total}] {rel_path}")
        result = process_file(path, rel_path, client, config.chunk_max_chars, console)
        if result.status == "ok":
            console(f"[{i}/{total}] {rel_path} ✓ 完成 → {os.path.splitext(rel_path)[0]}.jsonl")
        elif result.status == "empty":
            console(f"[{i}/{total}] {rel_path} EMPTY（无有效内容，未产出 .jsonl）")
        else:
            console(f"[{i}/{total}] {rel_path} ✗ 失败: {result.detail}")
        summary.results.append(result)

    # 汇总
    print()
    print("=" * 60)
    print(f"处理完成: 成功 {summary.count('ok')} | 跳过 {summary.count('skip')} | "
          f"空文件 {summary.count('empty')} | 失败 {summary.count('fail')}")
    if summary.count("fail"):
        print("失败文件:")
        for r in summary.results:
            if r.status == "fail":
                print(f"  - {r.rel_path}: {r.detail}")
    print("=" * 60)
    return 0 if summary.count("fail") == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
