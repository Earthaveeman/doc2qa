"""doc2qa 入口：批量扫描文本文件 -> 段落分块 -> 逐块调用百炼智能体 -> 原子写入同名 .jsonl。

语义增强模式（--enhance）：读取问答对 JSONL -> 逐条调用百炼语义增强应用 -> 结果写入 *_fivefold.jsonl。

用法:
    python main.py                    # 无参数：打印帮助信息并退出
    python main.py --generate         # 问答对生成流程，使用 .env 中的 INPUT_DIR
    python main.py --generate --input ./docs   # 问答对生成流程，覆盖输入目录
    python main.py --enhance ./qa.jsonl   # 语义增强流程，指定输入文件
    python main.py --enhance              # 语义增强流程，使用 .env 的 ENHANCE_INPUT
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
from enhance import run_enhance

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
    # Windows GBK 控制台兼容：无法编码的字符替换显示，避免 ✓/✗ 等符号导致崩溃
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
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


def _make_console():
    def console(msg: str) -> None:
        print(msg, flush=True)
        logging.info("%s", msg.rstrip())
    return console


def run_enhance_mode(enhance_arg: str) -> int:
    """语义增强流程入口：与问答对生成流程完全隔离。"""
    try:
        config = load_config(require_main_app=False)
    except ConfigError as e:
        print(f"[配置错误] {e}", file=sys.stderr)
        return 2

    if not config.enhance_app_id:
        print("[配置错误] 语义增强模式缺少必填配置: ENHANCE_APP_ID", file=sys.stderr)
        return 2

    input_path = enhance_arg or config.enhance_input
    if not input_path:
        print("[配置错误] 未指定输入文件：请通过 --enhance 传入，或在 .env 中配置 ENHANCE_INPUT",
              file=sys.stderr)
        return 2
    input_path = os.path.abspath(input_path)
    if not os.path.isfile(input_path):
        print(f"[错误] 输入文件不存在: {input_path}", file=sys.stderr)
        return 2

    setup_logging()
    console = _make_console()

    # 独立应用 ID：复用同一 API Key 与请求/重试/校验逻辑，仅指向语义增强应用
    client = BailianClient(config, app_id=config.enhance_app_id)
    console(f"语义增强模式: 输入 {input_path}（应用 ID: {config.enhance_app_id}）")
    return run_enhance(input_path, client, console)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="批量文本分块并调用阿里云百炼智能体，结果写入同名 .jsonl；"
                    "--enhance 进入语义增强流程",
        epilog="示例:\n"
               "  python main.py --generate --input ./docs   # 问答对生成流程：分块产问答对\n"
               "  python main.py --generate                 # 问答对生成流程：使用 .env 的 INPUT_DIR\n"
               "  python main.py --enhance ./qa.jsonl       # 语义增强流程：指定输入文件\n"
               "  python main.py --enhance                  # 语义增强流程：使用 .env 的 ENHANCE_INPUT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="问答对生成流程：批量扫描文本文件 -> 段落分块 -> 调用百炼智能体 -> 问答对写入同名 .jsonl",
    )
    parser.add_argument("--input", help="问答对生成流程的输入目录（默认取 .env 的 INPUT_DIR）")
    parser.add_argument(
        "--enhance",
        nargs="?",
        const="",
        default=None,
        metavar="FILE",
        help="语义增强模式：传入问答对 JSONL 文件路径（省略时取 .env 的 ENHANCE_INPUT）",
    )
    args = parser.parse_args(argv)

    # 参数分流：未传任何模式参数时打印帮助并退出
    if args.enhance is None and not args.generate:
        parser.print_help()
        print("\n[提示] 请至少指定一种执行模式：--generate（问答对生成流程）或 --enhance（语义增强流程）")
        return 2

    if args.enhance is not None:
        return run_enhance_mode(args.enhance)

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
