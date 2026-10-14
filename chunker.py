"""段落分块：按原生段落边界合并，超长段落段内切分，严格保证每块 <= max_chars。"""
import re
from typing import List, Optional, Tuple

# 句子结束符（中英文），用于超长段落的段内切分
_SENTENCE_END = "。！？；!?:;…～~"


class EncodingError(Exception):
    """文件编码无法识别（UTF-8 与 GBK 均失败）。"""


def read_text_file(path: str) -> str:
    """读取文本文件：UTF-8(含BOM) 优先，GBK 回退；统一换行符为 \\n。"""
    raw = open(path, "rb").read()
    for encoding in ("utf-8-sig", "gbk"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise EncodingError(f"无法以 UTF-8 或 GBK 解码: {path}")
    return text.replace("\r\n", "\n").replace("\r", "\n")


def split_paragraphs(text: str) -> List[str]:
    """按原生段落边界（空行）切分段落，段落内保留原始单行结构。"""
    paragraphs = re.split(r"\n[ \t]*\n+", text)
    result = [p.strip("\n") for p in paragraphs]
    return [p for p in result if p.strip()]


def _split_long_paragraph(paragraph: str, max_chars: int) -> List[str]:
    """切分超过 max_chars 的段落：优先句子边界，句子仍超长时固定长度硬切。"""
    # 按句子结束符切分，保留结束符本身
    pieces: List[str] = []
    current = ""
    for ch in paragraph:
        current += ch
        if ch in _SENTENCE_END:
            pieces.append(current)
            current = ""
    if current:
        pieces.append(current)

    # 贪心合并句子至 <= max_chars
    chunks: List[str] = []
    buf = ""
    for piece in pieces:
        if len(piece) > max_chars:
            # 先把已累积的落块
            if buf:
                chunks.append(buf)
                buf = ""
            # 单句仍超长：固定长度硬切
            for i in range(0, len(piece), max_chars):
                chunks.append(piece[i:i + max_chars])
        elif len(buf) + len(piece) <= max_chars:
            buf += piece
        else:
            chunks.append(buf)
            buf = piece
    if buf:
        chunks.append(buf)
    return [c for c in chunks if c.strip()]


def chunk_text(text: str, max_chars: int) -> List[str]:
    """主入口：段落合并 + 超长段落段内切分，返回每块 <= max_chars 的块列表。"""
    if max_chars < 1:
        raise ValueError("max_chars 必须 >= 1")
    paragraphs = split_paragraphs(text)
    if not paragraphs:
        return []

    chunks: List[str] = []
    buf: Optional[str] = None  # 当前累积的块
    for para in paragraphs:
        if len(para) > max_chars:
            # 超长段落：结束当前累积块，段内切分后逐块输出
            if buf is not None:
                chunks.append(buf)
                buf = None
            chunks.extend(_split_long_paragraph(para, max_chars))
            continue

        candidate = para if buf is None else buf + "\n\n" + para
        if len(candidate) <= max_chars:
            buf = candidate
        else:
            chunks.append(buf)
            buf = para
    if buf is not None:
        chunks.append(buf)

    return [c.strip() for c in chunks if c.strip()]
