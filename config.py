"""配置加载：从 .env 与环境变量读取所有配置，无硬编码。"""
import os
import sys
from dataclasses import dataclass, field
from typing import List

try:
    from dotenv import load_dotenv
except ImportError:
    print("缺少依赖 python-dotenv，请先执行: pip install -r requirements.txt", file=sys.stderr)
    raise

# .env 与代码同目录（.env.example 复制为 .env 使用）
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

DEFAULT_ENDPOINT = "https://dashscope.aliyuncs.com/api/v1/apps/{app_id}/completion"


@dataclass
class Config:
    app_id: str
    api_key: str
    endpoint_template: str
    timeout: int
    max_retries: int
    retry_backoff: float
    chunk_max_chars: int
    input_dir: str
    text_extensions: List[str] = field(default_factory=list)

    @property
    def endpoint(self) -> str:
        return self.endpoint_template.replace("{app_id}", self.app_id)


class ConfigError(Exception):
    pass


def _get_int(name: str, default: int, minimum: int = 1) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ConfigError(f"配置项 {name} 不是整数: {raw!r}")
    if value < minimum:
        raise ConfigError(f"配置项 {name} 必须 >= {minimum}, 当前: {value}")
    return value


def _get_float(name: str, default: float, minimum: float = 0) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        raise ConfigError(f"配置项 {name} 不是数字: {raw!r}")
    if value < minimum:
        raise ConfigError(f"配置项 {name} 必须 >= {minimum}, 当前: {value}")
    return value


def load_config() -> Config:
    """加载并校验配置。必填项缺失时抛 ConfigError。"""
    app_id = os.getenv("BAILIAN_APP_ID", "").strip()
    api_key = os.getenv("BAILIAN_API_KEY", "").strip()
    if not app_id or not api_key:
        missing = [n for n, v in (("BAILIAN_APP_ID", app_id), ("BAILIAN_API_KEY", api_key)) if not v]
        raise ConfigError(
            "缺少必填配置: " + ", ".join(missing)
            + "。请复制 .env.example 为 .env 并填写真实凭证。"
        )

    endpoint_template = os.getenv("BAILIAN_ENDPOINT", "").strip() or DEFAULT_ENDPOINT
    if "{app_id}" not in endpoint_template:
        raise ConfigError("BAILIAN_ENDPOINT 中必须包含 {app_id} 占位符")

    ext_raw = os.getenv("TEXT_EXTENSIONS", ".txt,.md").strip()
    extensions = [e.strip().lower() for e in ext_raw.split(",") if e.strip()]
    if not extensions:
        raise ConfigError("TEXT_EXTENSIONS 不能为空")
    # 规范化为以 . 开头
    extensions = [e if e.startswith(".") else "." + e for e in extensions]

    return Config(
        app_id=app_id,
        api_key=api_key,
        endpoint_template=endpoint_template,
        timeout=_get_int("BAILIAN_TIMEOUT", 60),
        max_retries=_get_int("BAILIAN_MAX_RETRIES", 3),
        retry_backoff=_get_float("BAILIAN_RETRY_BACKOFF", 2),
        chunk_max_chars=_get_int("CHUNK_MAX_CHARS", 2000),
        input_dir=os.getenv("INPUT_DIR", "./input").strip(),
        text_extensions=extensions,
    )
