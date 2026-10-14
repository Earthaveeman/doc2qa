"""阿里云百炼智能体调用客户端：HTTP 请求、JSONL 返回校验、重试/超时/异常分类。"""
import json
import logging
import re
import time
from typing import Callable, List, Optional

import requests

logger = logging.getLogger("doc2qa")

# markdown 代码围栏容错：```json ... ``` 或 ``` ... ```
_FENCE_RE = re.compile(r"^```[a-zA-Z0-9_-]*\s*\n(.*?)\n?\s*```$", re.DOTALL)

RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class ResultFormatError(Exception):
    """智能体返回内容不是合法 JSONL（任一行非法或 0 行有效内容）。"""


class BailianCallError(Exception):
    """调用最终失败（重试耗尽或不可重试错误）。"""


def _strip_fence(text: str) -> str:
    """去除整体包裹的 markdown 代码围栏（LLM 常见行为，轻量容错）。"""
    stripped = text.strip()
    match = _FENCE_RE.match(stripped)
    if match:
        return match.group(1).strip()
    return stripped


def validate_result(text: str) -> List[dict]:
    """逐行校验智能体返回的 JSONL。

    每行必须是合法 JSON 且为含 prompt、response 键的对象；空行跳过；
    至少一行有效内容，否则视为失败。失败抛 ResultFormatError。
    """
    content = _strip_fence(text or "")
    results: List[dict] = []
    for line_no, line in enumerate(content.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            raise ResultFormatError(
                f"第 {line_no} 行不是合法 JSON: {e.msg}; 片段: {line[:200]!r}"
            ) from e
        if not isinstance(obj, dict):
            raise ResultFormatError(
                f"第 {line_no} 行不是 JSON 对象: {line[:200]!r}"
            )
        missing = [k for k in ("prompt", "response") if k not in obj]
        if missing:
            raise ResultFormatError(
                f"第 {line_no} 行缺少键 {missing}: {line[:200]!r}"
            )
        results.append(obj)
    if not results:
        raise ResultFormatError(f"返回内容无有效 JSONL 行; 片段: {content[:200]!r}")
    return results


class BailianClient:
    def __init__(self, config):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {config.api_key}",
                "Content-Type": "application/json",
            }
        )

    def call(
        self,
        prompt: str,
        on_retry: Optional[Callable[[int, int, str], None]] = None,
    ) -> List[dict]:
        """调用智能体并校验返回 JSONL。

        可重试错误：网络异常、HTTP 429/5xx、返回格式非法。
        不可重试错误：其他 HTTP 4xx。
        重试耗尽抛 BailianCallError。
        on_retry(第几次, 共几次, 原因) 用于控制台提示。
        """
        url = self.config.endpoint
        body = {"input": {"prompt": prompt}}
        max_retries = self.config.max_retries
        attempt = 0
        last_error: Exception

        while True:
            attempt += 1
            error, reason = self._request_once(url, body)
            if error is None:
                return reason  # type: ignore[return-value]

            last_error = error
            retryable = self._is_retryable(error)
            logger.debug(
                "调用失败 attempt=%d/%d retryable=%s error=%s",
                attempt, max_retries, retryable, error,
            )
            if not retryable or attempt > max_retries:
                raise BailianCallError(
                    f"调用失败（attempt={attempt}, retryable={retryable}）: {error}"
                ) from error

            if on_retry is not None:
                on_retry(attempt, max_retries, reason or "error")
            delay = self.config.retry_backoff ** attempt
            logger.debug("退避 %.1fs 后重试", delay)
            time.sleep(delay)

    def _request_once(self, url: str, body: dict) -> tuple:
        """单次请求。返回 (error, reason)；成功时 (None, List[dict])。"""
        try:
            resp = self.session.post(url, json=body, timeout=self.config.timeout)
        except requests.Timeout as e:
            return e, "timeout"
        except requests.RequestException as e:
            return e, "network_error"

        if resp.status_code != 200:
            if resp.status_code in RETRYABLE_STATUS:
                return BailianCallError(
                    f"HTTP {resp.status_code}: {resp.text[:300]}"
                ), f"http_{resp.status_code}"
            return BailianCallError(
                f"HTTP {resp.status_code}: {resp.text[:300]}"
            ), f"http_{resp.status_code}"

        try:
            data = resp.json()
        except ValueError as e:
            return BailianCallError(f"响应体不是 JSON: {resp.text[:300]!r}"), "bad_response"

        output = data.get("output") or {}
        text = output.get("text")
        if not isinstance(text, str):
            return BailianCallError(
                f"响应缺少 output.text 字段: {str(data)[:300]}"
            ), "bad_response"

        try:
            results = validate_result(text)
        except ResultFormatError as e:
            return e, "invalid_json"

        logger.debug(
            "调用成功 request_id=%s usage=%s 行数=%d",
            data.get("request_id"), data.get("usage"), len(results),
        )
        return None, results

    @staticmethod
    def _is_retryable(error: Exception) -> bool:
        if isinstance(error, ResultFormatError):
            return True
        if isinstance(error, requests.RequestException):
            return True
        if isinstance(error, BailianCallError):
            # HTTP 状态类错误：仅当消息以可重试状态码开头时可重试
            msg = str(error)
            return any(msg.startswith(f"HTTP {code}") for code in RETRYABLE_STATUS)
        return False
