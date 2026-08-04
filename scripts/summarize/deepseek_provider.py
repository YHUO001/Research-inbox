from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504}


class DeepSeekRequestError(RuntimeError):
    def __init__(self, reason: str, status: int | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.status = status


@dataclass(frozen=True)
class DeepSeekResponse:
    content: str
    usage: dict[str, int]
    model: str


class DeepSeekClient:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.deepseek.com",
        timeout_seconds: float = 60,
        max_attempts: int = 3,
        opener: Callable[..., Any] = urlopen,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if not api_key.strip():
            raise ValueError("DeepSeek API key must not be empty")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.max_attempts = max(1, max_attempts)
        self.opener = opener
        self.sleeper = sleeper

    def complete_json(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        thinking_enabled: bool = False,
    ) -> DeepSeekResponse:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {"type": "json_object"},
            "max_tokens": max_tokens,
            "stream": False,
            "thinking": {
                "type": "enabled" if thinking_enabled else "disabled"
            },
        }
        request = Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "ResearchInbox/0.4 (summary generation)",
            },
            method="POST",
        )

        last_error: DeepSeekRequestError | None = None
        for attempt in range(self.max_attempts):
            try:
                with self.opener(request, timeout=self.timeout_seconds) as response:
                    raw = response.read().decode("utf-8")
                value = json.loads(raw)
                if not isinstance(value, dict):
                    raise DeepSeekRequestError("invalid_response_shape")
                choices = value.get("choices") or []
                if not choices or not isinstance(choices[0], dict):
                    raise DeepSeekRequestError("missing_choice")
                message = choices[0].get("message") or {}
                content = message.get("content")
                if not isinstance(content, str) or not content.strip():
                    raise DeepSeekRequestError("empty_content")
                usage_raw = value.get("usage") or {}
                usage = {
                    key: int(usage_raw.get(key) or 0)
                    for key in (
                        "prompt_tokens",
                        "completion_tokens",
                        "total_tokens",
                        "prompt_cache_hit_tokens",
                        "prompt_cache_miss_tokens",
                    )
                }
                return DeepSeekResponse(
                    content=content,
                    usage=usage,
                    model=str(value.get("model") or model),
                )
            except HTTPError as error:
                last_error = DeepSeekRequestError(f"http_{error.code}", error.code)
                retryable = error.code in RETRYABLE_STATUS
            except (URLError, TimeoutError, OSError):
                last_error = DeepSeekRequestError("network_error")
                retryable = True
            except UnicodeDecodeError:
                last_error = DeepSeekRequestError("invalid_utf8")
                retryable = False
            except json.JSONDecodeError:
                last_error = DeepSeekRequestError("invalid_api_json")
                retryable = False
            except DeepSeekRequestError as error:
                last_error = error
                retryable = error.reason == "empty_content"

            if not retryable or attempt + 1 >= self.max_attempts:
                break
            self.sleeper(float(2**attempt))

        raise last_error or DeepSeekRequestError("unknown_error")
