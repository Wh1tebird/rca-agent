"""OpenAI-compatible binding for the separately hosted Qwen investigation model."""

from __future__ import annotations

import os
import sys
from getpass import getpass
from urllib.parse import urlsplit

from langchain_openai import ChatOpenAI
from pydantic import SecretStr

DEFAULT_LOCAL_MODEL = "Qwen3-30B-A3B-Thinking-2507"
DEFAULT_LOCAL_BASE_URL = "http://127.0.0.1:11025/v1"


def create_local_chat_model(
    *,
    model: str = DEFAULT_LOCAL_MODEL,
    base_url: str | None = None,
    timeout_seconds: float = 120,
    max_retries: int = 0,
    max_completion_tokens: int = 8192,
    allow_insecure_http: bool = False,
) -> ChatOpenAI:
    """Read the credential only from the host environment; never serialize it."""
    endpoint = base_url or os.environ.get("RCA_LOCAL_BASE_URL", DEFAULT_LOCAL_BASE_URL)
    parsed = urlsplit(endpoint)
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("base_url must not contain credentials, query or fragment")
    if parsed.path.rstrip("/").endswith("/chat/completions"):
        raise ValueError("base_url must end at /v1, not /chat/completions")
    if parsed.scheme != "https" and not (
        parsed.scheme == "http" and parsed.hostname in {"127.0.0.1", "localhost", "::1"}
    ) and not (allow_insecure_http and parsed.scheme == "http"):
        raise ValueError("API key requires HTTPS or a loopback-only tunnel")
    key = os.environ.get("RCA_LOCAL_API_KEY")
    if not key and sys.stdin.isatty():
        key = getpass("RCA local model API key (hidden input): ")
        if key:
            os.environ["RCA_LOCAL_API_KEY"] = key
    if not key:
        raise RuntimeError("RCA_LOCAL_API_KEY is not set")
    return ChatOpenAI(
        model=model,
        api_key=SecretStr(key),
        base_url=endpoint,
        temperature=0,
        max_completion_tokens=max_completion_tokens,
        timeout=timeout_seconds,
        max_retries=max_retries,
    )
