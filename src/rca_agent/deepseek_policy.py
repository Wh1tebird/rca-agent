from __future__ import annotations

from langchain_deepseek import ChatDeepSeek
from pydantic import SecretStr

from rca_agent.langchain_discovery import LangChainDiscoveryAdapter
from rca_agent.langchain_policy import LangChainPolicyAdapter

DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-pro"


def create_deepseek_chat_model(
    *,
    api_key: SecretStr | None = None,
    model: str = DEFAULT_DEEPSEEK_MODEL,
    timeout_seconds: float = 120,
    max_retries: int = 2,
    thinking_enabled: bool = True,
) -> ChatDeepSeek:
    """Create the accuracy-first DeepSeek model used by the RCA investigation policy.

    When ``api_key`` is omitted, ChatDeepSeek reads ``DEEPSEEK_API_KEY`` from the process
    environment. The caller owns secret loading; this module never reads or persists plaintext.
    """

    credentials = {"api_key": api_key} if api_key is not None else {}
    reasoning = (
        {"reasoning_effort": "high", "extra_body": {"thinking": {"type": "enabled"}}}
        if thinking_enabled
        else {}
    )
    return ChatDeepSeek(
        model=model,
        temperature=0,
        max_tokens=8192,
        timeout=timeout_seconds,
        max_retries=max_retries,
        **reasoning,
        **credentials,
    )


def create_deepseek_policy(
    *,
    api_key: SecretStr | None = None,
    model: str = DEFAULT_DEEPSEEK_MODEL,
) -> LangChainPolicyAdapter:
    """Bind DeepSeek to the existing non-authoritative structured Policy boundary."""

    return LangChainPolicyAdapter.from_chat_model(
        create_deepseek_chat_model(api_key=api_key, model=model),
        structured_method="json_mode",
    )


def create_deepseek_discovery(
    *,
    api_key: SecretStr | None = None,
    model: str = DEFAULT_DEEPSEEK_MODEL,
    timeout_seconds: float = 30,
    thinking_enabled: bool = True,
) -> LangChainDiscoveryAdapter:
    """Construct (without calling) the separately budgeted open-discovery model."""
    return LangChainDiscoveryAdapter.from_chat_model(
        create_deepseek_chat_model(
            api_key=api_key,
            model=model,
            timeout_seconds=timeout_seconds,
            max_retries=0,
            thinking_enabled=thinking_enabled,
        ),
        structured_method="json_mode",
    )
