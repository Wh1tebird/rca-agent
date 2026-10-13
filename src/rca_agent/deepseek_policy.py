from __future__ import annotations

from langchain_deepseek import ChatDeepSeek
from pydantic import SecretStr

from rca_agent.langchain_discovery import (
    LangChainDiscoveryAdapter,
    LangChainMechanismDiscoveryAdapter,
)
from rca_agent.langchain_policy import LangChainPolicyAdapter
from rca_agent.langchain_verification_planner import LangChainVerificationPlannerAdapter

DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-pro"
DEFAULT_DEEPSEEK_MAX_TOKENS = 65536
DEFAULT_DEEPSEEK_TIMEOUT_SECONDS = 300


def create_deepseek_chat_model(
    *,
    api_key: SecretStr | None = None,
    model: str = DEFAULT_DEEPSEEK_MODEL,
    timeout_seconds: float = DEFAULT_DEEPSEEK_TIMEOUT_SECONDS,
    max_tokens: int = DEFAULT_DEEPSEEK_MAX_TOKENS,
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
        max_tokens=max_tokens,
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
    timeout_seconds: float = DEFAULT_DEEPSEEK_TIMEOUT_SECONDS,
    max_tokens: int = DEFAULT_DEEPSEEK_MAX_TOKENS,
    thinking_enabled: bool = True,
) -> LangChainDiscoveryAdapter:
    """Construct (without calling) the separately budgeted open-discovery model."""
    return LangChainDiscoveryAdapter.from_chat_model(
        create_deepseek_chat_model(
            api_key=api_key,
            model=model,
            timeout_seconds=timeout_seconds,
            max_tokens=max_tokens,
            max_retries=0,
            thinking_enabled=thinking_enabled,
        ),
        structured_method="json_mode",
    )


def create_deepseek_discovery_pipeline(
    *,
    api_key: SecretStr | None = None,
    model: str = DEFAULT_DEEPSEEK_MODEL,
    timeout_seconds: float = DEFAULT_DEEPSEEK_TIMEOUT_SECONDS,
    max_tokens: int = DEFAULT_DEEPSEEK_MAX_TOKENS,
    thinking_enabled: bool = True,
) -> tuple[LangChainMechanismDiscoveryAdapter, LangChainVerificationPlannerAdapter]:
    """Bind two prompts to one stateless provider model configuration."""
    chat_model = create_deepseek_chat_model(
        api_key=api_key,
        model=model,
        timeout_seconds=timeout_seconds,
        max_tokens=max_tokens,
        max_retries=0,
        thinking_enabled=thinking_enabled,
    )
    return (
        LangChainMechanismDiscoveryAdapter.from_chat_model(
            chat_model, structured_method="json_mode"
        ),
        LangChainVerificationPlannerAdapter.from_chat_model(
            chat_model, structured_method="json_mode"
        ),
    )
