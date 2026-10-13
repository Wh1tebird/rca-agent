"""Qwen binding for the existing non-authoritative RCA model boundaries."""

from __future__ import annotations

from rca_agent.langchain_discovery import (
    LangChainDiscoveryAdapter,
    LangChainMechanismDiscoveryAdapter,
)
from rca_agent.langchain_policy import LangChainPolicyAdapter
from rca_agent.langchain_verification_planner import LangChainVerificationPlannerAdapter
from rca_agent.local_model import DEFAULT_LOCAL_MODEL, create_local_chat_model


def create_qwen_policy(
    *, model: str = DEFAULT_LOCAL_MODEL, base_url: str | None = None,
    timeout_seconds: float = 120,
    allow_insecure_http: bool = False,
) -> LangChainPolicyAdapter:
    return LangChainPolicyAdapter.from_chat_model(
        create_local_chat_model(model=model, base_url=base_url,
                                timeout_seconds=timeout_seconds,
                                allow_insecure_http=allow_insecure_http),
        structured_method="json_schema",
    )


def create_qwen_discovery(
    *, model: str = DEFAULT_LOCAL_MODEL, base_url: str | None = None,
    timeout_seconds: float = 120,
    allow_insecure_http: bool = False,
) -> LangChainDiscoveryAdapter:
    return LangChainDiscoveryAdapter.from_chat_model(
        create_local_chat_model(model=model, base_url=base_url,
                                timeout_seconds=timeout_seconds,
                                allow_insecure_http=allow_insecure_http),
        structured_method="json_schema",
    )


def create_qwen_discovery_pipeline(
    *, model: str = DEFAULT_LOCAL_MODEL, base_url: str | None = None,
    timeout_seconds: float = 120,
    allow_insecure_http: bool = False,
) -> tuple[LangChainMechanismDiscoveryAdapter, LangChainVerificationPlannerAdapter]:
    chat_model = create_local_chat_model(model=model, base_url=base_url,
                                         timeout_seconds=timeout_seconds,
                                         max_completion_tokens=16384,
                                         allow_insecure_http=allow_insecure_http)
    return (
        LangChainMechanismDiscoveryAdapter.from_chat_model(
            chat_model, structured_method="json_schema",
        ),
        LangChainVerificationPlannerAdapter.from_chat_model(
            chat_model, structured_method="json_schema",
        ),
    )
