from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any, Literal

from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from pydantic import BaseModel, ConfigDict

from rca_agent.models import (
    CandidateFrontier,
    CoreAssessmentPreview,
    PolicyAction,
    ResetResponse,
    parse_policy_action,
)


class PolicyEnvelope(BaseModel):
    """The only structured output accepted from an investigation model."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    action: PolicyAction


class CandidateOption(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_id: str
    fault_type: str
    family: str
    required_signals: tuple[str, ...]
    required_capabilities: tuple[str, ...]


class PolicyContext(BaseModel):
    """Sanitized context supplied to LangChain; it contains no private adapter state."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    bootstrap: ResetResponse
    candidate_options: tuple[CandidateOption, ...]
    history: tuple[dict[str, Any], ...] = ()
    core_assessment: CoreAssessmentPreview
    candidate_frontier: CandidateFrontier = CandidateFrontier()


class LangChainPolicyAdapter:
    """Validate provider-neutral LangChain output as one closed PolicyAction."""

    def __init__(self, runnable: Runnable[dict[str, Any], Any]) -> None:
        self._runnable = runnable

    @classmethod
    def from_chat_model(
        cls,
        model: Any,
        *,
        structured_method: Literal["function_calling", "json_mode", "json_schema"] = "function_calling",
    ) -> LangChainPolicyAdapter:
        """Bind a future chat model to the closed action schema without invoking it now."""

        schema_json = json.dumps(
            PolicyEnvelope.model_json_schema(),
            ensure_ascii=False,
            sort_keys=True,
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "You are a non-authoritative RCA investigation policy. Return exactly one "
                        "schema-valid action. You may call only advertised tools, register a "
                        "hypothesis, attach issued observations to it, or request stop. Never output a final "
                        "decision, SQL, filesystem path, threshold, evidence strength, or evidence "
                        "weight. Return only JSON matching this schema: {policy_schema_json}"
                    ),
                ),
                ("human", "Investigation context:\n{context_json}"),
            ]
        ).partial(policy_schema_json=schema_json)
        structured = model.with_structured_output(
            PolicyEnvelope,
            method=structured_method,
        )
        return cls(prompt | structured)

    def next_action(self, context: PolicyContext) -> PolicyAction:
        payload = context.model_dump(mode="json")
        raw = self._runnable.invoke(
            {
                "context": payload,
                "context_json": json.dumps(payload, ensure_ascii=False, sort_keys=True),
            }
        )
        return _parse_envelope(raw).action


class ScriptedPolicyRunnable(Runnable[dict[str, Any], PolicyEnvelope]):
    """Deterministic LangChain Runnable for graph tests; it needs no model or API key."""

    def __init__(self, actions: Sequence[PolicyAction | dict[str, Any]]) -> None:
        self._actions = tuple(
            item if not isinstance(item, dict) else parse_policy_action(item) for item in actions
        )
        self._index = 0

    def invoke(
        self,
        input: dict[str, Any],
        config: Any = None,
        **kwargs: Any,
    ) -> PolicyEnvelope:
        del input, config, kwargs
        if self._index >= len(self._actions):
            raise RuntimeError("scripted policy exhausted before the graph reached a terminal state")
        action = self._actions[self._index]
        self._index += 1
        return PolicyEnvelope(action=action)


def _parse_envelope(raw: Any) -> PolicyEnvelope:
    if isinstance(raw, PolicyEnvelope):
        return raw
    if isinstance(raw, BaseMessage):
        content = raw.content
        if not isinstance(content, str):
            raise TypeError("structured policy message content must be JSON text")
        raw = json.loads(content)
    if isinstance(raw, dict) and "action" not in raw and "action_type" in raw:
        raw = {"action": raw}
    return PolicyEnvelope.model_validate(raw)
