"""Structured evidence-plan selection for one or more frozen mechanisms."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.prompts import ChatPromptTemplate

from rca_agent.core.dynamic import VerificationPlanPayload
from rca_agent.langchain_discovery import DiscoveryPayloadError, _raw_message_payload


class LangChainVerificationPlannerAdapter:
    """Select host-advertised capability handles without changing mechanism semantics."""

    def __init__(self, runnable: Any) -> None:
        self._runnable = runnable

    @classmethod
    def from_chat_model(cls, model: Any, *, structured_method: str = "function_calling"):
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                (
                    "Build a verification plan for every supplied frozen mechanism. Return only "
                    "the advertised schema. Copy each mechanism_id exactly. Select only advertised "
                    "capability_ref values; each handle already binds a legal Entity, Signal, Rule "
                    "and query recipe. Use support only when support_eligible is true. Every plan "
                    "needs at least two support predictions from different independent_group values, "
                    "at least one contradict prediction, and must reference every declared alt-* ID. "
                    "Each contradict prediction must reference exactly one alternative, and each "
                    "alternative needs its own causally relevant test whose capability refutes list "
                    "contains that alternative's mechanism_class. Support predictions cannot "
                    "exclude an alternative. Predict future verification observations rather than treating "
                    "the supplied discovery observations as new proof. On a retry, change only the "
                    "plan in response to the supplied safe error codes. You cannot modify the frozen "
                    "mechanism, entity, family, explanation or alternatives. Return no plan when the "
                    "advertised capabilities cannot verify the mechanism. On SUPPORT_CANNOT_REFUTE, "
                    "remove distinguishes only from support predictions and keep every alternative "
                    "on a matching contradict prediction. On MISSING_REFUTATION, restore exactly one "
                    "alternative ID on each matching contradict prediction. On "
                    "REFUTATION_SCOPE_MISMATCH, choose a contradict capability whose refutes list "
                    "contains that alternative class. If none exists, leave that alternative "
                    "unreferenced; the host will retain the mechanism as unresolved rather than "
                    "treating an unrelated signal as proof."
                ),
            ),
            (
                "human",
                "JSON schema:\n{schema_json}\n\nPlanning context:\n{context_json}",
            ),
        ])
        return cls(prompt | model.with_structured_output(
            VerificationPlanPayload,
            method=structured_method,
            include_raw=True,
        ))

    def plan(self, context: dict[str, Any]) -> VerificationPlanPayload:
        response = self._runnable.invoke({
            "schema_json": json.dumps(VerificationPlanPayload.model_json_schema(), sort_keys=True),
            "context_json": json.dumps(context, sort_keys=True),
        })
        if isinstance(response, dict) and "parsed" in response and "raw" in response:
            parsed = response.get("parsed")
            if parsed is not None:
                return VerificationPlanPayload.model_validate(parsed)
            raise DiscoveryPayloadError(_raw_message_payload(response.get("raw")))
        return VerificationPlanPayload.model_validate(response)
