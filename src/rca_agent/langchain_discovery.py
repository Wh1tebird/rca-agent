"""A separate, non-authoritative structured model interface for open discovery."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.prompts import ChatPromptTemplate

from rca_agent.core.dynamic import DiscoveryPayload


class LangChainDiscoveryAdapter:
    def __init__(self, runnable: Any) -> None:
        self._runnable = runnable

    @classmethod
    def from_chat_model(cls, model: Any, *, structured_method: str = "function_calling"):
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "Propose bounded RCA mechanisms for the supplied hypotheses. "
                        "Keep each source_hypothesis_id. Reuse an existing mechanism when it explains the "
                        "facts; otherwise propose a new mechanism with a causal explanation, alternatives "
                        "and testable support/refutation predictions. Select only advertised rule_id values. "
                        "A refutation prediction describes a phenomenon that would refute the mechanism. "
                        "Never invent measurements, thresholds, weights, queries or a final decision. "
                        "Return an empty hypotheses list if the available rules cannot test a mechanism. "
                    "Include at least one executable refutation test. For each listed alternative, "
                    "provide a distinguishing prediction with its exact name in distinguishes. "
                    "Choose evidence from different causal dimensions, and predict additional "
                    "observations beyond those already supplied. All predictions are required tests. "
                    "Completed generic verification can return the new mechanism as the root cause "
                    "for this investigation without Catalog promotion."
                    ),
                ),
                ("human", "{context_json}"),
            ]
        )
        return cls(
            prompt
            | model.with_structured_output(
                DiscoveryPayload,
                method=structured_method,
            )
        )

    def discover(self, context: dict[str, Any]) -> DiscoveryPayload:
        raw = self._runnable.invoke({"context_json": json.dumps(context, sort_keys=True)})
        return DiscoveryPayload.model_validate(raw)
