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
                        "Return one JSON object matching the supplied JSON schema. Propose bounded RCA "
                        "mechanisms for the supplied hypotheses. "
                        "Keep each source_hypothesis_id. Reuse an existing mechanism when it explains the "
                        "facts; otherwise propose a new mechanism with a causal explanation, alternatives "
                        "and testable support/refutation predictions. Select only advertised rule_id values. "
                        "A refutation prediction describes a phenomenon that would refute the mechanism. "
                        "Never invent measurements, thresholds, weights, queries or a final decision. "
                        "Return an empty hypotheses list if the available rules cannot test a mechanism. "
                        "A hypothesis is valid only when it has at least two support predictions from "
                        "different causal dimensions and at least one contradict prediction. For every "
                        "listed alternative, copy its complete exact string into the distinguishes array "
                        "of at least one prediction whose outcome separates that alternative. Never leave "
                        "all distinguishes arrays empty. If these requirements cannot be met, omit that "
                        "hypothesis instead of returning an unverifiable draft. "
                        "Choose evidence from different causal dimensions, and predict additional "
                        "observations beyond those already supplied. All predictions are required tests. "
                        "You may return up to four distinct mechanisms for one source hypothesis; copy "
                        "the same source_hypothesis_id into each and order them by plausibility. "
                        "Completed generic verification can return the new mechanism as the root cause "
                        "for this investigation without Catalog promotion."
                    ),
                ),
                (
                    "human",
                    "JSON schema:\n{schema_json}\n\nInvestigation context:\n{context_json}",
                ),
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
        raw = self._runnable.invoke({
            "schema_json": json.dumps(DiscoveryPayload.model_json_schema(), sort_keys=True),
            "context_json": json.dumps(context, sort_keys=True),
        })
        return DiscoveryPayload.model_validate(raw)
