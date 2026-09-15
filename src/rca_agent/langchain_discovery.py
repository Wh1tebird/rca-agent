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
                        "A support prediction may use only a rule whose support_eligible field is true. "
                        "Generic JVM GC count and pause rules establish only generic GC pressure; "
                        "they cannot establish the specific Full GC subtype. "
                        "Rules marked false may be used for contradicting or distinguishing tests, but "
                        "they cannot prove a causal mechanism. "
                        "Polarity describes whether the advertised predicate should be true: support "
                        "means the rule predicate must match when the mechanism is true; contradict means "
                        "a matching predicate refutes the mechanism. For example, if high connection "
                        "duration is advertised but a mechanism predicts fast connection acquisition, "
                        "use contradict, never support. "
                        "Never invent measurements, thresholds, weights, queries or a final decision. "
                        "Return an empty hypotheses list if the available rules cannot test a mechanism. "
                        "Choose causal_entity_ref from entity_graph.nodes. A prediction may set "
                        "target_entity_ref only to an advertised entity on which its rule_id is available; "
                        "omit target_entity_ref when it is the causal entity. Never invent an entity. "
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
                include_raw=True,
            )
        )

    def discover(self, context: dict[str, Any]) -> DiscoveryPayload:
        response = self._runnable.invoke({
            "schema_json": json.dumps(DiscoveryPayload.model_json_schema(), sort_keys=True),
            "context_json": json.dumps(context, sort_keys=True),
        })
        if isinstance(response, dict) and "parsed" in response and "raw" in response:
            parsed = response.get("parsed")
            if parsed is not None:
                return DiscoveryPayload.model_validate(parsed)
            raw_message = response.get("raw")
            content = getattr(raw_message, "content", None)
            raw_payload: Any = None
            if isinstance(content, str):
                try:
                    raw_payload = json.loads(content)
                except json.JSONDecodeError:
                    raw_payload = {"unparsed_text": content[:20_000]}
            raise DiscoveryPayloadError(raw_payload)
        return DiscoveryPayload.model_validate(response)


class DiscoveryPayloadError(ValueError):
    """A provider returned content that failed the advertised Discovery schema."""

    def __init__(self, raw_payload: Any) -> None:
        super().__init__("invalid structured discovery payload")
        self.raw_payload = raw_payload
