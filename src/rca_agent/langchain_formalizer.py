"""Structured Qwen boundary for semantics-preserving investigation formalization."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from pydantic import ValidationError

from rca_agent.core.dynamic import MechanismDiscoveryPayload
from rca_agent.investigation.discovery_format import DiscoveryFormatRepairContext
from rca_agent.investigation.formalizer import (
    MechanismFormalizationContext,
    MechanismFormalizationPayload,
)


class FormalizerBoundaryError(ValueError):
    """Safe, non-content diagnostic for a rejected structured model response."""

    def __init__(self, reason_code: str, error: Exception | None = None) -> None:
        super().__init__("invalid structured formalizer response")
        self.reason_code = reason_code
        self.field_paths = tuple(
            ".".join(str(part) for part in item["loc"])
            for item in error.errors()[:8]
        ) if isinstance(error, ValidationError) else ()


class LangChainMechanismFormalizerAdapter:
    supports_candidate_accounting = True

    def __init__(self, runnable: Any) -> None:
        self._runnable = runnable

    @classmethod
    def from_chat_model(
        cls, model: Any, *, structured_method: str = "function_calling",
    ) -> LangChainMechanismFormalizerAdapter:
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                (
                    "Return only the advertised JSON schema. Formalize at most four of the "
                    "supplied open investigation candidates. For each output, copy its exact "
                    "candidate_track_id, semantic_digest into source_semantic_digest, and "
                    "mechanism_summary into mechanism.mechanism. Do not rename, broaden, or "
                    "replace the causal mechanism. Use an existing proposal_id as "
                    "source_hypothesis_id and an issued entity_graph node as causal_entity_ref. "
                    "Explain the causal chain and include at least one plausible alternative "
                    "with a unique alt-* ID and closest mechanism_class. Classify the "
                    "mechanism and family. Account for EVERY offered candidate exactly once: "
                    "either formalize it or put its track ID in skipped_candidates with a "
                    "specific reason_code and brief reason_summary. Missing verification "
                    "evidence is not a reason to skip formalization; Core checks evidence later. "
                    "An empty mechanisms list is allowed only with a skip reason for each "
                    "candidate. A retry may include the previous response; retain every "
                    "previously formalized mechanism unchanged and reconsider skipped candidates "
                    "without inventing a new cause. Do not choose Signals, Rules, "
                    "queries, thresholds, evidence weights or a final decision."
                ),
            ),
            (
                "human",
                (
                    "JSON schema:\n{schema_json}\n\nInvestigation context:\n{context_json}"
                    "\n\nPrevious response and retry instruction (null on first call):"
                    "\n{feedback_json}"
                ),
            ),
        ])
        return cls(prompt | model.with_structured_output(
            MechanismFormalizationPayload,
            method=structured_method,
            include_raw=True,
        ))

    def formalize(
        self, context: MechanismFormalizationContext,
    ) -> MechanismFormalizationPayload:
        return self._invoke(context, feedback=None)

    def retry_formalize(
        self,
        context: MechanismFormalizationContext,
        previous: MechanismFormalizationPayload,
    ) -> MechanismFormalizationPayload:
        return self._invoke(context, feedback={
            "instruction": (
                "One final semantics-preserving review. Formalization is not verification. "
                "Return a mechanism for any candidate that can be typed without changing its "
                "cause; otherwise provide a specific skip reason for that candidate."
            ),
            "previous_response": previous.model_dump(mode="json"),
        })

    def _invoke(
        self,
        context: MechanismFormalizationContext,
        *,
        feedback: dict[str, Any] | None,
    ) -> MechanismFormalizationPayload:
        response = self._runnable.invoke({
            "schema_json": json.dumps(
                MechanismFormalizationPayload.model_json_schema(), sort_keys=True
            ),
            "context_json": context.model_dump_json(),
            "feedback_json": json.dumps(feedback, sort_keys=True),
        })
        if isinstance(response, dict) and "parsed" in response and "raw" in response:
            parsed = response.get("parsed")
            if parsed is not None:
                try:
                    return MechanismFormalizationPayload.model_validate(parsed)
                except ValidationError as exc:
                    raise FormalizerBoundaryError("SCHEMA_VALIDATION_FAILED", exc) from exc
            parsing_error = response.get("parsing_error")
            raise FormalizerBoundaryError(
                "STRUCTURED_PARSE_FAILED" if parsing_error is not None
                else "EMPTY_PARSED_RESPONSE",
                parsing_error if isinstance(parsing_error, Exception) else None,
            )
        try:
            return MechanismFormalizationPayload.model_validate(response)
        except ValidationError as exc:
            raise FormalizerBoundaryError("SCHEMA_VALIDATION_FAILED", exc) from exc


class LangChainDiscoveryFormatFormalizerAdapter:
    """Optional one-call format repair after semantic Discovery, not a new proposer."""

    def __init__(self, runnable: Any) -> None:
        self._runnable = runnable

    @classmethod
    def from_chat_model(
        cls, model: Any, *, structured_method: str = "function_calling",
    ) -> LangChainDiscoveryFormatFormalizerAdapter:
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                (
                    "Return only the advertised schema. Repair malformed structure in the "
                    "Discovery response. Copy EVERY anchored mechanism in its original order, "
                    "with its exact source_hypothesis_id, causal_entity_ref and mechanism text. "
                    "Do not add, remove, rename, generalize or replace a causal mechanism. "
                    "Preserve already valid mechanism_class, family and alternative mechanism "
                    "texts. Fill only missing or malformed typed fields when the raw response "
                    "supports them. If safe repair is impossible, return an empty mechanisms "
                    "list; the host will reject it. Do not choose queries, Signals, Rules, "
                    "thresholds, evidence weights or a final decision."
                ),
            ),
            (
                "human",
                (
                    "JSON schema:\n{schema_json}\n\nAnchored malformed Discovery response:"
                    "\n{context_json}"
                ),
            ),
        ])
        return cls(prompt | model.with_structured_output(
            MechanismDiscoveryPayload,
            method=structured_method,
            include_raw=True,
        ))

    def repair(self, context: DiscoveryFormatRepairContext) -> MechanismDiscoveryPayload:
        response = self._runnable.invoke({
            "schema_json": json.dumps(
                MechanismDiscoveryPayload.model_json_schema(), sort_keys=True
            ),
            "context_json": context.model_dump_json(),
        })
        if isinstance(response, dict) and "parsed" in response and "raw" in response:
            parsed = response.get("parsed")
            if parsed is None:
                parsing_error = response.get("parsing_error")
                raise FormalizerBoundaryError(
                    "STRUCTURED_PARSE_FAILED" if parsing_error is not None
                    else "EMPTY_PARSED_RESPONSE",
                    parsing_error if isinstance(parsing_error, Exception) else None,
                )
            response = parsed
        try:
            return MechanismDiscoveryPayload.model_validate(response)
        except ValidationError as exc:
            raise FormalizerBoundaryError("SCHEMA_VALIDATION_FAILED", exc) from exc
