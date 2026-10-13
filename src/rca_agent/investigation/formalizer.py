"""Semantics-preserving boundary from investigation tracks to typed mechanisms."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rca_agent.core.dynamic import MechanismDiscoveryPayload, MechanismDraft
from rca_agent.investigation.models import InvestigationAlternative


def _normalize(value: str) -> str:
    return " ".join(value.casefold().split())


class FormalizationSemanticError(ValueError):
    """Reject semantic drift while exposing only a stable diagnostic code."""

    def __init__(self, reason_code: str) -> None:
        super().__init__("formalizer changed or detached candidate causal semantics")
        self.reason_code = reason_code


class FormalizationCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_track_id: str = Field(pattern=r"^track-[a-f0-9]{20}$")
    semantic_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    mechanism_summary: str = Field(min_length=1, max_length=2000)
    mechanism_class: str | None = Field(default=None, max_length=80)
    causal_entity_ref: str | None = Field(default=None, pattern=r"^ent-[a-z0-9-]{3,40}$")
    causal_explanation: str | None = Field(default=None, max_length=4000)
    alternatives: tuple[InvestigationAlternative, ...] = Field(min_length=1, max_length=12)
    source_hypothesis_ids: tuple[str, ...] = Field(default=(), max_length=40)


class MechanismFormalizationContext(BaseModel):
    """GT-free input. It deliberately excludes Registry rules and Core thresholds."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    alert_summary: str | dict[str, Any]
    candidates: tuple[FormalizationCandidate, ...] = Field(min_length=1, max_length=20)
    entity_graph: dict[str, Any]
    observations: tuple[dict[str, Any], ...] = Field(default=(), max_length=1000)
    proposal_ids: tuple[str, ...] = Field(min_length=1, max_length=100)


class FormalizedMechanism(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_track_id: str = Field(pattern=r"^track-[a-f0-9]{20}$")
    source_semantic_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    mechanism: MechanismDraft


class FormalizationSkipReason(StrEnum):
    INSUFFICIENT_CAUSAL_DETAIL = "insufficient_causal_detail"
    AMBIGUOUS_ENTITY = "ambiguous_entity"
    AMBIGUOUS_MECHANISM = "ambiguous_mechanism"
    NO_PLAUSIBLE_ALTERNATIVE = "no_plausible_alternative"
    OTHER_SEMANTIC_BLOCKER = "other_semantic_blocker"


class SkippedFormalization(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_track_id: str = Field(pattern=r"^track-[a-f0-9]{20}$")
    reason_code: FormalizationSkipReason
    reason_summary: str = Field(min_length=5, max_length=300)


class MechanismFormalizationPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    mechanisms: tuple[FormalizedMechanism, ...] = Field(default=(), max_length=4)
    skipped_candidates: tuple[SkippedFormalization, ...] = Field(default=(), max_length=20)

    @model_validator(mode="after")
    def unique_tracks(self):
        tracks = [item.candidate_track_id for item in self.mechanisms]
        tracks.extend(item.candidate_track_id for item in self.skipped_candidates)
        if len(tracks) != len(set(tracks)):
            raise ValueError("formalizer returned a candidate track more than once")
        return self


def validate_formalization_accounting(
    payload: MechanismFormalizationPayload,
    context: MechanismFormalizationContext,
) -> None:
    offered = {item.candidate_track_id for item in context.candidates}
    accounted = {item.candidate_track_id for item in payload.mechanisms}
    accounted.update(item.candidate_track_id for item in payload.skipped_candidates)
    if accounted != offered:
        raise FormalizationSemanticError("INCOMPLETE_CANDIDATE_ACCOUNTING")


class MechanismFormalizer(Protocol):
    def formalize(
        self, context: MechanismFormalizationContext,
    ) -> MechanismFormalizationPayload | dict[str, Any]: ...


class DiscoveryMechanismFormalizerAdapter:
    """Compatibility wrapper for a legacy Discovery provider.

    The adapter may fill typed fields, but drops every draft whose mechanism text
    changes the bound InvestigationCandidate's causal semantics.
    """

    def __init__(self, discovery: Any) -> None:
        self.discovery = discovery

    def formalize(self, context: MechanismFormalizationContext) -> MechanismFormalizationPayload:
        raw = self.discovery.discover({
            "alert_summary": context.alert_summary,
            "entity_graph": context.entity_graph,
            "observations": list(context.observations),
            "investigation_candidates": [
                item.model_dump(mode="json") for item in context.candidates
            ],
            "constraints": [
                "formalize only an offered candidate_track_id",
                "preserve mechanism causal semantics exactly",
                "do not infer a new root-cause direction",
            ],
        })
        payload = MechanismDiscoveryPayload.model_validate(raw)
        by_mechanism = {
            _normalize(item.mechanism_summary): item for item in context.candidates
        }
        accepted = []
        for draft in payload.mechanisms:
            candidate = by_mechanism.get(_normalize(draft.mechanism))
            if candidate is None or draft.source_hypothesis_id not in context.proposal_ids:
                continue
            accepted.append(FormalizedMechanism(
                candidate_track_id=candidate.candidate_track_id,
                source_semantic_digest=candidate.semantic_digest,
                mechanism=draft,
            ))
        return MechanismFormalizationPayload(mechanisms=tuple(accepted))


def validate_formalization_semantics(
    payload: MechanismFormalizationPayload,
    context: MechanismFormalizationContext,
) -> MechanismFormalizationPayload:
    candidates = {item.candidate_track_id: item for item in context.candidates}
    normalized = []
    for item in payload.mechanisms:
        candidate = candidates.get(item.candidate_track_id)
        if candidate is None:
            raise FormalizationSemanticError("UNKNOWN_CANDIDATE_TRACK")
        if item.source_semantic_digest != candidate.semantic_digest:
            raise FormalizationSemanticError("SEMANTIC_DIGEST_MISMATCH")
        if _normalize(item.mechanism.mechanism) != _normalize(candidate.mechanism_summary):
            raise FormalizationSemanticError("MECHANISM_TEXT_CHANGED")
        if item.mechanism.source_hypothesis_id not in context.proposal_ids:
            if len(context.proposal_ids) != 1:
                raise FormalizationSemanticError("UNISSUED_SOURCE_HYPOTHESIS")
            # The candidate track/digest/text have already bound the causal meaning.
            # With one submitted proposal, its routing ID is host-owned and unambiguous.
            mechanism = MechanismDraft.model_validate({
                **item.mechanism.model_dump(mode="json"),
                "source_hypothesis_id": context.proposal_ids[0],
            })
            item = item.model_copy(update={"mechanism": mechanism})
        normalized.append(item)
    return payload.model_copy(update={"mechanisms": tuple(normalized)})
