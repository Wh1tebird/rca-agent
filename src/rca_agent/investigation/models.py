"""Typed contracts for the non-authoritative investigation workspace."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rca_agent.tools.fingerprint import sha256_digest

WORKSPACE_SCHEMA_VERSION = "investigation-workspace-v1"
WORKSPACE_REDUCER_VERSION = "investigation-reducer-v1"
WORKSPACE_REDUCER_VERSION_V2 = "investigation-reducer-v2"


def _normalized_text(value: str) -> str:
    return " ".join(value.casefold().split())


def candidate_semantic_digest(
    mechanism_summary: str,
    *,
    mechanism_class: str | None = None,
    causal_entity_ref: str | None = None,
) -> str:
    """Hash causal identity fields; presentation-only clarification must preserve this value."""

    return sha256_digest({
        "mechanism": _normalized_text(mechanism_summary),
        "mechanism_class": mechanism_class,
        "causal_entity_ref": causal_entity_ref,
    })


def alternative_semantic_digest(candidate_track_id: str, mechanism_summary: str) -> str:
    return sha256_digest({
        "candidate_track_id": candidate_track_id,
        "mechanism": _normalized_text(mechanism_summary),
    })


def investigation_alternative_id(candidate_track_id: str, semantic_digest: str) -> str:
    return "ialt-" + sha256_digest({
        "candidate_track_id": candidate_track_id,
        "semantic_digest": semantic_digest,
    })[:20]


def candidate_track_id(episode_id: str, semantic_digest: str) -> str:
    return "track-" + sha256_digest({
        "episode_id": episode_id,
        "semantic_digest": semantic_digest,
    })[:20]


class CandidateStage(StrEnum):
    TENTATIVE = "tentative"
    ACTIVE = "active"
    DEFERRED = "deferred"
    CLOSED = "closed"


class CandidateDisposition(StrEnum):
    OPEN = "open"
    DISFAVORED = "disfavored"


class CandidateSourceKind(StrEnum):
    INQUIRY = "inquiry"
    DISCOVERY_REVISION = "discovery_revision"
    COVERAGE_POLICY = "coverage_policy"
    DIAGNOSTIC_PLAYBOOK = "diagnostic_playbook"
    POLICY = "policy"


class CandidateTransitionType(StrEnum):
    ADD = "add"
    RETAIN = "retain"
    CLARIFY = "clarify"
    SUPERSEDE = "supersede"
    DEFER = "defer"
    REACTIVATE = "reactivate"
    FORMALIZE = "formalize"
    DISFAVOR = "disfavor"
    RECONSIDER = "reconsider"
    CLOSE = "close"


def candidate_transition_id(
    episode_id: str,
    sequence: int,
    candidate_track_id: str,
    transition_type: CandidateTransitionType,
) -> str:
    return "transition-" + sha256_digest({
        "episode_id": episode_id,
        "sequence": sequence,
        "candidate_track_id": candidate_track_id,
        "transition_type": transition_type,
    })[:20]


class CandidateCloseReason(StrEnum):
    SUPERSEDED = "superseded"
    DUPLICATE_MERGED = "duplicate_merged"
    DISFAVORED_BY_EVIDENCE = "disfavored_by_evidence"
    INVESTIGATION_SCOPE_MISMATCH = "investigation_scope_mismatch"


class CandidateVerificationReachability(StrEnum):
    NOT_ASSESSED = "not_assessed"
    ACTIONABLE = "actionable"
    PARTIALLY_ACTIONABLE = "partially_actionable"
    UNVERIFIABLE = "unverifiable"
    DISCRIMINATION_COMPLETE = "discrimination_complete"


class DiscriminationTaskStatus(StrEnum):
    OPEN = "open"
    SELECTED = "selected"
    RESOLVED = "resolved"
    AMBIGUOUS = "ambiguous"
    UNVERIFIABLE = "unverifiable"
    CANCELLED = "cancelled"


class DiscriminationImportance(StrEnum):
    PRIMARY = "primary"
    SUPPLEMENTAL = "supplemental"


class AttemptOutcome(StrEnum):
    DISTINGUISHES_CANDIDATE = "distinguishes_candidate"
    DISTINGUISHES_COMPARATOR = "distinguishes_comparator"
    AMBIGUOUS = "ambiguous"
    NO_DATA = "no_data"
    QUERY_FAILED = "query_failed"


def discrimination_task_id(
    episode_id: str,
    candidate_track_id: str,
    alternative_id: str,
    question: str,
    candidate_prediction: str,
    comparator_prediction: str,
    recurrence_round: int | None = None,
    query_fingerprint: str | None = None,
    binding_id: str | None = None,
) -> str:
    fields = {
        "episode_id": episode_id,
        "candidate_track_id": candidate_track_id,
        "alternative_id": alternative_id,
        "question": _normalized_text(question),
        "candidate_prediction": _normalized_text(candidate_prediction),
        "comparator_prediction": _normalized_text(comparator_prediction),
    }
    if recurrence_round is not None:
        fields["recurrence_round"] = recurrence_round
    if query_fingerprint is not None:
        fields["query_strategy_v2_fingerprint"] = query_fingerprint
    if binding_id is not None:
        fields["binding_id"] = binding_id
    return "task-" + sha256_digest(fields)[:20]


def discrimination_attempt_id(
    episode_id: str,
    task_id: str,
    capability_ref: str,
    query_fingerprint: str,
) -> str:
    return "attempt-" + sha256_digest({
        "episode_id": episode_id,
        "task_id": task_id,
        "capability_ref": capability_ref,
        "query_fingerprint": query_fingerprint,
    })[:20]


class InvestigationAlternative(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    source_alternative_id: str | None = Field(default=None, exclude_if=lambda v: v is None)

    alternative_id: str = Field(pattern=r"^ialt-[a-f0-9]{20}$")
    mechanism_summary: str = Field(min_length=1, max_length=2000)
    mechanism_class: str | None = Field(default=None, max_length=80)
    related_candidate_track_id: str | None = Field(
        default=None, pattern=r"^track-[a-f0-9]{20}$"
    )
    source_event_refs: tuple[str, ...] = Field(default=(), max_length=20)
    semantic_digest: str = Field(pattern=r"^[a-f0-9]{64}$")


class InvestigationCandidate(BaseModel):
    """A revisable investigation track with no Core decision authority."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_track_id: str = Field(pattern=r"^track-[a-f0-9]{20}$")
    mechanism_summary: str = Field(min_length=1, max_length=2000)
    mechanism_class: str | None = Field(default=None, max_length=80)
    causal_entity_ref: str | None = Field(
        default=None, pattern=r"^ent-[a-z0-9-]{3,40}$"
    )
    causal_explanation: str | None = Field(default=None, max_length=4000)
    system_layer: str | None = Field(default=None, max_length=100)
    stage: CandidateStage
    disposition: CandidateDisposition
    alternatives: tuple[InvestigationAlternative, ...] = Field(default=(), max_length=12)
    source_kind: CandidateSourceKind
    source_event_refs: tuple[str, ...] = Field(default=(), max_length=40)
    source_hypothesis_ids: tuple[str, ...] = Field(default=(), max_length=40)
    first_seen_round: int | None = Field(default=None, ge=1, le=100)
    last_updated_round: int | None = Field(default=None, ge=1, le=100)
    semantic_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    supersedes_candidate_track_id: str | None = Field(
        default=None, pattern=r"^track-[a-f0-9]{20}$"
    )
    superseded_by_candidate_track_id: str | None = Field(
        default=None, pattern=r"^track-[a-f0-9]{20}$"
    )
    formal_mechanism_id: str | None = Field(
        default=None, pattern=r"^mech-[a-f0-9]{20}$"
    )
    compiled_candidate_ids: tuple[str, ...] = Field(default=(), max_length=20)
    created_sequence: int = Field(ge=1)
    last_transition_sequence: int = Field(ge=1)

    @model_validator(mode="after")
    def validate_identity(self):
        if self.candidate_track_id in {
            self.supersedes_candidate_track_id,
            self.superseded_by_candidate_track_id,
        }:
            raise ValueError("candidate cannot supersede itself")
        if len(self.source_hypothesis_ids) != len(set(self.source_hypothesis_ids)):
            raise ValueError("duplicate source hypothesis IDs")
        if len(self.compiled_candidate_ids) != len(set(self.compiled_candidate_ids)):
            raise ValueError("duplicate compiled candidate IDs")
        if self.compiled_candidate_ids and self.formal_mechanism_id is None:
            raise ValueError("compiled candidates require a formal mechanism")
        return self


class TransitionBasis(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    observation_refs: tuple[str, ...] = Field(default=(), max_length=100)
    query_refs: tuple[str, ...] = Field(default=(), max_length=100)
    capability_refs: tuple[str, ...] = Field(default=(), max_length=100)
    source_event_refs: tuple[str, ...] = Field(default=(), max_length=40)
    source_round: int | None = Field(default=None, ge=1, le=100)
    registry_digest: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    reason_code: str = Field(pattern=r"^[A-Z][A-Z0-9_]{1,79}$")
    rationale: str = Field(min_length=1, max_length=2000)

    @model_validator(mode="after")
    def require_replayable_basis(self):
        if not (
            self.observation_refs
            or self.query_refs
            or self.capability_refs
            or self.source_event_refs
            or self.source_round is not None
        ):
            raise ValueError("transition basis has no replayable reference")
        return self


class CandidateTransition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    transition_id: str = Field(pattern=r"^transition-[a-f0-9]{20}$")
    candidate_track_id: str = Field(pattern=r"^track-[a-f0-9]{20}$")
    transition_type: CandidateTransitionType
    from_stage: CandidateStage | None
    to_stage: CandidateStage
    from_disposition: CandidateDisposition | None
    to_disposition: CandidateDisposition
    previous_semantic_digest: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    resulting_semantic_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    mechanism_summary: str | None = Field(default=None, min_length=1, max_length=2000)
    alternatives_added: tuple[InvestigationAlternative, ...] = Field(
        default=(), max_length=12
    )
    source_hypothesis_ids_added: tuple[str, ...] = Field(default=(), max_length=20)
    formal_mechanism_id: str | None = Field(
        default=None, pattern=r"^mech-[a-f0-9]{20}$"
    )
    compiled_candidate_ids_added: tuple[str, ...] = Field(default=(), max_length=20)
    close_reason: CandidateCloseReason | None = None
    basis: TransitionBasis
    sequence: int = Field(ge=1)


class WorkspaceCandidateAdded(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate: InvestigationCandidate
    transition: CandidateTransition


class CandidateSuperseded(BaseModel):
    """Atomic replacement of one investigation track by a semantically different successor."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    transition_id: str = Field(pattern=r"^transition-[a-f0-9]{20}$")
    predecessor_track_id: str = Field(pattern=r"^track-[a-f0-9]{20}$")
    predecessor_semantic_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    successor: InvestigationCandidate
    basis: TransitionBasis
    sequence: int = Field(ge=1)


class DiscriminationAttempt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    attempt_id: str = Field(pattern=r"^attempt-[a-f0-9]{20}$")
    task_id: str = Field(pattern=r"^task-[a-f0-9]{20}$")
    capability_ref: str = Field(pattern=r"^cap-[a-f0-9]{20}$")
    query_ref: str = Field(min_length=1, max_length=120)
    query_fingerprint: str = Field(pattern=r"^qry-[a-f0-9]{16}$")
    query_cost: int = Field(ge=0)
    independent_group: str = Field(min_length=1, max_length=100)
    registry_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    observation_refs: tuple[str, ...] = Field(default=(), max_length=100)
    tool_status: str | None = Field(default=None, max_length=40)
    outcome: AttemptOutcome | None = None
    selection_reason_codes: tuple[str, ...] = Field(default=(), max_length=20)
    selected_sequence: int = Field(ge=1)
    completed_sequence: int | None = Field(default=None, ge=1)


class ResolvedCapability(BaseModel):
    """Transcript-safe view of one executable Registry handle and its bounded query."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    capability_ref: str = Field(pattern=r"^cap-[a-f0-9]{20}$")
    entity_ref: str = Field(pattern=r"^ent-[a-z0-9-]{3,40}$")
    rule_id: str = Field(min_length=1, max_length=80)
    signal: str = Field(min_length=1, max_length=120)
    independent_group: str = Field(min_length=1, max_length=120)
    support_eligible: bool
    refutes: tuple[str, ...] = Field(default=(), max_length=20)
    query_fingerprint: str = Field(pattern=r"^qry-[a-f0-9]{16}$")
    tool_name: str = Field(min_length=1, max_length=80)
    query_cost: int = Field(ge=0)
    executable: bool
    query_action: dict[str, Any] | None = Field(
        default=None, exclude_if=lambda value: value is None,
    )


class CapabilityRegistrySnapshot(BaseModel):
    """Episode-specific Registry projection; it grants no verification authority."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    registry_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    capabilities: tuple[ResolvedCapability, ...] = Field(default=(), max_length=500)
    snapshot_digest: str = Field(pattern=r"^[a-f0-9]{64}$")

    def verify_digest(self) -> bool:
        return sha256_digest(
            self.model_dump(mode="json", exclude={"snapshot_digest"})
        ) == self.snapshot_digest


class DiscriminationTask(BaseModel):
    """One explicit need to distinguish a candidate from a competing explanation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str = Field(pattern=r"^task-[a-f0-9]{20}$")
    candidate_track_id: str = Field(pattern=r"^track-[a-f0-9]{20}$")
    source_need_id: str = Field(min_length=1, max_length=80)
    source_round: int = Field(ge=1, le=100)
    source_binding_id: str | None = Field(default=None, exclude_if=lambda value: value is None)
    question: str = Field(min_length=1, max_length=2000)
    alternative_id: str | None = Field(default=None, pattern=r"^ialt-[a-f0-9]{20}$")
    compared_candidate_track_id: str | None = Field(
        default=None, pattern=r"^track-[a-f0-9]{20}$"
    )
    observable_kind: str = Field(min_length=1, max_length=120)
    candidate_prediction: str = Field(min_length=1, max_length=2000)
    comparator_prediction: str = Field(min_length=1, max_length=2000)
    importance: DiscriminationImportance
    status: DiscriminationTaskStatus
    capability_options: tuple[ResolvedCapability, ...] = Field(default=(), max_length=100)
    legacy_choice_ref: str | None = Field(default=None, max_length=120)
    legacy_query_ref: str | None = Field(default=None, max_length=120)
    attempts: tuple[DiscriminationAttempt, ...] = Field(default=(), max_length=20)
    selected_attempt_id: str | None = Field(
        default=None, pattern=r"^attempt-[a-f0-9]{20}$"
    )
    created_registry_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    last_evaluated_registry_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    unresolved_capability_reason: str | None = Field(default=None, max_length=1000)
    supersedes_task_id: str | None = Field(default=None, pattern=r"^task-[a-f0-9]{20}$")
    created_sequence: int = Field(ge=1)
    last_transition_sequence: int = Field(ge=1)
    scheduling: dict[str, Any] | None = Field(
        default=None, exclude_if=lambda value: value is None,
    )
    query_strategy_version: Literal[2] | None = Field(
        default=None, exclude_if=lambda value: value is None,
    )

    @model_validator(mode="after")
    def require_one_comparator(self):
        if (self.alternative_id is None) == (self.compared_candidate_track_id is None):
            raise ValueError("task requires exactly one alternative or candidate comparator")
        return self


class DiscriminationTaskSelected(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str = Field(pattern=r"^task-[a-f0-9]{20}$")
    selection_mode: Literal["shadow", "active"] = "shadow"
    from_status: Literal[DiscriminationTaskStatus.OPEN] = DiscriminationTaskStatus.OPEN
    attempt: DiscriminationAttempt
    sequence: int = Field(ge=1)


class DiscriminationTaskCompleted(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str = Field(pattern=r"^task-[a-f0-9]{20}$")
    attempt_id: str = Field(pattern=r"^attempt-[a-f0-9]{20}$")
    from_status: Literal[DiscriminationTaskStatus.SELECTED] = (
        DiscriminationTaskStatus.SELECTED
    )
    to_status: Literal[
        DiscriminationTaskStatus.RESOLVED,
        DiscriminationTaskStatus.AMBIGUOUS,
        DiscriminationTaskStatus.UNVERIFIABLE,
    ]
    observation_refs: tuple[str, ...] = Field(default=(), max_length=100)
    tool_status: str = Field(min_length=1, max_length=40)
    outcome: AttemptOutcome
    reason_code: str = Field(pattern=r"^[A-Z][A-Z0-9_]{1,79}$")
    sequence: int = Field(ge=1)


class DiscriminationTaskUnverifiable(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str = Field(pattern=r"^task-[a-f0-9]{20}$")
    from_status: Literal[DiscriminationTaskStatus.OPEN] = DiscriminationTaskStatus.OPEN
    reason_code: str = Field(pattern=r"^[A-Z][A-Z0-9_]{1,79}$")
    reason: str = Field(min_length=1, max_length=1000)
    registry_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    sequence: int = Field(ge=1)


class DiscriminationMatrixEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    rank: int = Field(ge=1)
    task_id: str = Field(pattern=r"^task-[a-f0-9]{20}$")
    candidate_track_id: str = Field(pattern=r"^track-[a-f0-9]{20}$")
    capability_ref: str = Field(pattern=r"^cap-[a-f0-9]{20}$")
    query_fingerprint: str = Field(pattern=r"^qry-[a-f0-9]{16}$")
    query_cost: int = Field(ge=0)
    independent_group: str = Field(min_length=1, max_length=120)
    score: int
    reason_codes: tuple[str, ...] = Field(default=(), max_length=20)
    legacy_selected: bool


class DiscriminationMatrix(BaseModel):
    """Derived ranking only; never a persisted source of candidate truth."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    registry_digest: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    entries: tuple[DiscriminationMatrixEntry, ...] = ()
    matrix_digest: str = Field(pattern=r"^[a-f0-9]{64}$")

    def verify_digest(self) -> bool:
        return sha256_digest(
            self.model_dump(mode="json", exclude={"matrix_digest"})
        ) == self.matrix_digest


class DiscriminationShadowSelection(BaseModel):
    """Persisted diagnostic comparison, not an executed-query instruction."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    selection_id: str = Field(pattern=r"^selection-[a-f0-9]{20}$")
    matrix_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    evaluated_task_ids: tuple[str, ...] = Field(default=(), max_length=100)
    selected_task_id: str | None = Field(default=None, pattern=r"^task-[a-f0-9]{20}$")
    selected_capability_ref: str | None = Field(
        default=None, pattern=r"^cap-[a-f0-9]{20}$"
    )
    legacy_first_task_id: str | None = Field(
        default=None, pattern=r"^task-[a-f0-9]{20}$"
    )
    matches_legacy_first: bool | None = None
    reason_codes: tuple[str, ...] = Field(default=(), max_length=20)
    sequence: int = Field(ge=1)


class DiscriminationActiveSelection(BaseModel):
    """Replayable Matrix choice made before one model-offered query executes."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    selection_id: str = Field(pattern=r"^active-selection-[a-f0-9]{20}$")
    source_round: int = Field(ge=1, le=100)
    matrix_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    offered_query_fingerprints: tuple[str, ...] = Field(max_length=10)
    selected_task_id: str = Field(pattern=r"^task-[a-f0-9]{20}$")
    selected_capability_ref: str = Field(pattern=r"^cap-[a-f0-9]{20}$")
    selected_query_fingerprint: str = Field(pattern=r"^qry-[a-f0-9]{16}$")
    suppressed_query_fingerprints: tuple[str, ...] = Field(default=(), max_length=10)
    reason_codes: tuple[str, ...] = Field(default=(), max_length=20)
    sequence: int = Field(ge=1)
    strategy_version: Literal[2] | None = Field(
        default=None, exclude_if=lambda value: value is None,
    )
    pending_query_fingerprints: tuple[str, ...] | None = Field(
        default=None, max_length=10, exclude_if=lambda value: value is None,
    )


class InvestigationFrontier(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tentative_candidate_track_ids: tuple[str, ...] = ()
    active_candidate_track_ids: tuple[str, ...] = ()
    deferred_candidate_track_ids: tuple[str, ...] = ()
    closed_candidate_track_ids: tuple[str, ...] = ()
    open_candidate_track_ids: tuple[str, ...] = ()
    disfavored_candidate_track_ids: tuple[str, ...] = ()
    formalized_candidate_track_ids: tuple[str, ...] = ()


class CandidateReachabilityView(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_track_id: str = Field(pattern=r"^track-[a-f0-9]{20}$")
    reachability: CandidateVerificationReachability
    primary_task_ids: tuple[str, ...] = ()


class InvestigationWorkspaceSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    episode_id: str = Field(pattern=r"^ep-[a-f0-9]{32}$")
    schema_version: Literal["investigation-workspace-v1"] = WORKSPACE_SCHEMA_VERSION
    reducer_version: Literal["investigation-reducer-v1", "investigation-reducer-v2"] = WORKSPACE_REDUCER_VERSION
    candidates: tuple[InvestigationCandidate, ...] = ()
    tasks: tuple[DiscriminationTask, ...] = ()
    frontier: InvestigationFrontier
    reachability: tuple[CandidateReachabilityView, ...] = ()
    discrimination_matrix: DiscriminationMatrix | None = None
    shadow_selections: tuple[DiscriminationShadowSelection, ...] = ()
    active_selections: tuple[DiscriminationActiveSelection, ...] = ()
    inquiry_loop: dict[str, Any] | None = Field(default=None, exclude_if=lambda v: v is None)
    transcript_head_sequence: int = Field(ge=1)
    transcript_head_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    registry_digest: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    snapshot_digest: str = Field(pattern=r"^[a-f0-9]{64}$")

    def verify_digest(self) -> bool:
        current = self.model_dump(mode="json", exclude={"snapshot_digest"})
        if sha256_digest(current) == self.snapshot_digest:
            return True
        previous_phase = dict(current)
        previous_phase.pop("active_selections", None)
        if sha256_digest(previous_phase) == self.snapshot_digest:
            return True
        if (
            self.discrimination_matrix is not None
            or self.shadow_selections
            or self.active_selections
        ):
            return False
        legacy = dict(current)
        legacy.pop("discrimination_matrix", None)
        legacy.pop("shadow_selections", None)
        if sha256_digest(legacy) == self.snapshot_digest:
            return True
        legacy.pop("active_selections", None)
        return sha256_digest(legacy) == self.snapshot_digest
