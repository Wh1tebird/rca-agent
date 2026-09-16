from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rca_agent.models.actions import HypothesisProposal, StopReason
from rca_agent.tools.fingerprint import sha256_digest


class EpisodeStatus(StrEnum):
    CREATED = "created"
    ACTIVE = "active"
    ANSWER_FROZEN = "answer_frozen"
    EVALUATED = "evaluated"


class DecisionStatus(StrEnum):
    INCONCLUSIVE = "inconclusive"
    SUPPORTED_HYPOTHESIS = "supported_hypothesis"
    MULTIPLE_SUPPORTED = "multiple_supported"
    CONFIRMED_ROOT_CAUSE = "confirmed_root_cause"
    DISCOVERED_VERIFIED = "discovered_verified"
    OUT_OF_SCOPE = "out_of_scope"
    QUERY_FAILED = "query_failed"


class DecisionSource(StrEnum):
    CORE = "rca_agent_core"
    PHASE2_LIFECYCLE_DEFAULT = "phase2_lifecycle_default"


class CandidateEvidenceStatus(StrEnum):
    UNASSESSED = "unassessed"
    INCONCLUSIVE = "inconclusive"
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    CONFIRMED = "confirmed"
    DISCOVERED_VERIFIED = "discovered_verified"


class AuthoritativeDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: DecisionStatus
    source: DecisionSource
    winning_hypothesis_id: str | None = None
    supported_hypothesis_ids: tuple[str, ...] = ()
    reason: str = Field(min_length=1, max_length=1000)

    @model_validator(mode="after")
    def confirmation_requires_core_authority(self) -> AuthoritativeDecision:
        if self.status in {DecisionStatus.CONFIRMED_ROOT_CAUSE, DecisionStatus.DISCOVERED_VERIFIED} and (
            self.source != DecisionSource.CORE or self.winning_hypothesis_id is None
        ):
            raise ValueError("confirmed root cause requires domain-core authority and a winner")
        if self.status == DecisionStatus.MULTIPLE_SUPPORTED and (
            self.source != DecisionSource.CORE
            or self.winning_hypothesis_id is not None
            or len(self.supported_hypothesis_ids) < 2
        ):
            raise ValueError("multiple supported requires Core authority and at least two candidates")
        return self


class CandidateAssessmentPreview(BaseModel):
    """Threshold-free view of one Core-owned candidate assessment."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_id: str
    hypothesis_ids: tuple[str, ...]
    evidence_status: CandidateEvidenceStatus
    satisfied_signals: tuple[str, ...] = ()
    missing_signals: tuple[str, ...] = ()
    contradicting_signals: tuple[str, ...] = ()
    unavailable_signals: tuple[str, ...] = ()
    required_capabilities: tuple[str, ...] = ()
    verification_pending: bool = False


class DiscriminatingSignalView(BaseModel):
    """Safe query gap shared with Policy; no comparison rule or score is exposed."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    signal: str
    candidate_ids: tuple[str, ...]
    compared_candidate_ids: tuple[str, ...]
    required_capability: str
    required_tool: str | None = None
    issued_observation_refs: tuple[str, ...] = ()
    available: bool
    reason_code: Literal["separates_unresolved_candidates"] = (
        "separates_unresolved_candidates"
    )


class CoreAssessmentPreview(BaseModel):
    """Read-only multi-candidate projection used for orchestration routing."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    overall_decision: AuthoritativeDecision
    candidate_assessments: tuple[CandidateAssessmentPreview, ...] = ()
    discriminating_signals: tuple[DiscriminatingSignalView, ...] = ()


class CandidateFrontier(BaseModel):
    """Graph projection derived exclusively from the latest Core preview."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    active_candidate_ids: tuple[str, ...] = ()
    supported_candidate_ids: tuple[str, ...] = ()
    contradicted_candidate_ids: tuple[str, ...] = ()
    inconclusive_candidate_ids: tuple[str, ...] = ()
    confirmed_candidate_ids: tuple[str, ...] = ()
    discovered_verified_candidate_ids: tuple[str, ...] = ()
    unsubmitted_candidate_ids: tuple[str, ...] = ()


class AnswerArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    episode_id: str = Field(pattern=r"^ep-[a-f0-9]{32}$")
    termination_reason: StopReason
    policy_proposals: tuple[HypothesisProposal, ...]
    authoritative_decision: AuthoritativeDecision
    observation_refs: tuple[str, ...]
    missing_evidence: tuple[str, ...]
    transcript_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    component_versions: dict[str, str]
    frozen_at: datetime
    artifact_digest: str = Field(pattern=r"^[a-f0-9]{64}$")

    def verify_digest(self) -> bool:
        payload = self.model_dump(mode="json", exclude={"artifact_digest"})
        return sha256_digest(payload) == self.artifact_digest
