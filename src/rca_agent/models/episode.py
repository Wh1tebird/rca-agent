from __future__ import annotations

from datetime import datetime
from enum import StrEnum

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
    CONFIRMED_ROOT_CAUSE = "confirmed_root_cause"
    OUT_OF_SCOPE = "out_of_scope"
    QUERY_FAILED = "query_failed"


class DecisionSource(StrEnum):
    CORE = "rca_agent_core"
    PHASE2_LIFECYCLE_DEFAULT = "phase2_lifecycle_default"


class AuthoritativeDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: DecisionStatus
    source: DecisionSource
    winning_hypothesis_id: str | None = None
    reason: str = Field(min_length=1, max_length=1000)

    @model_validator(mode="after")
    def confirmation_requires_core_authority(self) -> AuthoritativeDecision:
        if self.status == DecisionStatus.CONFIRMED_ROOT_CAUSE and (
            self.source != DecisionSource.CORE or self.winning_hypothesis_id is None
        ):
            raise ValueError("confirmed root cause requires domain-core authority and a winner")
        return self


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
