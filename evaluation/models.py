from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from rca_agent.tools.fingerprint import sha256_digest


class EvaluationOutcome(StrEnum):
    CORRECT = "correct"
    INCORRECT = "incorrect"
    INCONCLUSIVE = "inconclusive"
    NOT_EVALUABLE = "not_evaluable"


class FailureCategory(StrEnum):
    EXECUTION = "execution"
    PROTOCOL = "protocol"
    STATE = "state"
    INVESTIGATION = "investigation"
    GROUNDING = "grounding"
    CONTRACT = "contract"
    REPLAY = "replay"
    ADAPTER_ORCHESTRATION = "adapter_orchestration"


class FailureTaxonomyEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    category: FailureCategory
    failure_kind: str = Field(min_length=1, max_length=120)
    phase: str = Field(min_length=1, max_length=80)
    triggering_event_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    severity: Literal["info", "warning", "error", "fatal"]
    retryable: bool
    user_visible: bool
    provenance: str = Field(default="transcript_event", min_length=1, max_length=80)
    evidence: tuple[str, ...] = ()
    remediation_hint: str = Field(min_length=1, max_length=300)


class OutcomeEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    result: EvaluationOutcome
    detail_code: str


class EvidenceEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    result: Literal["supported", "unsupported", "insufficient", "not_evaluable"]
    valid_observation_count: int = Field(ge=0)
    unissued_observation_count: int = Field(ge=0)
    explicit_admission_count: int = Field(ge=0)
    cross_candidate_leak_count: int = Field(ge=0)
    ineligible_positive_count: int = Field(ge=0)
    same_episode: bool


class EfficiencyEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    query_count: int = Field(ge=0)
    budget_consumed: int = Field(ge=0)
    invalid_query_count: int = Field(ge=0)
    rejected_or_failed_query_count: int = Field(ge=0)
    duplicate_query_count: int = Field(ge=0)
    stop_quality: Literal["clean", "budget_exhausted", "failure", "manual", "unknown"]


class ContractEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    compliant: bool
    violation_count: int = Field(ge=0)


class ReplayEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    valid: bool
    head_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    answer_digest: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    workspace_digest: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    candidate_frontier_digest: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    stop_reason: str | None = None
    budget_remaining: int | None = Field(default=None, ge=0)


class EvaluationProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    evaluator_version: str
    schema_version: str = "evaluation-report-v1"
    phase: Literal["post_freeze"] = "post_freeze"
    record_digest: str = Field(pattern=r"^[a-f0-9]{64}$")


class EvaluationReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    episode_id: str = Field(pattern=r"^ep-[a-f0-9]{32}$")
    answer_artifact_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    outcome: OutcomeEvaluation
    evidence: EvidenceEvaluation
    efficiency: EfficiencyEvaluation
    contract: ContractEvaluation
    replay: ReplayEvaluation
    failure_taxonomy: tuple[FailureTaxonomyEntry, ...] = ()
    provenance: EvaluationProvenance
    report_digest: str = Field(pattern=r"^[a-f0-9]{64}$")

    def verify_digest(self) -> bool:
        return (
            sha256_digest(self.model_dump(mode="json", exclude={"report_digest"}))
            == self.report_digest
        )


class EvaluationArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    episode_id: str = Field(pattern=r"^ep-[a-f0-9]{32}$")
    answer_artifact_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    evaluator_version: str
    evaluated_at: datetime
    evaluation_status: str
    evaluator_record_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    artifact_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    report: EvaluationReport | None = None

    def verify_digest(self) -> bool:
        return (
            sha256_digest(self.model_dump(mode="json", exclude={"artifact_digest"}))
            == self.artifact_digest
        )
