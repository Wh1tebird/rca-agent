from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rca_agent.core.evidence_control import EvidenceAssertion, EvidenceDecision


class MechanismFamily(StrEnum):
    RESOURCE = "resource"
    NETWORK = "network"
    APPLICATION = "application"
    DATABASE_DEPENDENCY = "database_dependency"


class InvestigationStatus(StrEnum):
    CONTINUE_INVESTIGATION = "continue_investigation"
    SUPPORTED_HYPOTHESIS = "supported_hypothesis"
    CONFIRMED_ROOT_CAUSE = "confirmed_root_cause"
    INCONCLUSIVE = "inconclusive"
    OUT_OF_SCOPE = "out_of_scope"
    QUERY_FAILED = "query_failed"


class ConditionPolarity(StrEnum):
    SUPPORT = "support"
    CONTRADICT = "contradict"


class Comparison(StrEnum):
    GTE = "gte"
    LTE = "lte"
    EQ = "eq"
    CONTAINS = "contains"


class Incident(BaseModel):
    model_config = ConfigDict(extra="forbid")

    incident_id: str
    alert_title: str
    alert_entity: str
    window_start: str
    window_end: str


class ObservationCapability(BaseModel):
    model_config = ConfigDict(extra="forbid")

    capability_id: str
    modality: str
    signals: list[str] = Field(min_length=1)
    typed_query: str
    available: bool
    unavailable_reason: str | None = None


class EvidenceCondition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    condition_id: str
    candidate_id: str
    polarity: ConditionPolarity
    signal: str
    comparison: Comparison
    threshold: float | str
    capability_id: str
    independent_group: str
    strength: float = Field(gt=0.0, le=10.0)
    reason: str


class Candidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    fault_type: str
    family: MechanismFamily
    causal_entity: str
    proposal_reason: str = Field(min_length=1)
    positive_evidence_requirements: list[str] = Field(min_length=1)
    falsification_conditions: list[str] = Field(min_length=1)
    confounders: list[str] = Field(min_length=1)
    required_capabilities: list[str] = Field(min_length=1)
    conditions: list[EvidenceCondition] = Field(min_length=1)
    knowledge_provenance: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_candidate_contract(self) -> Candidate:
        condition_ids = {item.condition_id for item in self.conditions}
        if not set(self.positive_evidence_requirements) <= condition_ids:
            raise ValueError("positive requirements must reference candidate conditions")
        if not any(item.polarity == ConditionPolarity.SUPPORT for item in self.conditions):
            raise ValueError("candidate requires at least one positive condition")
        if any(item.candidate_id != self.candidate_id for item in self.conditions):
            raise ValueError("condition candidate IDs must match their owner")
        return self


class CandidateProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    generator: str
    candidates: list[Candidate]
    rejected_candidates: list[str] = Field(default_factory=list)


class ConditionAvailability(BaseModel):
    model_config = ConfigDict(extra="forbid")

    condition_id: str
    executable: bool
    reason: str


class TypedQueryPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    condition_id: str
    capability_id: str
    query_type: str
    signal: str
    bounded_window_start: str
    bounded_window_end: str


class Observation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    signal: str
    value: float | str
    query_ref: str
    observed_entity: str


class EvidenceAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    condition_id: str
    candidate_id: str
    matched: bool
    assertion: EvidenceAssertion | None
    observation: Observation | None
    unavailable_reason: str | None = None


class InvestigationRound(BaseModel):
    model_config = ConfigDict(extra="forbid")

    round_index: int = Field(ge=1)
    plans: list[TypedQueryPlan]
    assessments: list[EvidenceAssessment]
    decision: EvidenceDecision


class OpenRcaResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    incident: Incident
    capabilities: list[ObservationCapability]
    proposal: CandidateProposal
    condition_availability: list[ConditionAvailability]
    rounds: list[InvestigationRound]
    status: InvestigationStatus
    winner: str | None
    supported_hypotheses: list[str]
    missing_evidence: list[str]
    stop_reason: str
