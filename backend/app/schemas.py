"""Public, GT-free contracts for the investigation console API."""
from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class CaseReadiness(StrEnum):
    READY = "ready"
    INCOMPLETE = "incomplete"
    INVALID = "invalid"


class RunStatus(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    RUNNING = "running"
    QUEUED = "queued"
    CANCELLED = "cancelled"


class ReplayStatus(StrEnum):
    VALID = "valid"
    INVALID = "invalid"
    NOT_AVAILABLE = "not_available"


class CaseSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str
    title: str
    alert_entity: str
    alert_entity_type: str
    triggered_at: datetime
    window_start: datetime
    window_end: datetime
    modalities: tuple[str, ...]
    readiness: CaseReadiness
    missing_modalities: tuple[str, ...] = ()
    run_count: int = Field(ge=0)


class CaseDetail(CaseSummary):
    prompt_text: str


class RunSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    case_id: str
    status: RunStatus
    replay_status: ReplayStatus
    started_at: datetime
    finished_at: datetime | None = None
    duration_ms: int | None = Field(default=None, ge=0)
    query_count: int = Field(ge=0)
    final_status: str | None = None
    profile_id: str
    model_label: str | None = None
    error_code: str | None = None


class RunCreateRequest(BaseModel):
    """Only server-approved profiles are accepted; no provider secrets or Core controls."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    profile_id: Literal["scripted-trace-demo", "qwen-investigation"] = "qwen-investigation"


class RunCreateResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    case_id: str
    status: RunStatus
    profile_id: str


class CandidateView(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    name: str
    mechanism: str
    status: str
    mentioned: bool
    formalized: bool
    compiled: bool
    hypothesis_id: str | None = None
    mechanism_id: str | None = None
    compiled_candidate_id: str | None = None
    summary: str
    evidence_ids: tuple[str, ...] = ()


class EvidenceView(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    title: str
    kind: str
    role: str
    value: str
    unit: str | None = None
    source: str
    time_range: str | None = None
    query_status: str
    observation: str
    interpretation: str
    candidate_ids: tuple[str, ...] = ()
    query_ref: str | None = None
    observation_ref: str | None = None
    capability_ref: str | None = None
    points: tuple[dict[str, Any], ...] = ()
    raw: dict[str, Any] = Field(default_factory=dict)


class TimelineStepView(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    sequence: int = Field(ge=1)
    time: datetime
    actor: str
    title: str
    summary: str
    status: str
    reasoning: str | None = None
    query: str | None = None
    observation: str | None = None
    decision: str | None = None
    gap: str | None = None
    evidence_id: str | None = None
    candidate_ids: tuple[str, ...] = ()
    developer: dict[str, Any] = Field(default_factory=dict)


class FinalDecisionView(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: str
    explanation: str
    supporting: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()
    alternatives: tuple[str, ...] = ()
    why_not_verified: str


class InvestigationRunView(BaseModel):
    """One read-only, post-freeze presentation of a recorded investigation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run: RunSummary
    case: CaseDetail
    candidates: tuple[CandidateView, ...]
    timeline: tuple[TimelineStepView, ...]
    evidence: tuple[EvidenceView, ...]
    final_decision: FinalDecisionView | None = None
    raw: dict[str, Any] = Field(default_factory=dict)


class CaseListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    items: tuple[CaseSummary, ...]
    total: int = Field(ge=0)


class RunListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    items: tuple[RunSummary, ...]
    total: int = Field(ge=0)


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: str
    cases_available: bool
