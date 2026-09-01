from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from rca_agent.models.common import TimeWindow


class GetMetricSeriesRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    entity_ref: str = Field(pattern=r"^ent-[a-z0-9-]{3,40}$")
    signal: str = Field(min_length=1, max_length=120)
    window: TimeWindow
    max_points: int = Field(default=50, ge=1, le=100)


class CompareMetricWindowsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    entity_ref: str = Field(pattern=r"^ent-[a-z0-9-]{3,40}$")
    signal: str = Field(min_length=1, max_length=120)
    incident_window: TimeWindow


class ListRelatedEntitiesRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    entity_ref: str = Field(pattern=r"^ent-[a-z0-9-]{3,40}$")
    max_results: int = Field(default=20, ge=1, le=50)


class GetServiceTraceSummaryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    entity_ref: str = Field(pattern=r"^ent-[a-z0-9-]{3,40}$")
    window: TimeWindow
    min_duration_ms: float = Field(default=100.0, ge=0.0, le=3_600_000.0)
    max_traces: int = Field(default=5, ge=1, le=10)


class GetTraceStageBreakdownRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    trace_ref: str = Field(pattern=r"^trace-[a-f0-9]{16}$")
    max_stages: int = Field(default=8, ge=1, le=12)


class InspectTopologyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    entity_ref: str = Field(pattern=r"^ent-[a-z0-9-]{3,40}$")
    relation: str | None = Field(default=None, min_length=1, max_length=120)
    max_results: int = Field(default=20, ge=1, le=50)
