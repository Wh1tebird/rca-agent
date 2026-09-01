from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class DataStatus(StrEnum):
    """Trust state carried by every observation value."""

    OBSERVED = "observed"
    DERIVED = "derived"
    IMPUTED = "imputed"
    MODEL_INFERRED = "model_inferred"
    MISSING = "missing"
    UNKNOWN = "unknown"


class ToolStatus(StrEnum):
    SUCCESS = "success"
    EMPTY = "empty"
    CAPABILITY_UNAVAILABLE = "capability_unavailable"
    QUERY_FAILED = "query_failed"
    COVERAGE_INSUFFICIENT = "coverage_insufficient"
    REJECTED = "rejected"
    BUDGET_EXHAUSTED = "budget_exhausted"


class Modality(StrEnum):
    METRIC = "metric"
    TRACE = "trace"
    TOPOLOGY = "topology"
    LOG = "log"


class TimeWindow(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    start: datetime
    end: datetime

    @model_validator(mode="after")
    def validate_window(self) -> TimeWindow:
        if self.start.tzinfo is None or self.end.tzinfo is None:
            raise ValueError("time window must be timezone-aware")
        if self.end <= self.start:
            raise ValueError("window end must be after start")
        return self

    @property
    def seconds(self) -> float:
        return (self.end - self.start).total_seconds()


class EntityRef(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    entity_ref: str = Field(pattern=r"^ent-[a-z0-9-]{3,40}$")
    display_name: str = Field(min_length=1, max_length=300)
    entity_type: str = Field(min_length=1, max_length=120)


class Observation(BaseModel):
    """Candidate-neutral tool output; interpretation belongs to the Agent domain core."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    observation_id: str = Field(pattern=r"^obs-[a-f0-9]{16}$")
    modality: Modality
    entity_ref: str = Field(pattern=r"^ent-[a-z0-9-]{3,40}$")
    signal: str = Field(min_length=1, max_length=120)
    value: Any | None = None
    unit: str | None = Field(default=None, max_length=40)
    data_status: DataStatus
    observed_at: datetime | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class QueryCost(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    units: int = Field(ge=0)
    budget_before: int = Field(ge=0)
    budget_after: int = Field(ge=0)


class ToolResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tool_name: str
    query_ref: str = Field(pattern=r"^qry-[a-f0-9]{16}$")
    status: ToolStatus
    observations: tuple[Observation, ...] = ()
    cost: QueryCost
    message: str
    truncated: bool = False
