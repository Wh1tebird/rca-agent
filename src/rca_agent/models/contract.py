from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from rca_agent.models.common import EntityRef, Modality, TimeWindow


class CapabilityDescriptor(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tool_name: str
    modality: Modality
    available: bool
    unavailable_reason: str | None = None
    query_cost: int = Field(ge=0)
    max_window_seconds: int = Field(gt=0)
    max_results: int = Field(gt=0)


class QueryBudgetView(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    initial: int = Field(ge=0)
    remaining: int = Field(ge=0)


class ResetResponse(BaseModel):
    """The complete Agent-visible episode bootstrap payload."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    event_time: datetime
    alert_summary: str = Field(min_length=1)
    alert_entity: EntityRef
    observation_horizon: TimeWindow
    capabilities: tuple[CapabilityDescriptor, ...]
    query_budget: QueryBudgetView
