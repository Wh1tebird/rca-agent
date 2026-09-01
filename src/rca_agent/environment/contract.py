from __future__ import annotations

from typing import Protocol

from rca_agent.models import (
    CompareMetricWindowsRequest,
    GetMetricSeriesRequest,
    GetServiceTraceSummaryRequest,
    GetTraceStageBreakdownRequest,
    InspectTopologyRequest,
    ListRelatedEntitiesRequest,
    ResetResponse,
    ToolResponse,
)


class RcaEnvironment(Protocol):
    """Only this interface is supplied to an investigation policy."""

    def reset(self) -> ResetResponse: ...

    def get_metric_series(self, request: GetMetricSeriesRequest) -> ToolResponse: ...

    def compare_metric_windows(self, request: CompareMetricWindowsRequest) -> ToolResponse: ...

    def list_related_entities(self, request: ListRelatedEntitiesRequest) -> ToolResponse: ...

    def get_service_trace_summary(self, request: GetServiceTraceSummaryRequest) -> ToolResponse: ...

    def get_trace_stage_breakdown(self, request: GetTraceStageBreakdownRequest) -> ToolResponse: ...

    def inspect_topology(self, request: InspectTopologyRequest) -> ToolResponse: ...
