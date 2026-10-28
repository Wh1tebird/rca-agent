"""Provider-neutral evidence queries and bounded Observation responses.

This layer adapts the existing typed Environment calls without exposing backend
queries, filesystem paths, raw trace identifiers, or RCA interpretations.
EpisodeRunner remains the only caller and Observation admission point.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rca_agent.models import (
    CallToolAction,
    CompareMetricWindowsRequest,
    GetMetricSeriesRequest,
    GetServiceTraceSummaryRequest,
    GetTraceStageBreakdownRequest,
    InspectTopologyRequest,
    ListRelatedEntitiesRequest,
    Modality,
    Observation,
    QueryCost,
    TimeWindow,
    ToolStatus,
)
from rca_agent.tools.fingerprint import query_fingerprint

EvidenceOperation = Literal[
    "metric_series",
    "metric_comparison",
    "log_markers",
    "trace_summary",
    "trace_stages",
    "topology",
    "topology_related",
    "cmdb",
    "database",
]

_OPERATION_MODALITY: dict[EvidenceOperation, Modality] = {
    "metric_series": Modality.METRIC,
    "metric_comparison": Modality.METRIC,
    "log_markers": Modality.LOG,
    "trace_summary": Modality.TRACE,
    "trace_stages": Modality.TRACE,
    "topology": Modality.TOPOLOGY,
    "topology_related": Modality.TOPOLOGY,
    "cmdb": Modality.TOPOLOGY,
    "database": Modality.METRIC,
}
_FORBIDDEN_INTERPRETATION_KEYS = frozenset({
    "candidate_id", "contradiction", "root_cause", "support", "winner", "answer",
})


class EvidenceQuery(BaseModel):
    """Closed, bounded query contract with no raw backend expression."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    operation: EvidenceOperation
    modality: Modality
    entity_ref: str | None = Field(default=None, pattern=r"^ent-[a-z0-9-]{3,40}$")
    signal: str | None = Field(default=None, min_length=1, max_length=120)
    window: TimeWindow | None = None
    min_duration_ms: float = Field(default=100.0, ge=0.0, le=3_600_000.0)
    trace_ref: str | None = Field(default=None, pattern=r"^trace-[a-f0-9]{16}$")
    relation: str | None = Field(default=None, min_length=1, max_length=120)
    limit: int = Field(default=50, ge=1, le=100)
    cost_units: int = Field(default=1, ge=0, le=10)

    @model_validator(mode="after")
    def validate_shape(self) -> EvidenceQuery:
        if _OPERATION_MODALITY[self.operation] != self.modality:
            raise ValueError("operation and modality do not match")
        if self.operation != "trace_stages" and self.entity_ref is None:
            raise ValueError("entity scope is required for this evidence query")
        if self.operation in {"metric_series", "metric_comparison", "log_markers", "trace_summary"} \
                and self.window is None:
            raise ValueError("bounded time window is required")
        if self.operation == "trace_stages" and self.trace_ref is None:
            raise ValueError("trace stages require an issued opaque trace reference")
        if self.operation in {"metric_series", "metric_comparison"} and self.signal is None:
            raise ValueError("metric evidence requires a signal")
        if self.operation == "topology" and self.trace_ref is not None:
            raise ValueError("topology query cannot carry a trace reference")
        return self

    @property
    def query_ref(self) -> str:
        tool_name, request = self.to_legacy_call()
        return query_fingerprint(tool_name, request)

    def to_legacy_call(self) -> tuple[str, BaseModel]:
        if self.operation == "metric_series":
            return "get_metric_series", GetMetricSeriesRequest(
                entity_ref=self.entity_ref, signal=self.signal or "", window=self.window,
                max_points=self.limit,
            )
        if self.operation == "metric_comparison":
            return "compare_metric_windows", CompareMetricWindowsRequest(
                entity_ref=self.entity_ref, signal=self.signal or "", incident_window=self.window,
            )
        if self.operation == "trace_summary":
            return "get_service_trace_summary", GetServiceTraceSummaryRequest(
                entity_ref=self.entity_ref, window=self.window,
                min_duration_ms=self.min_duration_ms,
                max_traces=min(self.limit, 10),
            )
        if self.operation == "trace_stages":
            return "get_trace_stage_breakdown", GetTraceStageBreakdownRequest(
                trace_ref=self.trace_ref or "trace-0000000000000000", max_stages=min(self.limit, 12),
            )
        if self.operation == "topology":
            return "inspect_topology", InspectTopologyRequest(
                entity_ref=self.entity_ref, relation=self.relation, max_results=self.limit,
            )
        if self.operation == "topology_related":
            return "list_related_entities", ListRelatedEntitiesRequest(
                entity_ref=self.entity_ref, max_results=self.limit,
            )
        if self.operation == "log_markers":
            return "search_log_markers", ListRelatedEntitiesRequest(
                entity_ref=self.entity_ref, max_results=self.limit,
            )
        return "database_evidence", ListRelatedEntitiesRequest(
            entity_ref=self.entity_ref, max_results=self.limit,
        )


def evidence_query_from_action(action: CallToolAction) -> EvidenceQuery:
    """Translate an existing typed action without exposing backend query syntax."""
    request = action.request
    tool = action.tool_name.value
    if tool == "get_metric_series":
        return EvidenceQuery(
            operation="metric_series", modality=Modality.METRIC,
            entity_ref=request.entity_ref, signal=request.signal, window=request.window,
            limit=request.max_points,
        )
    if tool == "compare_metric_windows":
        return EvidenceQuery(
            operation="metric_comparison", modality=Modality.METRIC,
            entity_ref=request.entity_ref, signal=request.signal, window=request.incident_window,
        )
    if tool == "get_service_trace_summary":
        return EvidenceQuery(
            operation="trace_summary", modality=Modality.TRACE,
            entity_ref=request.entity_ref, window=request.window, limit=request.max_traces,
            min_duration_ms=request.min_duration_ms,
        )
    if tool == "get_trace_stage_breakdown":
        return EvidenceQuery(
            operation="trace_stages", modality=Modality.TRACE,
            trace_ref=request.trace_ref, limit=request.max_stages,
        )
    if tool == "inspect_topology":
        return EvidenceQuery(
            operation="topology", modality=Modality.TOPOLOGY,
            entity_ref=request.entity_ref, relation=request.relation, limit=request.max_results,
        )
    if tool == "list_related_entities":
        return EvidenceQuery(
            operation="topology_related", modality=Modality.TOPOLOGY,
            entity_ref=request.entity_ref, limit=request.max_results,
        )
    raise ValueError(f"unsupported EvidenceQuery action: {tool}")


class EvidenceCapability(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    capability_ref: str = Field(pattern=r"^evcap-[a-f0-9]{16}$")
    modality: Modality
    operation: EvidenceOperation
    max_window_seconds: int = Field(default=86_400, ge=1)
    max_results: int = Field(default=100, ge=1, le=100)
    cost_units: int = Field(default=1, ge=0, le=10)
    required_permission: str | None = Field(default=None, max_length=120)
    supported_query_fields: tuple[str, ...] = ()
    output_schema: str = "Observation-v1"
    entity_scope: Literal["episode_issued", "opaque_reference"] = "episode_issued"
    available: bool = True
    unavailable_reason: str | None = None


class EvidenceResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    query_ref: str = Field(pattern=r"^qry-[a-f0-9]{16}$")
    status: ToolStatus
    observations: tuple[Observation, ...] = ()
    cost: QueryCost
    message: str
    truncated: bool = False

    @model_validator(mode="after")
    def observations_are_neutral(self) -> EvidenceResult:
        for observation in self.observations:
            if _FORBIDDEN_INTERPRETATION_KEYS & set(observation.details):
                raise ValueError("Observation details contain an RCA interpretation")
        return self


class EvidenceProvider(Protocol):
    """Host capability adapter returning neutral Observation values only.

    Implementations do not receive a Candidate and cannot decide support,
    contradiction, winner or root cause. EpisodeRunner performs admission and
    records the result before CoreDecisionProvider evaluates it.
    """

    provider_name: str

    def capabilities(self) -> tuple[EvidenceCapability, ...]: ...

    def execute(self, query: EvidenceQuery, *, budget_remaining: int) -> EvidenceResult: ...


class EnvironmentEvidenceProvider:
    """Compatibility adapter over the existing typed RcaEnvironment API."""

    def __init__(self, environment: Any, *, granted_permissions: frozenset[str] = frozenset()) -> None:
        self._environment = environment
        self.provider_name = "environment"
        self._granted_permissions = granted_permissions

    def capabilities(self) -> tuple[EvidenceCapability, ...]:
        return tuple(_capability(operation, modality) for operation, modality in (
            ("metric_series", Modality.METRIC),
            ("metric_comparison", Modality.METRIC),
            ("trace_summary", Modality.TRACE),
            ("trace_stages", Modality.TRACE),
            ("topology", Modality.TOPOLOGY),
            ("topology_related", Modality.TOPOLOGY),
            ("log_markers", Modality.LOG),
        ))

    def execute(self, query: EvidenceQuery, *, budget_remaining: int) -> EvidenceResult:
        capability = next(item for item in self.capabilities() if item.operation == query.operation)
        window_seconds = query.window.seconds if query.window is not None else 0
        if query.limit > capability.max_results or window_seconds > capability.max_window_seconds:
            return _empty_result(query, ToolStatus.REJECTED, budget_remaining, 0,
                                 "evidence query exceeds capability bounds")
        if (capability.required_permission is not None
                and capability.required_permission not in self._granted_permissions):
            return _empty_result(query, ToolStatus.REJECTED, budget_remaining, 0,
                                 "evidence permission is unavailable")
        if not capability.available or budget_remaining < capability.cost_units:
            return _empty_result(query, ToolStatus.BUDGET_EXHAUSTED if budget_remaining < capability.cost_units
                                 else ToolStatus.CAPABILITY_UNAVAILABLE,
                                 budget_remaining, 0,
                                 capability.unavailable_reason or "evidence capability unavailable")
        tool_name, request = query.to_legacy_call()
        tool = getattr(self._environment, tool_name, None)
        if tool is None:
            return _empty_result(query, ToolStatus.CAPABILITY_UNAVAILABLE, budget_remaining, 0,
                                 "evidence capability unavailable")
        response = tool(request)
        if response.query_ref != query.query_ref:
            raise ValueError("environment query fingerprint diverged from EvidenceQuery")
        return EvidenceResult.model_validate(
            response.model_dump(mode="json", exclude={"tool_name"})
        )


class FixtureEvidenceProvider:
    """Deterministic provider for offline contract tests and local development."""

    def __init__(
        self,
        capability: EvidenceCapability,
        observations: Iterable[Observation] = (),
        *,
        granted_permissions: frozenset[str] = frozenset(),
    ) -> None:
        self._capability = capability
        self._observations = tuple(observations)
        self.provider_name = "deterministic_fixture"
        self._granted_permissions = granted_permissions

    def capabilities(self) -> tuple[EvidenceCapability, ...]:
        return (self._capability,)

    def execute(self, query: EvidenceQuery, *, budget_remaining: int) -> EvidenceResult:
        if query.operation != self._capability.operation or query.modality != self._capability.modality:
            return _empty_result(query, ToolStatus.CAPABILITY_UNAVAILABLE, budget_remaining, 0,
                                 "provider capability does not match query")
        if (self._capability.required_permission is not None
                and self._capability.required_permission not in self._granted_permissions):
            return _empty_result(query, ToolStatus.REJECTED, budget_remaining, 0,
                                 "evidence permission is unavailable")
        window_seconds = query.window.seconds if query.window is not None else 0
        if query.limit > self._capability.max_results or window_seconds > self._capability.max_window_seconds:
            return _empty_result(query, ToolStatus.REJECTED, budget_remaining, 0,
                                 "evidence query exceeds capability bounds")
        if budget_remaining < self._capability.cost_units:
            return _empty_result(query, ToolStatus.BUDGET_EXHAUSTED, budget_remaining, 0,
                                 "query budget exhausted")
        values = tuple(
            item for item in self._observations
            if (query.entity_ref is None or item.entity_ref == query.entity_ref)
            and (query.signal is None or item.signal == query.signal)
        )[:query.limit]
        status = ToolStatus.SUCCESS if values else ToolStatus.EMPTY
        return EvidenceResult(
            query_ref=query.query_ref,
            status=status,
            observations=values,
            cost=QueryCost(units=self._capability.cost_units,
                           budget_before=budget_remaining,
                           budget_after=budget_remaining - self._capability.cost_units),
            message="deterministic evidence fixture",
            truncated=len(values) >= query.limit,
        )


class MetricEvidenceProvider(FixtureEvidenceProvider):
    def __init__(self, observations: Iterable[Observation] = ()) -> None:
        super().__init__(_capability("metric_series", Modality.METRIC), observations)


class LogEvidenceProvider(FixtureEvidenceProvider):
    def __init__(self, observations: Iterable[Observation] = ()) -> None:
        super().__init__(_capability("log_markers", Modality.LOG, available=True), observations)


class TraceEvidenceProvider(FixtureEvidenceProvider):
    def __init__(self, observations: Iterable[Observation] = ()) -> None:
        super().__init__(_capability("trace_summary", Modality.TRACE), observations)


class TopologyEvidenceProvider(FixtureEvidenceProvider):
    def __init__(self, observations: Iterable[Observation] = ()) -> None:
        super().__init__(_capability("topology", Modality.TOPOLOGY), observations)


class UnavailableEvidenceProvider:
    """Explicit CMDB/database extension boundary with no fabricated data."""

    def __init__(self, operation: Literal["cmdb", "database"]) -> None:
        self._capability = _capability(operation, _OPERATION_MODALITY[operation])
        self.provider_name = "unavailable_stub"

    def capabilities(self) -> tuple[EvidenceCapability, ...]:
        return (self._capability,)

    def execute(self, query: EvidenceQuery, *, budget_remaining: int) -> EvidenceResult:
        return _empty_result(
            query, ToolStatus.CAPABILITY_UNAVAILABLE, budget_remaining, 0,
            self._capability.unavailable_reason or "provider is unavailable",
        )


class CompositeEvidenceProvider:
    def __init__(self, providers: Iterable[EvidenceProvider]) -> None:
        self._providers = tuple(providers)
        self.provider_name = "composite"

    def capabilities(self) -> tuple[EvidenceCapability, ...]:
        return tuple(capability for provider in self._providers for capability in provider.capabilities())

    def execute(self, query: EvidenceQuery, *, budget_remaining: int) -> EvidenceResult:
        for provider in self._providers:
            if any(item.operation == query.operation for item in provider.capabilities()):
                return provider.execute(query, budget_remaining=budget_remaining)
        return _empty_result(query, ToolStatus.CAPABILITY_UNAVAILABLE, budget_remaining, 0,
                             "no provider capability for this evidence query")


def _capability(
    operation: EvidenceOperation,
    modality: Modality,
    *,
    available: bool | None = None,
) -> EvidenceCapability:
    digest = query_fingerprint("evidence_capability", {"operation": operation})[4:]
    return EvidenceCapability(
        capability_ref="evcap-" + digest[:16],
        operation=operation,
        modality=modality,
        available=(operation not in {"log_markers", "cmdb", "database"}
                   if available is None else available),
        unavailable_reason=("deterministic stub only"
                            if operation in {"log_markers", "cmdb", "database"}
                            and available is not True else None),
    )


def _empty_result(
    query: EvidenceQuery,
    status: ToolStatus,
    budget_before: int,
    units: int,
    message: str,
) -> EvidenceResult:
    return EvidenceResult(
        query_ref=query.query_ref,
        status=status,
        cost=QueryCost(units=units, budget_before=budget_before, budget_after=budget_before),
        message=message,
    )
