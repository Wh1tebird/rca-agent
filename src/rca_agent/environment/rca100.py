from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from rca_agent.environment._private_adapter import (
    EntityBinding,
    PrivateRca100Adapter,
    TraceSelection,
)
from rca_agent.environment.budget import QueryBudget
from rca_agent.models import (
    CapabilityDescriptor,
    CompareMetricWindowsRequest,
    DataStatus,
    EntityRef,
    GetMetricSeriesRequest,
    GetServiceTraceSummaryRequest,
    GetTraceStageBreakdownRequest,
    InspectTopologyRequest,
    ListRelatedEntitiesRequest,
    Modality,
    Observation,
    QueryBudgetView,
    QueryCost,
    ResetResponse,
    TimeWindow,
    ToolResponse,
    ToolStatus,
)
from rca_agent.tools.fingerprint import query_fingerprint, sha256_digest

MAX_WINDOW_SECONDS = 3600


class Rca100Environment:
    """One isolated RCA100 episode with a bounded Agent-facing interface."""

    def __init__(self, private_adapter: PrivateRca100Adapter, query_budget: int = 20) -> None:
        self.__adapter = private_adapter
        self.__budget = QueryBudget.create(query_budget)
        incident_duration = private_adapter.incident_end - private_adapter.incident_start
        self.__horizon = TimeWindow(
            start=private_adapter.incident_start - incident_duration,
            end=private_adapter.incident_end,
        )
        alert = private_adapter.alert_entity
        self.__entities: dict[str, EntityBinding] = {alert.public_ref: alert}
        self.__traces: dict[str, TraceSelection] = {}

    @classmethod
    def from_case(
        cls, cases_root: Path, case_selector: str, query_budget: int = 20
    ) -> Rca100Environment:
        """Host-only construction path. Investigation policies receive the built environment."""

        return cls(PrivateRca100Adapter(cases_root, case_selector), query_budget=query_budget)

    def reset(self) -> ResetResponse:
        self.__budget.reset()
        self.__traces.clear()
        alert = self.__adapter.alert_entity
        self.__entities = {alert.public_ref: alert}
        return ResetResponse(
            event_time=self.__adapter.event_time,
            alert_summary=self.__adapter.alert_summary,
            alert_entity=self._public_entity(alert),
            observation_horizon=self.__horizon,
            capabilities=self._capabilities(),
            query_budget=QueryBudgetView(
                initial=self.__budget.initial, remaining=self.__budget.remaining
            ),
        )

    def get_metric_series(self, request: GetMetricSeriesRequest) -> ToolResponse:
        tool = "get_metric_series"
        query_ref = _query_ref(tool, request)
        rejection = self._validate_entity_and_window(
            tool, query_ref, request.entity_ref, request.window, cost=1
        )
        if rejection is not None:
            return rejection
        entity = self.__entities[request.entity_ref]
        target = self.__adapter.metric_target(entity, request.signal)
        if target is None:
            return self._rejected(tool, query_ref, "signal is not available for this entity type")
        return self._execute(
            tool,
            query_ref,
            1,
            lambda: self._metric_series_observations(request, entity, target, query_ref),
        )

    def compare_metric_windows(self, request: CompareMetricWindowsRequest) -> ToolResponse:
        tool = "compare_metric_windows"
        query_ref = _query_ref(tool, request)
        rejection = self._validate_entity_and_window(
            tool, query_ref, request.entity_ref, request.incident_window, cost=1
        )
        if rejection is not None:
            return rejection
        if (
            request.incident_window.start
            - (request.incident_window.end - request.incident_window.start)
            < self.__horizon.start
        ):
            return self._rejected(tool, query_ref, "equal baseline window exceeds episode horizon")
        entity = self.__entities[request.entity_ref]
        target = self.__adapter.metric_target(entity, request.signal)
        if target is None:
            return self._rejected(tool, query_ref, "signal is not available for this entity type")
        return self._execute(
            tool,
            query_ref,
            1,
            lambda: self._metric_comparison_observations(request, entity, target, query_ref),
        )

    def list_related_entities(self, request: ListRelatedEntitiesRequest) -> ToolResponse:
        tool = "list_related_entities"
        query_ref = _query_ref(tool, request)
        rejection = self._validate_entity(tool, query_ref, request.entity_ref, cost=1)
        if rejection is not None:
            return rejection

        def operation() -> tuple[ToolStatus, tuple[Observation, ...], str, bool]:
            related = self.__adapter.related_entities(self.__entities[request.entity_ref])
            truncated = len(related) > request.max_results
            observations = []
            for index, item in enumerate(related[: request.max_results]):
                binding = item["binding"]
                self.__entities[binding.public_ref] = binding
                observations.append(
                    self._observation(
                        query_ref,
                        index,
                        Modality.TOPOLOGY,
                        request.entity_ref,
                        "related_entity",
                        self._public_entity(binding).model_dump(mode="json"),
                        DataStatus.DERIVED
                        if item["relation"] == "operation_of"
                        else DataStatus.OBSERVED,
                        details={
                            "direction": item["direction"],
                            "relation": item["relation"],
                        },
                    )
                )
            status = ToolStatus.SUCCESS if observations else ToolStatus.EMPTY
            return status, tuple(observations), "bounded related entities returned", truncated

        return self._execute(tool, query_ref, 1, operation)

    def get_service_trace_summary(self, request: GetServiceTraceSummaryRequest) -> ToolResponse:
        tool = "get_service_trace_summary"
        query_ref = _query_ref(tool, request)
        rejection = self._validate_entity_and_window(
            tool, query_ref, request.entity_ref, request.window, cost=1
        )
        if rejection is not None:
            return rejection
        entity = self.__entities[request.entity_ref]
        if entity.entity_type not in {"apm.operation", "apm.service"}:
            return self._rejected(
                tool, query_ref, "trace summary requires an operation or service entity"
            )

        def operation() -> tuple[ToolStatus, tuple[Observation, ...], str, bool]:
            rows, selections = self.__adapter.trace_summary(
                entity,
                _to_ns(request.window.start),
                _to_ns(request.window.end),
                request.min_duration_ms,
                request.max_traces,
            )
            self.__traces.update({item.public_trace_ref: item for item in selections})
            observations = tuple(
                self._observation(
                    query_ref,
                    index,
                    Modality.TRACE,
                    request.entity_ref,
                    "operation_trace_duration_ms",
                    row["duration_ms"],
                    DataStatus.OBSERVED,
                    unit="ms",
                    observed_at=row["start_time"],
                    details={
                        "trace_ref": row["trace_ref"],
                        "status_code": row["status_code"],
                    },
                )
                for index, row in enumerate(rows)
            )
            status = ToolStatus.SUCCESS if observations else ToolStatus.EMPTY
            return status, observations, "bounded trace summaries returned", False

        return self._execute(tool, query_ref, 1, operation)

    def get_trace_stage_breakdown(self, request: GetTraceStageBreakdownRequest) -> ToolResponse:
        tool = "get_trace_stage_breakdown"
        query_ref = _query_ref(tool, request)
        if request.trace_ref not in self.__traces:
            return self._rejected(tool, query_ref, "trace reference was not issued by this episode")

        def operation() -> tuple[ToolStatus, tuple[Observation, ...], str, bool]:
            selection = self.__traces[request.trace_ref]
            try:
                values, resolved_entity, resolution = self.__adapter.trace_stage_breakdown(selection)
            except RuntimeError:
                missing = self._observation(
                    query_ref,
                    0,
                    Modality.TRACE,
                    selection.source_entity_ref,
                    "trace_stage_bundle",
                    None,
                    DataStatus.MISSING,
                    details={"trace_ref": request.trace_ref},
                )
                return (
                    ToolStatus.EMPTY,
                    (missing,),
                    "trace has no complete database stage bundle",
                    False,
                )
            self.__entities[resolved_entity.public_ref] = resolved_entity
            statuses = {
                "operation_duration_ms": DataStatus.OBSERVED,
                "connection_duration_ms": DataStatus.OBSERVED,
                "connection_acquisition_ms": DataStatus.DERIVED,
                "sql_duration_ms": DataStatus.OBSERVED,
                "db_client_span_duration_ms": DataStatus.DERIVED,
                "sql_duration_share": DataStatus.DERIVED,
                "sql_shape": DataStatus.DERIVED,
            }
            units = {
                "operation_duration_ms": "ms",
                "connection_duration_ms": "ms",
                "connection_acquisition_ms": "ms",
                "sql_duration_ms": "ms",
                "db_client_span_duration_ms": "ms",
                "sql_duration_share": "ratio",
                "sql_shape": None,
            }
            items = list(values.items())
            truncated = len(items) > request.max_stages
            observations = tuple(
                self._observation(
                    query_ref,
                    index,
                    Modality.TRACE,
                    resolved_entity.public_ref,
                    signal,
                    value,
                    statuses[signal],
                    unit=units[signal],
                    details={
                        "trace_ref": request.trace_ref,
                        "resolved_entity": self._public_entity(resolved_entity).model_dump(mode="json"),
                        "resolution": resolution,
                    },
                )
                for index, (signal, value) in enumerate(items[: request.max_stages])
            )
            return ToolStatus.SUCCESS, observations, "bounded trace-stage facts returned", truncated

        return self._execute(tool, query_ref, 1, operation)

    def inspect_topology(self, request: InspectTopologyRequest) -> ToolResponse:
        tool = "inspect_topology"
        query_ref = _query_ref(tool, request)
        rejection = self._validate_entity(tool, query_ref, request.entity_ref, cost=1)
        if rejection is not None:
            return rejection

        def operation() -> tuple[ToolStatus, tuple[Observation, ...], str, bool]:
            related = self.__adapter.related_entities(self.__entities[request.entity_ref])
            if request.relation is not None:
                related = [item for item in related if item["relation"] == request.relation]
            truncated = len(related) > request.max_results
            observations = []
            for index, item in enumerate(related[: request.max_results]):
                binding = item["binding"]
                self.__entities[binding.public_ref] = binding
                observations.append(
                    self._observation(
                        query_ref,
                        index,
                        Modality.TOPOLOGY,
                        request.entity_ref,
                        "topology_edge",
                        item["relation"],
                        DataStatus.DERIVED
                        if item["relation"] == "operation_of"
                        else DataStatus.OBSERVED,
                        details={
                            "direction": item["direction"],
                            "neighbor": self._public_entity(binding).model_dump(mode="json"),
                        },
                    )
                )
            status = ToolStatus.SUCCESS if observations else ToolStatus.EMPTY
            return status, tuple(observations), "bounded topology facts returned", truncated

        return self._execute(tool, query_ref, 1, operation)

    def _metric_series_observations(
        self, request: GetMetricSeriesRequest, entity: EntityBinding, target: Any, query_ref: str
    ) -> tuple[ToolStatus, tuple[Observation, ...], str, bool]:
        rows, truncated = self.__adapter.metric_series(
            entity,
            target,
            _to_us(request.window.start),
            _to_us(request.window.end),
            request.max_points,
        )
        if not rows:
            missing = self._observation(
                query_ref,
                0,
                Modality.METRIC,
                request.entity_ref,
                request.signal,
                None,
                DataStatus.MISSING,
            )
            return ToolStatus.EMPTY, (missing,), "no samples in the requested scope", False
        observations = tuple(
            self._observation(
                query_ref,
                index,
                Modality.METRIC,
                request.entity_ref,
                request.signal,
                row["value"],
                DataStatus.OBSERVED,
                unit=target.unit,
                observed_at=row["time"],
            )
            for index, row in enumerate(rows)
        )
        status = ToolStatus.COVERAGE_INSUFFICIENT if truncated else ToolStatus.SUCCESS
        message = (
            "result cap reached; returned samples are partial" if truncated else "samples returned"
        )
        return status, observations, message, truncated

    def _metric_comparison_observations(
        self,
        request: CompareMetricWindowsRequest,
        entity: EntityBinding,
        target: Any,
        query_ref: str,
    ) -> tuple[ToolStatus, tuple[Observation, ...], str, bool]:
        row = self.__adapter.compare_metric_windows(
            entity,
            target,
            _to_us(request.incident_window.start),
            _to_us(request.incident_window.end),
        )
        if row is None:
            missing = self._observation(
                query_ref,
                0,
                Modality.METRIC,
                request.entity_ref,
                f"{request.signal}_window_comparison",
                None,
                DataStatus.MISSING,
                details={"coverage": "one_or_both_windows_missing"},
            )
            return (
                ToolStatus.COVERAGE_INSUFFICIENT,
                (missing,),
                "equal-window comparison lacks source coverage",
                False,
            )
        value = {
            "before_mean": row["before_mean"],
            "incident_mean": row["incident_mean"],
            "incident_min": row["incident_min"],
            "incident_max": row["incident_max"],
            "before_count": row["before_count"],
            "incident_count": row["incident_count"],
            "ratio": row["ratio"],
        }
        status = DataStatus.DERIVED if row["ratio"] is not None else DataStatus.UNKNOWN
        observation = self._observation(
            query_ref,
            0,
            Modality.METRIC,
            request.entity_ref,
            f"{request.signal}_window_comparison",
            value,
            status,
            unit=target.unit,
        )
        return ToolStatus.SUCCESS, (observation,), "equal-window comparison returned", False

    def _validate_entity_and_window(
        self,
        tool: str,
        query_ref: str,
        entity_ref: str,
        window: TimeWindow,
        cost: int,
    ) -> ToolResponse | None:
        entity_rejection = self._validate_entity(tool, query_ref, entity_ref, cost)
        if entity_rejection is not None:
            return entity_rejection
        if window.seconds > MAX_WINDOW_SECONDS:
            return self._rejected(tool, query_ref, "query window exceeds maximum duration")
        if window.start < self.__horizon.start or window.end > self.__horizon.end:
            return self._rejected(tool, query_ref, "query window exceeds episode horizon")
        return None

    def _validate_entity(
        self, tool: str, query_ref: str, entity_ref: str, cost: int
    ) -> ToolResponse | None:
        if entity_ref not in self.__entities:
            return self._rejected(tool, query_ref, "entity is not authorized in this episode")
        if not self.__budget.can_spend(cost):
            return ToolResponse(
                tool_name=tool,
                query_ref=query_ref,
                status=ToolStatus.BUDGET_EXHAUSTED,
                cost=QueryCost(
                    units=0,
                    budget_before=self.__budget.remaining,
                    budget_after=self.__budget.remaining,
                ),
                message="query budget exhausted",
            )
        return None

    def _execute(
        self,
        tool: str,
        query_ref: str,
        cost: int,
        operation: Callable[[], tuple[ToolStatus, tuple[Observation, ...], str, bool]],
    ) -> ToolResponse:
        before, after = self.__budget.spend(cost)
        try:
            status, observations, message, truncated = operation()
        except Exception:  # noqa: BLE001 - private backend detail must not cross the boundary
            return ToolResponse(
                tool_name=tool,
                query_ref=query_ref,
                status=ToolStatus.QUERY_FAILED,
                cost=QueryCost(units=cost, budget_before=before, budget_after=after),
                message="bounded query failed inside the environment",
            )
        return ToolResponse(
            tool_name=tool,
            query_ref=query_ref,
            status=status,
            observations=observations,
            cost=QueryCost(units=cost, budget_before=before, budget_after=after),
            message=message,
            truncated=truncated,
        )

    def _rejected(self, tool: str, query_ref: str, message: str) -> ToolResponse:
        return ToolResponse(
            tool_name=tool,
            query_ref=query_ref,
            status=ToolStatus.REJECTED,
            cost=QueryCost(
                units=0,
                budget_before=self.__budget.remaining,
                budget_after=self.__budget.remaining,
            ),
            message=message,
        )

    @staticmethod
    def _public_entity(binding: EntityBinding) -> EntityRef:
        return EntityRef(
            entity_ref=binding.public_ref,
            display_name=binding.name,
            entity_type=binding.entity_type,
        )

    @staticmethod
    def _observation(
        query_ref: str,
        index: int,
        modality: Modality,
        entity_ref: str,
        signal: str,
        value: Any,
        data_status: DataStatus,
        *,
        unit: str | None = None,
        observed_at: datetime | None = None,
        details: dict[str, Any] | None = None,
    ) -> Observation:
        return Observation(
            observation_id=f"obs-{sha256_digest(f'{query_ref}:{index}:{signal}')[:16]}",
            modality=modality,
            entity_ref=entity_ref,
            signal=signal,
            value=value,
            unit=unit,
            data_status=data_status,
            observed_at=observed_at,
            details=details or {},
        )

    @staticmethod
    def _capabilities() -> tuple[CapabilityDescriptor, ...]:
        return (
            CapabilityDescriptor(
                tool_name="get_metric_series",
                modality=Modality.METRIC,
                available=True,
                query_cost=1,
                max_window_seconds=MAX_WINDOW_SECONDS,
                max_results=100,
            ),
            CapabilityDescriptor(
                tool_name="compare_metric_windows",
                modality=Modality.METRIC,
                available=True,
                query_cost=1,
                max_window_seconds=MAX_WINDOW_SECONDS,
                max_results=1,
            ),
            CapabilityDescriptor(
                tool_name="list_related_entities",
                modality=Modality.TOPOLOGY,
                available=True,
                query_cost=1,
                max_window_seconds=MAX_WINDOW_SECONDS,
                max_results=50,
            ),
            CapabilityDescriptor(
                tool_name="get_service_trace_summary",
                modality=Modality.TRACE,
                available=True,
                query_cost=1,
                max_window_seconds=MAX_WINDOW_SECONDS,
                max_results=10,
            ),
            CapabilityDescriptor(
                tool_name="get_trace_stage_breakdown",
                modality=Modality.TRACE,
                available=True,
                query_cost=1,
                max_window_seconds=MAX_WINDOW_SECONDS,
                max_results=12,
            ),
            CapabilityDescriptor(
                tool_name="inspect_topology",
                modality=Modality.TOPOLOGY,
                available=True,
                query_cost=1,
                max_window_seconds=MAX_WINDOW_SECONDS,
                max_results=50,
            ),
            CapabilityDescriptor(
                tool_name="search_log_markers",
                modality=Modality.LOG,
                available=False,
                unavailable_reason="bounded log semantics are not reliable in phase 1",
                query_cost=1,
                max_window_seconds=MAX_WINDOW_SECONDS,
                max_results=50,
            ),
        )


def _to_us(value: datetime) -> int:
    return int(value.timestamp() * 1_000_000)


def _to_ns(value: datetime) -> int:
    return int(value.timestamp() * 1_000_000_000)


def _query_ref(tool: str, request: Any) -> str:
    return query_fingerprint(tool, request)
