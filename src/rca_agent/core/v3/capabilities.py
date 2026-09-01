from __future__ import annotations

from rca_agent.core.v3.models import (
    Candidate,
    ConditionAvailability,
    ObservationCapability,
    TypedQueryPlan,
)


def rca100_capabilities() -> list[ObservationCapability]:
    return [
        ObservationCapability(
            capability_id="node_resource_metrics",
            modality="metric",
            signals=[
                "node_cpu_peak",
                "node_cpu_high_fraction",
                "node_memory_peak",
                "node_disk_peak",
            ],
            typed_query="NodeResourceSummaryQuery",
            available=True,
        ),
        ObservationCapability(
            capability_id="service_metrics",
            modality="metric",
            signals=[
                "service_error_ratio",
                "service_latency_ratio",
                "service_workload_ratio",
                "service_cpu_ratio",
            ],
            typed_query="ServiceMechanismPreflightQuery",
            available=True,
        ),
        ObservationCapability(
            capability_id="trace_stage_decomposition",
            modality="trace",
            signals=[
                "network_stage_ms",
                "sql_duration_share",
                "sql_shape",
                "connection_duration_ms",
            ],
            typed_query="SlowTraceQuery+TraceByIdQuery",
            available=True,
        ),
        ObservationCapability(
            capability_id="bounded_log_markers",
            modality="log",
            signals=[
                "application_error_marker",
                "rate_limit_marker",
                "dependency_error_marker",
            ],
            typed_query="BoundedLogMarkerQuery",
            available=False,
            unavailable_reason="no bounded v3 log marker adapter is implemented yet",
        ),
        ObservationCapability(
            capability_id="network_metrics",
            modality="metric",
            signals=["network_error_ratio"],
            typed_query="NetworkMetricQuery",
            available=False,
            unavailable_reason="portable network loss metric is not mapped across datasets",
        ),
    ]


def resolve_conditions(
    candidates: list[Candidate], capabilities: list[ObservationCapability]
) -> list[ConditionAvailability]:
    by_id = {item.capability_id: item for item in capabilities}
    result = []
    for candidate in candidates:
        for condition in candidate.conditions:
            capability = by_id.get(condition.capability_id)
            executable = capability is not None and capability.available
            reason = (
                "typed capability is available"
                if executable
                else (
                    capability.unavailable_reason
                    if capability is not None
                    else "required capability is absent from inventory"
                )
            )
            result.append(
                ConditionAvailability(
                    condition_id=condition.condition_id,
                    executable=executable,
                    reason=reason or "capability unavailable",
                )
            )
    return result


def compile_typed_plans(
    candidates: list[Candidate],
    capabilities: list[ObservationCapability],
    window_start: str,
    window_end: str,
) -> list[TypedQueryPlan]:
    capability_by_id = {item.capability_id: item for item in capabilities}
    plans = []
    for candidate in candidates:
        for condition in candidate.conditions:
            capability = capability_by_id.get(condition.capability_id)
            if capability is None or not capability.available:
                continue
            plans.append(
                TypedQueryPlan(
                    condition_id=condition.condition_id,
                    capability_id=condition.capability_id,
                    query_type=capability.typed_query,
                    signal=condition.signal,
                    bounded_window_start=window_start,
                    bounded_window_end=window_end,
                )
            )
    return plans
