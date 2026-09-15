"""Initial RCA100 recipes grounded in issued entities/traces and existing rules.

Hosts can replace this builder with their monitoring inventory. No data is queried here.
"""

from rca_agent.core.dynamic import EvidenceRegistry, RuleSpec, SignalSpec
from rca_agent.models import (
    CallToolAction,
    DataStatus,
    GetMetricSeriesRequest,
    GetTraceStageBreakdownRequest,
    Observation,
    ResetResponse,
)

TRACE_SIGNAL_CONTRACTS = {
    "operation_duration_ms": ("ms", "end_to_end_latency", "operation latency is high", "gte", 100),
    "connection_duration_ms": (
        "ms", "database_connection_delay",
        "instrumented Hikari connection acquisition span is delayed", "gte", 100,
    ),
    "connection_acquisition_ms": (
        "ms", "database_connection_delay",
        "derived alias of the Hikari connection acquisition span is delayed", "gte", 100,
    ),
    "sql_duration_ms": (
        "ms", "database_query_duration", "instrumented JDBC client span is slow", "gte", 100,
    ),
    "db_client_span_duration_ms": (
        "ms", "database_query_duration",
        "derived alias of the instrumented JDBC client span is slow", "gte", 100,
    ),
    "sql_duration_share": (
        "ratio", "database_query_duration", "SQL client span dominates operation time", "gte", 0.8,
    ),
    "sql_shape": (
        None, "database_query_structure", "SQL has a known complex cross-join structure",
        "eq", "complex_cross_join",
    ),
}


def rca100_evidence_registry(
    bootstrap: ResetResponse,
    observations: tuple[Observation, ...],
) -> EvidenceRegistry:
    """Reuse existing trace predicates; expose only issued trace/entity query recipes."""
    tools = {c.tool_name: c for c in bootstrap.capabilities if c.available}
    specs: dict[tuple[str, str], SignalSpec] = {}
    rules: dict[str, RuleSpec] = {}
    trace_observations = [
        observation for observation in observations
        if observation.signal in TRACE_SIGNAL_CONTRACTS
        and observation.details.get("trace_ref")
        and observation.data_status in {DataStatus.OBSERVED, DataStatus.DERIVED}
    ]
    summary_trace_refs = [
        observation.details["trace_ref"] for observation in observations
        if observation.signal == "operation_trace_duration_ms"
        and observation.details.get("trace_ref")
    ]
    discovery_trace_refs = {observation.details["trace_ref"] for observation in trace_observations}
    attempted_trace_refs = {
        observation.details["trace_ref"] for observation in observations
        if observation.details.get("trace_ref")
        and observation.signal in {*TRACE_SIGNAL_CONTRACTS, "trace_stage_bundle"}
    }
    verification_trace_ref = next(
        (trace_ref for trace_ref in summary_trace_refs if trace_ref not in attempted_trace_refs),
        min(discovery_trace_refs, default=None),
    )
    trace_entities = sorted({observation.entity_ref for observation in trace_observations})
    if verification_trace_ref and "get_trace_stage_breakdown" in tools:
        for entity_ref in trace_entities:
            for signal, (unit, group, description, comparison, threshold) in (
                TRACE_SIGNAL_CONTRACTS.items()
            ):
                spec = SignalSpec(
                    signal=signal,
                    entity_ref=entity_ref,
                    data_type="string" if signal == "sql_shape" else "number",
                    unit=unit,
                    description=description,
                    independent_group=group,
                    query=CallToolAction(
                        tool_name="get_trace_stage_breakdown",
                        request=GetTraceStageBreakdownRequest(
                            trace_ref=verification_trace_ref,
                            max_stages=min(12, tools["get_trace_stage_breakdown"].max_results),
                        ),
                    ),
                )
                specs[(entity_ref, signal)] = spec
                rules[signal] = RuleSpec(
                    rule_id=signal,
                    signal=signal,
                    comparison=comparison,
                    threshold=threshold,
                    source="configuration",
                    provenance="RCA100 instrumented trace-stage contract",
                    description=description,
                )
            rules["sql_duration_share_low"] = RuleSpec(
                rule_id="sql_duration_share_low",
                signal="sql_duration_share",
                comparison="lte",
                threshold=0.2,
                source="configuration",
                provenance="RCA100 trace-stage differential contract",
                description="SQL client span is a small part of operation duration",
            )
    for observation in observations:
        if observation.data_status not in {DataStatus.OBSERVED, DataStatus.DERIVED}:
            continue
        if (
            observation.signal in {"cpu_usage_rate", "memory_usage_rate", "disk_usage_rate"}
            and observation.unit == "percent"
            and "get_metric_series" in tools
        ):
            signal = observation.signal
            spec = SignalSpec(
                signal=signal,
                entity_ref=observation.entity_ref,
                data_type="number",
                unit="percent",
                description="bounded node resource utilization",
                independent_group=signal,
                minimum=0,
                maximum=100,
                query=CallToolAction(
                    tool_name="get_metric_series",
                    request=GetMetricSeriesRequest(
                        entity_ref=observation.entity_ref,
                        signal=signal,
                        window=bootstrap.observation_horizon,
                        max_points=min(100, tools["get_metric_series"].max_results),
                    ),
                ),
            )
            specs.setdefault((spec.entity_ref, signal), spec)
            rules[signal] = RuleSpec(
                rule_id=signal,
                signal=signal,
                comparison="gte",
                threshold=80 if signal == "cpu_usage_rate" else 90,
                source="configuration",
                provenance="existing v3 node saturation thresholds",
                aggregation="max",
                description="high utilization within the bounded window",
            )
    return EvidenceRegistry(signals=tuple(specs.values()), rules=tuple(rules.values()))
