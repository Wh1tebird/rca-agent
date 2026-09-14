"""Initial RCA100 recipes grounded in issued entities/traces and existing rules.

Hosts can replace this builder with their monitoring inventory. No data is queried here.
"""

from rca_agent.core.dynamic import EvidenceRegistry, RuleSpec, SignalSpec
from rca_agent.core.v3.discovery import _TRUSTED_EVIDENCE_TEMPLATES
from rca_agent.models import (
    CallToolAction,
    DataStatus,
    GetMetricSeriesRequest,
    GetTraceStageBreakdownRequest,
    Observation,
    ResetResponse,
)


def rca100_evidence_registry(
    bootstrap: ResetResponse,
    observations: tuple[Observation, ...],
) -> EvidenceRegistry:
    """Reuse existing trace predicates; expose only issued trace/entity query recipes."""
    tools = {c.tool_name: c for c in bootstrap.capabilities if c.available}
    specs: dict[tuple[str, str], SignalSpec] = {}
    rules: dict[str, RuleSpec] = {}
    trace_units = {"connection_duration_ms": "ms", "sql_duration_share": "ratio", "sql_shape": None}
    templates = {t.signal: t for t in _TRUSTED_EVIDENCE_TEMPLATES}
    for observation in observations:
        if observation.data_status not in {DataStatus.OBSERVED, DataStatus.DERIVED}:
            continue
        trace_ref = observation.details.get("trace_ref")
        if trace_ref and "get_trace_stage_breakdown" in tools:
            for signal, unit in trace_units.items():
                template = templates[signal]
                spec = SignalSpec(
                    signal=signal,
                    entity_ref=observation.entity_ref,
                    data_type="string" if signal == "sql_shape" else "number",
                    unit=unit,
                    description=signal.replace("_", " "),
                    independent_group=template.independent_group,
                    query=CallToolAction(
                        tool_name="get_trace_stage_breakdown",
                        request=GetTraceStageBreakdownRequest(
                            trace_ref=trace_ref,
                            max_stages=min(12, tools["get_trace_stage_breakdown"].max_results),
                        ),
                    ),
                )
                specs.setdefault((spec.entity_ref, signal), spec)
                descriptions = {
                    "connection_duration_ms": "database connection phase is materially delayed",
                    "sql_duration_share": "SQL execution dominates the operation duration",
                    "sql_shape": "SQL has the known complex cross-join structure",
                }
                rules[signal] = RuleSpec(
                    rule_id=signal,
                    signal=signal,
                    comparison=template.comparison,
                    threshold=template.threshold,
                    source="configuration",
                    provenance="existing v3 trusted evidence template",
                    description=descriptions[signal],
                )
                if signal == "sql_duration_share":
                    rules["sql_duration_share_low"] = RuleSpec(
                        rule_id="sql_duration_share_low",
                        signal=signal,
                        comparison="lte",
                        threshold=0.2,
                        source="configuration",
                        provenance="RCA100 trace-stage differential contract",
                        description="SQL execution is a small part of operation duration",
                    )
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
