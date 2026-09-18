"""Initial RCA100 recipes grounded in issued entities/traces and existing rules.

Hosts can replace this builder with their monitoring inventory. No data is queried here.
"""

from rca_agent.core.dynamic import EvidenceRegistry, MechanismClass, RuleSpec, SignalSpec
from rca_agent.environment.metric_catalog import (
    EXTRA_SERVICE_METRICS,
    metric_description,
)
from rca_agent.models import (
    CallToolAction,
    CompareMetricWindowsRequest,
    DataStatus,
    GetMetricSeriesRequest,
    GetTraceStageBreakdownRequest,
    Observation,
    ResetResponse,
    TimeWindow,
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

NON_CAUSAL_SUPPORT_TRACE_RULES = frozenset({
    "operation_duration_ms",
    "sql_duration_ms",
    "db_client_span_duration_ms",
})

METRIC_SIGNAL_CONTRACTS = {
    **{
        signal: {
            "unit": spec[2], "group": "thread_aggregate_context",
            "description": metric_description(signal),
        }
        for signal, spec in EXTRA_SERVICE_METRICS.items()
    },
    "request_count": {
        "unit": "count", "group": "request_volume",
        "description": metric_description("request_count"),
    },
    "thread_pool_max": {
        "unit": None, "group": "thread_pool_capacity_reference",
        "description": "source-provided aggregate maximum-thread statistic",
    },
    "thread_pool_active": {
        "unit": None, "group": "thread_pool_saturation",
        "description": "source-provided aggregate active-thread statistic",
    },
    "thread_pool_queue_depth": {
        "unit": "count", "group": "thread_pool_saturation",
        "description": "aggregate application thread-pool queue depth",
    },
    "thread_pool_rejected_tasks": {
        "unit": "count", "group": "thread_pool_saturation",
        "description": "the aggregate application thread-pool rejected-task signal is positive",
        "comparison": "gte", "threshold": 1, "source": "configuration",
    },
    "jvm_runnable_threads": {
        "unit": "count", "group": "jvm_scheduler_pressure",
        "description": "aggregate runnable JVM thread count",
    },
    "jvm_thread_count": {
        "unit": "count", "group": "jvm_thread_population",
        "description": "aggregate JVM thread population",
    },
    "jvm_thread_cpu_time_ns": {
        "unit": "ns", "group": "jvm_thread_cpu_pressure",
        "description": "aggregate grouped JVM thread CPU time",
    },
    "jvm_heap_used_bytes": {
        "unit": "bytes", "group": "jvm_heap_capacity",
        "description": "aggregate JVM heap usage",
    },
    "jvm_heap_max_bytes": {
        "unit": "bytes", "group": "jvm_heap_capacity_reference",
        "description": "observed aggregate maximum JVM heap",
    },
    "jvm_gc_collection_delta": {
        "unit": "count", "group": "jvm_gc_activity",
        "description": "aggregate JVM garbage-collection count delta",
    },
    "jvm_gc_pause_seconds_delta": {
        "unit": "s", "group": "jvm_gc_activity",
        "description": "aggregate JVM garbage-collection pause-time delta",
    },
}

BASELINE_SIGNAL_CONTRACTS = {
    "request_count_baseline_ratio": {
        "base_signal": "request_count", "group": "request_volume",
        "description": "Same-entity mean request volume at least doubles over equal windows; "
        "does not rule out short bursts or changes of request mix.",
        "threshold": 2.0,
    },
    "request_latency_seconds_baseline_ratio": {
        "base_signal": "request_latency_seconds", "group": "end_to_end_latency",
        "description": "Mean latency at least doubles over equal windows; symptom only.",
        "threshold": 2.0, "support_eligible": False,
    },
    "error_rate_baseline_ratio": {
        "base_signal": "error_rate", "group": "request_errors",
        "description": "Mean error ratio at least doubles over equal windows; symptom only.",
        "threshold": 2.0, "support_eligible": False,
    },
    "thread_pool_active_baseline_ratio": {
        "base_signal": "thread_pool_active", "group": "thread_pool_saturation",
        "description": "active thread-pool fraction increased at least twofold from baseline",
        "threshold": 2.0,
    },
    "thread_pool_queue_depth_baseline_ratio": {
        "base_signal": "thread_pool_queue_depth", "group": "thread_pool_saturation",
        "description": "thread-pool queue depth increased materially from baseline",
        "threshold": 1.5,
    },
    "jvm_runnable_threads_baseline_ratio": {
        "base_signal": "jvm_runnable_threads", "group": "jvm_scheduler_pressure",
        "description": "runnable JVM threads increased materially from baseline",
        "threshold": 2.0,
    },
    "jvm_thread_count_baseline_ratio": {
        "base_signal": "jvm_thread_count", "group": "jvm_thread_population",
        "description": "JVM thread population increased materially from baseline",
        "threshold": 1.2,
    },
    "jvm_thread_cpu_time_ns_baseline_ratio": {
        "base_signal": "jvm_thread_cpu_time_ns", "group": "jvm_thread_cpu_pressure",
        "description": "grouped JVM thread CPU time increased at least fivefold from baseline",
        "threshold": 5.0,
    },
    "jvm_heap_used_bytes_baseline_ratio": {
        "base_signal": "jvm_heap_used_bytes", "group": "jvm_heap_growth",
        "description": "JVM heap usage increased at least forty percent from baseline",
        "threshold": 1.4,
    },
    "jvm_gc_collection_delta_baseline_ratio": {
        "base_signal": "jvm_gc_collection_delta", "group": "jvm_gc_activity",
        "description": "generic garbage-collection frequency increased at least threefold",
        "threshold": 3.0,
    },
    "jvm_gc_pause_seconds_delta_baseline_ratio": {
        "base_signal": "jvm_gc_pause_seconds_delta", "group": "jvm_gc_activity",
        "description": "generic garbage-collection pause time increased at least tenfold",
        "threshold": 10.0,
    },
}

REFUTATION_CLASSES_BY_SIGNAL = {
    "connection_duration_ms": (MechanismClass.DATABASE_CONNECTION_POOL,),
    "connection_acquisition_ms": (MechanismClass.DATABASE_CONNECTION_POOL,),
    "sql_duration_ms": (MechanismClass.DATABASE_EXECUTION,),
    "db_client_span_duration_ms": (MechanismClass.DATABASE_EXECUTION,),
    "sql_duration_share": (MechanismClass.DATABASE_EXECUTION,),
    "thread_pool_active": (MechanismClass.THREAD_POOL_SATURATION,),
    "thread_pool_queue_depth": (MechanismClass.THREAD_POOL_SATURATION,),
    "thread_pool_rejected_tasks": (MechanismClass.THREAD_POOL_SATURATION,),
    "thread_pool_active_baseline_ratio": (MechanismClass.THREAD_POOL_SATURATION,),
    "thread_pool_queue_depth_baseline_ratio": (MechanismClass.THREAD_POOL_SATURATION,),
    "jvm_runnable_threads_baseline_ratio": (MechanismClass.CPU_SATURATION,),
    "jvm_thread_cpu_time_ns_baseline_ratio": (MechanismClass.CPU_SATURATION,),
    "jvm_heap_used_bytes_baseline_ratio": (MechanismClass.MEMORY_PRESSURE,),
    "jvm_gc_collection_delta_baseline_ratio": (MechanismClass.GC_PAUSE,),
    "jvm_gc_pause_seconds_delta_baseline_ratio": (MechanismClass.GC_PAUSE,),
    "cpu_usage_rate": (MechanismClass.CPU_SATURATION,),
    "memory_usage_rate": (MechanismClass.MEMORY_PRESSURE,),
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
                    support_eligible=signal not in NON_CAUSAL_SUPPORT_TRACE_RULES,
                    refutes=REFUTATION_CLASSES_BY_SIGNAL.get(signal, ()),
                )
            rules["sql_duration_share_low"] = RuleSpec(
                rule_id="sql_duration_share_low",
                signal="sql_duration_share",
                comparison="lte",
                threshold=0.2,
                source="configuration",
                provenance="RCA100 trace-stage differential contract",
                description="SQL client span is a small part of operation duration",
                support_eligible=False,
                refutes=(),
            )
    issued_entity_types = {bootstrap.alert_entity.entity_ref: bootstrap.alert_entity.entity_type}
    for observation in observations:
        for value in (
            observation.value
            if observation.signal in {"related_entity", "trace_participant"}
            else None,
            observation.details.get("neighbor"),
            observation.details.get("resolved_entity"),
            observation.details.get("resolved_service"),
        ):
            if isinstance(value, dict) and value.get("entity_ref") and value.get("entity_type"):
                issued_entity_types[value["entity_ref"]] = value["entity_type"]
        if observation.signal in METRIC_SIGNAL_CONTRACTS and observation.entity_ref not in (
            issued_entity_types
        ):
            issued_entity_types[observation.entity_ref] = "apm.service"
        if observation.signal in {"cpu_usage_rate", "memory_usage_rate", "disk_usage_rate"}:
            issued_entity_types[observation.entity_ref] = "k8s.node"
    metric_specs: dict[tuple[str, str], SignalSpec] = {}
    comparison_specs: dict[tuple[str, str], SignalSpec] = {}
    if "get_metric_series" in tools:
        for entity_ref, entity_type in issued_entity_types.items():
            if entity_type == "apm.service":
                contracts = METRIC_SIGNAL_CONTRACTS
            elif entity_type == "k8s.node":
                contracts = {
                    signal: {
                        "unit": "percent", "group": signal,
                        "description": "bounded node resource utilization",
                    }
                    for signal in ("cpu_usage_rate", "memory_usage_rate", "disk_usage_rate")
                }
            else:
                contracts = {}
            # Public descriptors expose neutral metrics for every issued entity type.
            # Only explicit host contracts below create causal validation rules.
            contracts = dict(contracts)
            for descriptor in tools["get_metric_series"].metrics:
                if descriptor.entity_type == entity_type:
                    contracts.setdefault(descriptor.signal, {
                        "unit": descriptor.unit, "group": "context_only",
                        "description": descriptor.description,
                    })
            for signal, contract in contracts.items():
                spec = SignalSpec(
                    signal=signal,
                    entity_ref=entity_ref,
                    data_type="number",
                    unit=contract["unit"],
                    description=contract["description"],
                    independent_group=contract["group"],
                    minimum=0,
                    maximum=100 if signal in {
                        "cpu_usage_rate", "memory_usage_rate", "disk_usage_rate"
                    } else None,
                    query=CallToolAction(
                        tool_name="get_metric_series",
                        request=GetMetricSeriesRequest(
                            entity_ref=entity_ref,
                            signal=signal,
                            window=bootstrap.observation_horizon,
                            max_points=min(100, tools["get_metric_series"].max_results),
                        ),
                    ),
                )
                metric_specs[(entity_ref, signal)] = spec
            if entity_type in {"apm.service", "apm.operation", "apm.instance"} and (
                "compare_metric_windows" in tools
            ):
                horizon = bootstrap.observation_horizon
                incident_start = horizon.start + (horizon.end - horizon.start) / 2
                incident_window = TimeWindow(start=incident_start, end=horizon.end)
                for signal, contract in BASELINE_SIGNAL_CONTRACTS.items():
                    if contract["base_signal"] not in contracts:
                        continue
                    spec = SignalSpec(
                        signal=signal,
                        entity_ref=entity_ref,
                        data_type="number",
                        unit="ratio",
                        description=contract["description"],
                        independent_group=contract["group"],
                        minimum=0,
                        query=CallToolAction(
                            tool_name="compare_metric_windows",
                            request=CompareMetricWindowsRequest(
                                entity_ref=entity_ref,
                                signal=contract["base_signal"],
                                incident_window=incident_window,
                            ),
                        ),
                    )
                    comparison_specs[(entity_ref, signal)] = spec
    specs.update(metric_specs)
    specs.update(comparison_specs)
    for signal in ("cpu_usage_rate", "memory_usage_rate", "disk_usage_rate"):
        if any(spec.signal == signal for spec in metric_specs.values()):
            rules[signal] = RuleSpec(
                rule_id=signal,
                signal=signal,
                comparison="gte",
                threshold=80 if signal == "cpu_usage_rate" else 90,
                source="configuration",
                provenance="existing v3 node saturation thresholds",
                aggregation="max",
                description="high utilization within the bounded window",
                refutes=REFUTATION_CLASSES_BY_SIGNAL.get(signal, ()),
            )
    for signal, contract in METRIC_SIGNAL_CONTRACTS.items():
        if "comparison" not in contract:
            continue
        targets = [spec for spec in metric_specs.values() if spec.signal == signal]
        reference_signal = contract.get("reference_signal")
        if not targets or (
            reference_signal is not None
            and any((spec.entity_ref, reference_signal) not in metric_specs for spec in targets)
        ):
            continue
        rules[signal] = RuleSpec(
            rule_id=signal,
            signal=signal,
            comparison=contract["comparison"],
            threshold=contract.get("threshold"),
            reference_signal=reference_signal,
            factor=contract.get("factor", 1),
            source=contract["source"],
            provenance="RCA100 aggregate JVM metric contract v1",
            aggregation="max",
            description=contract["description"],
            refutes=REFUTATION_CLASSES_BY_SIGNAL.get(signal, ()),
        )
    for signal, contract in BASELINE_SIGNAL_CONTRACTS.items():
        if any(spec.signal == signal for spec in comparison_specs.values()):
            rules[signal] = RuleSpec(
                rule_id=signal,
                signal=signal,
                comparison="gte",
                threshold=contract["threshold"],
                source="configuration",
                provenance="RCA100 equal-window baseline-change contract v1",
                description=contract["description"],
                support_eligible=contract.get("support_eligible", True),
                refutes=REFUTATION_CLASSES_BY_SIGNAL.get(signal, ()),
            )
    return EvidenceRegistry(signals=tuple(specs.values()), rules=tuple(rules.values()))
