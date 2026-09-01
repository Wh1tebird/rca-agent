from __future__ import annotations

from rca_agent.core.v3.models import (
    Candidate,
    Comparison,
    ConditionPolarity,
    EvidenceCondition,
    MechanismFamily,
)


def mechanism_catalog(entity: str) -> tuple[Candidate, ...]:
    """Human-authored seed catalog; candidate selection remains incident-dependent."""

    return (
        _candidate(
            "nodeCpuHigh",
            MechanismFamily.RESOURCE,
            entity,
            "Node CPU saturation can reduce local service capacity.",
            [
                _condition("nodeCpuHigh", "node_cpu_saturation", "node_cpu_peak", "gte", 80),
                _condition(
                    "nodeCpuHigh", "node_cpu_sustained", "node_cpu_high_fraction", "gte", 0.2
                ),
            ],
            ["memory pressure", "traffic-induced CPU response", "high baseline"],
        ),
        _candidate(
            "memoryPressure",
            MechanismFamily.RESOURCE,
            entity,
            "Memory saturation can trigger reclaim, eviction, or service degradation.",
            [_condition("memoryPressure", "memory_saturation", "node_memory_peak", "gte", 90)],
            ["cache growth", "normal resident-set variation"],
        ),
        _candidate(
            "diskIOHigh",
            MechanismFamily.RESOURCE,
            entity,
            "Disk saturation can delay storage-backed request processing.",
            [_condition("diskIOHigh", "disk_saturation", "node_disk_peak", "gte", 90)],
            ["filesystem capacity without active IO", "database-side delay"],
        ),
        _candidate(
            "networkLatency",
            MechanismFamily.NETWORK,
            entity,
            "Transport delay shifts latency into network-facing trace stages.",
            [_condition("networkLatency", "network_delay", "network_stage_ms", "gte", 100)],
            ["slow downstream processing", "queueing"],
        ),
        _candidate(
            "networkLoss",
            MechanismFamily.NETWORK,
            entity,
            "Packet loss produces retries, incomplete traces, and transport errors.",
            [_condition("networkLoss", "network_loss", "network_error_ratio", "gte", 2)],
            ["application 5xx", "rate limiting"],
        ),
        _candidate(
            "httpError5xx",
            MechanismFamily.APPLICATION,
            entity,
            "Application failures elevate operation or service error evidence.",
            [
                _condition(
                    "httpError5xx", "application_error", "service_error_ratio", "gte", 2
                ),
                _condition(
                    "httpError5xx",
                    "application_error_marker",
                    "application_error_marker",
                    "eq",
                    "present",
                ),
            ],
            ["dependency error propagation", "client cancellation"],
        ),
        _candidate(
            "trafficSurge",
            MechanismFamily.APPLICATION,
            entity,
            "Demand surge increases frontend workload and consumes frontend CPU.",
            [
                _condition(
                    "trafficSurge", "traffic_demand", "service_workload_ratio", "gte", 2
                ),
                _condition(
                    "trafficSurge", "traffic_cpu_response", "service_cpu_ratio", "gte", 1.5
                ),
                _condition(
                    "trafficSurge",
                    "traffic_latency_confounder",
                    "service_latency_ratio",
                    "gte",
                    3,
                    polarity=ConditionPolarity.CONTRADICT,
                    strength=4.0,
                ),
            ],
            ["retry storm", "database latency with queued demand", "load-test traffic"],
        ),
        _candidate(
            "rateLimiting",
            MechanismFamily.APPLICATION,
            entity,
            "Explicit throttling produces bounded status or log evidence.",
            [
                _condition(
                    "rateLimiting", "rate_limit_status", "rate_limit_marker", "eq", "present"
                )
            ],
            ["generic 4xx", "load balancer rejection"],
        ),
        _candidate(
            "slowSQL",
            MechanismFamily.DATABASE_DEPENDENCY,
            entity,
            "SQL execution dominates request duration.",
            [
                _condition("slowSQL", "sql_dominance", "sql_duration_share", "gte", 0.5),
                _condition(
                    "slowSQL",
                    "sql_structure",
                    "sql_shape",
                    "eq",
                    "complex_cross_join",
                ),
            ],
            ["database connection delay", "database overload"],
        ),
        _candidate(
            "dbNetworkLatency",
            MechanismFamily.DATABASE_DEPENDENCY,
            entity,
            "Database connection or transport time dominates before SQL execution.",
            [
                _condition(
                    "dbNetworkLatency",
                    "database_connection_delay",
                    "connection_duration_ms",
                    "gte",
                    100,
                )
            ],
            ["pool exhaustion", "slow SQL"],
        ),
        _candidate(
            "dependencyUnavailable",
            MechanismFamily.DATABASE_DEPENDENCY,
            entity,
            "A required downstream dependency becomes unavailable.",
            [
                _condition(
                    "dependencyUnavailable",
                    "dependency_failure",
                    "dependency_error_marker",
                    "eq",
                    "present",
                )
            ],
            ["network partition", "application exception"],
        ),
    )


def _candidate(
    fault_type: str,
    family: MechanismFamily,
    entity: str,
    reason: str,
    conditions: list[EvidenceCondition],
    confounders: list[str],
) -> Candidate:
    candidate_id = fault_type
    return Candidate(
        candidate_id=candidate_id,
        fault_type=fault_type,
        family=family,
        causal_entity=entity,
        proposal_reason=reason,
        positive_evidence_requirements=[
            item.condition_id
            for item in conditions
            if item.polarity == ConditionPolarity.SUPPORT
        ],
        falsification_conditions=[
            f"positive condition {item.condition_id} is observed and does not match"
            for item in conditions
        ],
        confounders=confounders,
        required_capabilities=sorted({item.capability_id for item in conditions}),
        conditions=conditions,
        knowledge_provenance="human-authored v3 seed mechanism catalog",
    )


def _condition(
    candidate: str,
    group: str,
    signal: str,
    comparison: str,
    threshold: float | str,
    *,
    polarity: ConditionPolarity = ConditionPolarity.SUPPORT,
    strength: float = 2.0,
) -> EvidenceCondition:
    capability_by_signal = {
        "node_cpu_peak": "node_resource_metrics",
        "node_cpu_high_fraction": "node_resource_metrics",
        "node_memory_peak": "node_resource_metrics",
        "node_disk_peak": "node_resource_metrics",
        "network_stage_ms": "trace_stage_decomposition",
        "network_error_ratio": "network_metrics",
        "service_error_ratio": "service_metrics",
        "application_error_marker": "bounded_log_markers",
        "service_workload_ratio": "service_metrics",
        "service_cpu_ratio": "service_metrics",
        "service_latency_ratio": "service_metrics",
        "rate_limit_marker": "bounded_log_markers",
        "sql_duration_share": "trace_stage_decomposition",
        "sql_shape": "trace_stage_decomposition",
        "connection_duration_ms": "trace_stage_decomposition",
        "dependency_error_marker": "bounded_log_markers",
    }
    return EvidenceCondition(
        condition_id=f"{candidate}-{signal}",
        candidate_id=candidate,
        polarity=polarity,
        signal=signal,
        comparison=Comparison(comparison),
        threshold=threshold,
        capability_id=capability_by_signal[signal],
        independent_group=group,
        strength=strength,
        reason=f"Positive {signal} evidence is owned by {candidate}.",
    )
