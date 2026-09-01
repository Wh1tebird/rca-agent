from rca_agent.core.v3.capabilities import rca100_capabilities
from rca_agent.core.v3.engine import OpenRcaEngine
from rca_agent.core.v3.execution import DictionaryObservationProvider
from rca_agent.core.v3.models import Incident, InvestigationStatus


def _incident(title: str = "frontend latency") -> Incident:
    return Incident(
        incident_id="demo",
        alert_title=title,
        alert_entity="frontend",
        window_start="2026-01-01T00:00:00Z",
        window_end="2026-01-01T00:05:00Z",
    )


def test_rule_engine_confirms_only_after_all_required_positive_conditions():
    provider = DictionaryObservationProvider(
        {
            "node_cpu_peak": 95.0,
            "node_cpu_high_fraction": 0.5,
            "node_memory_peak": 60.0,
            "node_disk_peak": 40.0,
            "service_error_ratio": 1.0,
            "network_stage_ms": 10.0,
            "sql_duration_share": 0.1,
            "connection_duration_ms": 5.0,
        },
        "node-1",
    )

    result = OpenRcaEngine(rca100_capabilities()).investigate(_incident(), provider)

    assert result.status == InvestigationStatus.CONFIRMED_ROOT_CAUSE
    assert result.winner == "nodeCpuHigh"


def test_partial_node_cpu_support_cannot_confirm():
    provider = DictionaryObservationProvider(
        {
            "node_cpu_peak": 95.0,
            "node_cpu_high_fraction": 0.1,
            "node_memory_peak": 60.0,
            "node_disk_peak": 40.0,
            "service_error_ratio": 1.0,
            "network_stage_ms": 10.0,
            "sql_duration_share": 0.1,
            "connection_duration_ms": 5.0,
        }
    )

    result = OpenRcaEngine(rca100_capabilities()).investigate(_incident(), provider)

    assert result.winner is None
    assert "nodeCpuHigh" in result.supported_hypotheses


def test_unavailable_observation_is_missing_not_contradicting():
    result = OpenRcaEngine(rca100_capabilities()).investigate(
        _incident("rate limit error"),
        DictionaryObservationProvider({"service_error_ratio": 1.0}),
    )

    assert result.winner is None
    assert any("rateLimiting-rate_limit_marker" in item for item in result.missing_evidence)
    assert all(
        state.contradict_strength == 0
        for round_item in result.rounds
        for state in round_item.decision.candidate_states
    )


def test_metric_round_precedes_trace_round_and_can_stop_early():
    provider = DictionaryObservationProvider(
        {
            "node_cpu_peak": 99.0,
            "node_cpu_high_fraction": 0.8,
            "node_memory_peak": 50.0,
            "node_disk_peak": 30.0,
            "service_error_ratio": 1.0,
            "network_stage_ms": 500.0,
            "sql_duration_share": 0.8,
            "connection_duration_ms": 500.0,
        }
    )

    result = OpenRcaEngine(rca100_capabilities()).investigate(_incident(), provider)

    assert len(result.rounds) == 1
    assert all(
        plan.query_type != "SlowTraceQuery+TraceByIdQuery"
        for plan in provider.executed_plans
    )


def test_trace_round_runs_when_metric_round_is_inconclusive():
    provider = DictionaryObservationProvider(
        {
            "node_cpu_peak": 30.0,
            "node_cpu_high_fraction": 0.0,
            "node_memory_peak": 50.0,
            "node_disk_peak": 30.0,
            "service_error_ratio": 1.0,
            "network_stage_ms": 5.0,
            "sql_duration_share": 0.8,
            "sql_shape": "complex_cross_join",
            "connection_duration_ms": 5.0,
        }
    )

    result = OpenRcaEngine(rca100_capabilities()).investigate(_incident(), provider)

    assert len(result.rounds) == 2
    assert result.winner == "slowSQL"
    assert result.status == InvestigationStatus.CONFIRMED_ROOT_CAUSE


def test_query_exception_becomes_query_failed_state():
    class FailingProvider(DictionaryObservationProvider):
        def execute(self, plans):
            raise RuntimeError("backend unavailable")

    result = OpenRcaEngine(rca100_capabilities()).investigate(
        _incident(),
        FailingProvider({"node_cpu_peak": 99.0}),
    )

    assert result.status == InvestigationStatus.QUERY_FAILED
    assert result.winner is None
    assert any("RuntimeError" in item for item in result.missing_evidence)


def test_missing_evidence_without_support_is_inconclusive_not_supported():
    result = OpenRcaEngine(rca100_capabilities()).investigate(
        _incident("rate limit error"),
        DictionaryObservationProvider({"service_error_ratio": 1.0}),
    )

    assert result.status == InvestigationStatus.INCONCLUSIVE
    assert not result.supported_hypotheses
