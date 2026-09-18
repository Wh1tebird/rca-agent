from datetime import timedelta

import pytest

from rca_agent.core.runtime_registry import rca100_evidence_registry
from rca_agent.environment.rca100 import Rca100Environment
from rca_agent.models import (
    CompareMetricWindowsRequest,
    DataStatus,
    EntityRef,
    GetMetricSeriesRequest,
    GetServiceTraceSummaryRequest,
    GetTraceStageBreakdownRequest,
    TimeWindow,
    ToolStatus,
)


def test_metric_disclosure_is_neutral_and_source_free(cases_root):
    bootstrap = Rca100Environment.from_case(cases_root, "t007").reset()
    metric = next(c for c in bootstrap.capabilities if c.tool_name == "get_metric_series")
    assert {m.entity_type for m in metric.metrics} >= {"apm.instance", "k8s.pod"}
    assert "db_network_rtt_ms" in metric.unavailable_signals
    payload = metric.model_dump_json()
    for secret in ("arms_", ".parquet", "threshold", "t007", str(cases_root)):
        assert secret not in payload


@pytest.mark.parametrize("signal", [
    "jvm_waiting_threads", "jvm_timed_waiting_threads", "thread_pool_core_size",
    "thread_pool_current_size", "thread_pool_max_size", "thread_pool_completed_tasks",
    "thread_pool_scheduled_tasks",
])
def test_new_service_metrics_are_observed_without_fabricated_units(cases_root, signal):
    env = Rca100Environment.from_case(cases_root, "t007")
    bootstrap = env.reset()
    summary = env.get_service_trace_summary(GetServiceTraceSummaryRequest(
        entity_ref=bootstrap.alert_entity.entity_ref,
        window=bootstrap.observation_horizon, max_traces=1,
    ))
    related = env.get_trace_stage_breakdown(GetTraceStageBreakdownRequest(
        trace_ref=summary.observations[0].details["trace_ref"], max_stages=12,
    ))
    service = next(o.value for o in related.observations
                   if isinstance(o.value, dict) and o.value.get("entity_type") == "apm.service")
    result = env.get_metric_series(GetMetricSeriesRequest(
        entity_ref=service["entity_ref"], signal=signal,
        window=bootstrap.observation_horizon,
    ))
    assert result.status == ToolStatus.SUCCESS
    assert result.observations
    assert all(o.data_status == DataStatus.OBSERVED and o.unit is None
               for o in result.observations)
    registry = rca100_evidence_registry(bootstrap, related.observations)
    assert any(s.signal == signal for s in registry.signals)
    assert not any(r.signal == signal for r in registry.rules)


@pytest.mark.parametrize("entity_type,signal", [
    ("apm.instance", "error_rate_baseline_ratio"),
    ("apm.operation", "request_count_baseline_ratio"),
    ("k8s.deployment", "cpu_usage_vs_limits"),
    ("k8s.pod", "pod_restart_count"),
])
def test_registry_preserves_entity_types_and_context_only_rules(cases_root, entity_type, signal):
    bootstrap = Rca100Environment.from_case(cases_root, "t007").reset()
    bootstrap = bootstrap.model_copy(update={"alert_entity": EntityRef(
        entity_ref="ent-target", entity_type=entity_type, display_name="target",
    )})
    registry = rca100_evidence_registry(bootstrap, ())
    spec = next(s for s in registry.signals if s.signal == signal)
    assert spec.entity_ref == "ent-target"
    assert not any(s.signal == "jvm_thread_count" for s in registry.signals)
    rules = [r for r in registry.rules if r.signal == signal]
    if signal == "error_rate_baseline_ratio":
        assert rules and not rules[0].support_eligible and not rules[0].refutes
    elif signal == "request_count_baseline_ratio":
        assert rules and rules[0].support_eligible
        # Window averages alone cannot exclude brief bursts or request-mix changes.
        assert not rules[0].refutes
    else:
        assert not rules


def test_missing_new_metric_is_not_zero_and_unsupported_is_distinct(cases_root):
    env = Rca100Environment.from_case(cases_root, "t007")
    bootstrap = env.reset()
    # Empty bounded interval beyond the last sample, within the accepted horizon.
    end = bootstrap.observation_horizon.end
    window = TimeWindow(start=end - timedelta(microseconds=1), end=end)
    response = env.compare_metric_windows(CompareMetricWindowsRequest(
        entity_ref=bootstrap.alert_entity.entity_ref, signal="error_rate",
        incident_window=window,
    ))
    assert response.status == ToolStatus.COVERAGE_INSUFFICIENT
    assert all(o.value is None and o.data_status == DataStatus.MISSING
               for o in response.observations)
    unsupported = env.get_metric_series(GetMetricSeriesRequest(
        entity_ref=bootstrap.alert_entity.entity_ref, signal="db_network_rtt_ms",
        window=bootstrap.observation_horizon,
    ))
    assert unsupported.status == ToolStatus.REJECTED
