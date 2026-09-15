from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from rca_agent.environment.rca100 import Rca100Environment
from rca_agent.models import (
    CompareMetricWindowsRequest,
    DataStatus,
    GetMetricSeriesRequest,
    GetServiceTraceSummaryRequest,
    GetTraceStageBreakdownRequest,
    InspectTopologyRequest,
    ListRelatedEntitiesRequest,
    TimeWindow,
    ToolStatus,
)


def _environment(cases_root: Path, case_selector: str = "t025", budget: int = 20):
    return Rca100Environment.from_case(cases_root, case_selector, query_budget=budget)


def _incident_window(bootstrap) -> TimeWindow:
    duration = (bootstrap.observation_horizon.end - bootstrap.observation_horizon.start) / 2
    return TimeWindow(
        start=bootstrap.observation_horizon.start + duration,
        end=bootstrap.observation_horizon.end,
    )


def test_reset_exposes_only_bootstrap_contract(cases_root: Path):
    bootstrap = _environment(cases_root).reset()

    assert set(bootstrap.model_dump()) == {
        "event_time",
        "alert_summary",
        "alert_entity",
        "observation_horizon",
        "capabilities",
        "query_budget",
    }
    payload = bootstrap.model_dump_json()
    assert "t025" not in payload
    assert str(cases_root) not in payload
    assert "answer_key" not in payload
    assert {item.tool_name for item in bootstrap.capabilities} >= {
        "get_metric_series",
        "compare_metric_windows",
        "get_service_trace_summary",
        "get_trace_stage_breakdown",
    }
    log_capability = next(
        item for item in bootstrap.capabilities if item.tool_name == "search_log_markers"
    )
    assert not log_capability.available
    assert log_capability.unavailable_reason


def test_public_responses_hide_case_path_and_raw_trace_identifier(cases_root: Path):
    environment = _environment(cases_root)
    bootstrap = environment.reset()
    response = environment.get_service_trace_summary(
        GetServiceTraceSummaryRequest(
            entity_ref=bootstrap.alert_entity.entity_ref,
            window=_incident_window(bootstrap),
            max_traces=1,
        )
    )

    payload = response.model_dump_json()
    assert response.status == ToolStatus.SUCCESS
    assert "t025" not in payload
    assert str(cases_root) not in payload
    assert ".parquet" not in payload
    trace_ref = response.observations[0].details["trace_ref"]
    assert trace_ref.startswith("trace-")
    assert len(trace_ref) != 32


def test_rejects_out_of_scope_window_entity_and_return_limit(cases_root: Path):
    environment = _environment(cases_root)
    bootstrap = environment.reset()
    outside = TimeWindow(
        start=bootstrap.observation_horizon.start - timedelta(seconds=1),
        end=bootstrap.observation_horizon.start + timedelta(seconds=1),
    )
    response = environment.get_metric_series(
        GetMetricSeriesRequest(
            entity_ref=bootstrap.alert_entity.entity_ref,
            signal="request_count",
            window=outside,
        )
    )
    assert response.status == ToolStatus.REJECTED
    assert response.cost.units == 0

    response = environment.get_metric_series(
        GetMetricSeriesRequest(
            entity_ref="ent-not-issued",
            signal="request_count",
            window=_incident_window(bootstrap),
        )
    )
    assert response.status == ToolStatus.REJECTED

    with pytest.raises(ValidationError):
        GetMetricSeriesRequest(
            entity_ref=bootstrap.alert_entity.entity_ref,
            signal="request_count",
            window=_incident_window(bootstrap),
            max_points=101,
        )


def test_query_budget_exhaustion_refuses_next_query(cases_root: Path):
    environment = _environment(cases_root, budget=1)
    bootstrap = environment.reset()
    request = GetMetricSeriesRequest(
        entity_ref=bootstrap.alert_entity.entity_ref,
        signal="request_count",
        window=_incident_window(bootstrap),
        max_points=10,
    )

    first = environment.get_metric_series(request)
    second = environment.get_metric_series(request)

    assert first.cost.budget_after == 0
    assert second.status == ToolStatus.BUDGET_EXHAUSTED
    assert second.cost.units == 0


def test_same_query_replays_deterministically(cases_root: Path):
    def run():
        environment = _environment(cases_root)
        bootstrap = environment.reset()
        return environment.compare_metric_windows(
            CompareMetricWindowsRequest(
                entity_ref=bootstrap.alert_entity.entity_ref,
                signal="request_count",
                incident_window=_incident_window(bootstrap),
            )
        )

    response = run()
    assert response == run()
    ratio = next(
        item for item in response.observations if item.signal == "request_count_baseline_ratio"
    )
    assert ratio.unit == "ratio"
    assert ratio.data_status in {DataStatus.DERIVED, DataStatus.UNKNOWN}
    assert {"before_mean", "incident_mean", "incident_max"} <= ratio.details.keys()


def test_missing_metric_remains_missing_without_interpolation(cases_root: Path):
    environment = _environment(cases_root)
    bootstrap = environment.reset()
    start = bootstrap.observation_horizon.start + timedelta(microseconds=1)
    response = environment.get_metric_series(
        GetMetricSeriesRequest(
            entity_ref=bootstrap.alert_entity.entity_ref,
            signal="request_count",
            window=TimeWindow(start=start, end=start + timedelta(microseconds=1)),
        )
    )

    assert response.status == ToolStatus.EMPTY
    assert response.observations[0].data_status == DataStatus.MISSING
    assert response.observations[0].value is None


def test_observation_contract_contains_no_candidate_or_decision_semantics(cases_root: Path):
    environment = _environment(cases_root)
    bootstrap = environment.reset()
    response = environment.get_metric_series(
        GetMetricSeriesRequest(
            entity_ref=bootstrap.alert_entity.entity_ref,
            signal="request_count",
            window=_incident_window(bootstrap),
            max_points=5,
        )
    )
    payload = json.dumps(response.model_dump(mode="json"), ensure_ascii=False).casefold()

    for forbidden in ("candidate", "root_cause", "supports", "contradicts", "evidence_weight"):
        assert forbidden not in payload


def test_list_related_entities_returns_bounded_authorized_entity(cases_root: Path):
    environment = _environment(cases_root)
    bootstrap = environment.reset()

    response = environment.list_related_entities(
        ListRelatedEntitiesRequest(
            entity_ref=bootstrap.alert_entity.entity_ref,
            max_results=5,
        )
    )

    assert response.status == ToolStatus.SUCCESS
    assert response.cost.units == 1
    assert len(response.observations) == 1
    observation = response.observations[0]
    assert observation.signal == "related_entity"
    assert observation.details == {"direction": "parent", "relation": "operation_of"}
    assert observation.value["entity_ref"].startswith("ent-")
    assert observation.data_status == DataStatus.DERIVED


def test_inspect_topology_returns_observed_edges_for_discovered_service(cases_root: Path):
    environment = _environment(cases_root)
    bootstrap = environment.reset()
    related = environment.list_related_entities(
        ListRelatedEntitiesRequest(entity_ref=bootstrap.alert_entity.entity_ref, max_results=5)
    )
    service_ref = related.observations[0].value["entity_ref"]

    response = environment.inspect_topology(
        InspectTopologyRequest(entity_ref=service_ref, max_results=10)
    )

    assert response.status == ToolStatus.SUCCESS
    assert response.cost.units == 1
    assert response.observations
    assert all(item.signal == "topology_edge" for item in response.observations)
    assert all(item.data_status == DataStatus.OBSERVED for item in response.observations)
    assert all(
        item.details["neighbor"]["entity_ref"].startswith("ent-") for item in response.observations
    )


def test_service_exposes_neutral_jvm_and_thread_pool_metrics(cases_root: Path):
    environment = _environment(cases_root, "t007", budget=50)
    bootstrap = environment.reset()
    related = environment.list_related_entities(
        ListRelatedEntitiesRequest(entity_ref=bootstrap.alert_entity.entity_ref, max_results=5)
    )
    queue = [related.observations[0].value]
    visited = set()
    service_ref = None
    while queue and service_ref is None:
        entity = queue.pop(0)
        if entity["entity_ref"] in visited or entity["entity_type"] != "apm.service":
            continue
        visited.add(entity["entity_ref"])
        if entity["display_name"] == "inventory":
            service_ref = entity["entity_ref"]
            break
        topology = environment.inspect_topology(
            InspectTopologyRequest(entity_ref=entity["entity_ref"], max_results=50)
        )
        queue.extend(
            item.details["neighbor"]
            for item in topology.observations
            if item.details["neighbor"]["entity_type"] == "apm.service"
            and item.details["direction"] == "outgoing"
        )
    assert service_ref is not None

    for signal, unit in (
        ("thread_pool_active", None),
        ("thread_pool_max", None),
        ("thread_pool_queue_depth", "count"),
        ("thread_pool_rejected_tasks", "count"),
        ("jvm_runnable_threads", "count"),
        ("jvm_thread_cpu_time_ns", "ns"),
        ("jvm_heap_used_bytes", "bytes"),
        ("jvm_gc_pause_seconds_delta", "s"),
    ):
        response = environment.get_metric_series(
            GetMetricSeriesRequest(
                entity_ref=service_ref,
                signal=signal,
                window=_incident_window(bootstrap),
                max_points=10,
            )
        )
        assert response.status == ToolStatus.SUCCESS
        assert response.observations
        assert all(item.unit == unit for item in response.observations)
        assert all(item.entity_ref == service_ref for item in response.observations)


def test_unqualified_apm_operation_uses_bounded_service_trace_fallback(cases_root: Path):
    environment = _environment(cases_root, "t040")
    bootstrap = environment.reset()
    response = environment.get_service_trace_summary(
        GetServiceTraceSummaryRequest(
            entity_ref=bootstrap.alert_entity.entity_ref,
            window=_incident_window(bootstrap),
            min_duration_ms=100,
            max_traces=2,
        )
    )

    assert response.status in {ToolStatus.SUCCESS, ToolStatus.EMPTY}
    assert response.status != ToolStatus.QUERY_FAILED


@pytest.mark.parametrize(
    ("case_selector", "expected_connection", "expected_share", "expected_shape"),
    [
        (
            "t025",
            pytest.approx(5000.344, abs=0.01),
            pytest.approx(0.200019, abs=1e-6),
            "simple_point_lookup",
        ),
        (
            "t030",
            pytest.approx(0.024785, abs=0.001),
            pytest.approx(0.998102, abs=1e-6),
            "complex_cross_join",
        ),
    ],
)
def test_t025_t030_expose_distinct_neutral_trace_stage_facts(
    cases_root: Path,
    case_selector: str,
    expected_connection,
    expected_share,
    expected_shape: str,
):
    environment = _environment(cases_root, case_selector)
    bootstrap = environment.reset()
    summary = environment.get_service_trace_summary(
        GetServiceTraceSummaryRequest(
            entity_ref=bootstrap.alert_entity.entity_ref,
            window=_incident_window(bootstrap),
            max_traces=1,
        )
    )
    breakdown = environment.get_trace_stage_breakdown(
        GetTraceStageBreakdownRequest(
            trace_ref=summary.observations[0].details["trace_ref"],
            max_stages=8,
        )
    )
    facts = {item.signal: item for item in breakdown.observations}

    assert breakdown.status == ToolStatus.SUCCESS
    assert facts["connection_duration_ms"].value == expected_connection
    assert facts["sql_duration_share"].value == expected_share
    assert facts["sql_shape"].value == expected_shape
    assert facts["connection_duration_ms"].data_status == DataStatus.OBSERVED
    assert facts["sql_duration_share"].data_status == DataStatus.DERIVED
    assert facts["sql_shape"].data_status == DataStatus.DERIVED


def test_t021_trace_breakdown_resolves_the_downstream_database_operation(
    cases_root: Path,
):
    environment = _environment(cases_root, "t021")
    bootstrap = environment.reset()
    summary = environment.get_service_trace_summary(
        GetServiceTraceSummaryRequest(
            entity_ref=bootstrap.alert_entity.entity_ref,
            window=_incident_window(bootstrap),
            min_duration_ms=100,
            max_traces=1,
        )
    )
    breakdown = environment.get_trace_stage_breakdown(
        GetTraceStageBreakdownRequest(
            trace_ref=summary.observations[0].details["trace_ref"],
            max_stages=12,
        )
    )

    facts = {item.signal: item for item in breakdown.observations}
    assert breakdown.status == ToolStatus.SUCCESS
    assert facts["sql_duration_ms"].entity_ref != bootstrap.alert_entity.entity_ref
    assert facts["connection_acquisition_ms"].value == facts["connection_duration_ms"].value
    assert facts["db_client_span_duration_ms"].value == facts["sql_duration_ms"].value
    assert all(
        item.details["resolution"] == "downstream_database_operation"
        for item in breakdown.observations
    )
    resolved_service = facts["sql_duration_ms"].details["resolved_service"]
    assert resolved_service["entity_type"] == "apm.service"
    assert resolved_service["entity_ref"].startswith("ent-")


def test_non_database_trace_still_issues_ranked_service_participants(cases_root: Path):
    environment = _environment(cases_root, "t007")
    bootstrap = environment.reset()
    summary = environment.get_service_trace_summary(
        GetServiceTraceSummaryRequest(
            entity_ref=bootstrap.alert_entity.entity_ref,
            window=_incident_window(bootstrap),
            min_duration_ms=100,
            max_traces=1,
        )
    )
    breakdown = environment.get_trace_stage_breakdown(
        GetTraceStageBreakdownRequest(
            trace_ref=summary.observations[0].details["trace_ref"],
            max_stages=12,
        )
    )

    participants = [
        item for item in breakdown.observations if item.signal == "trace_participant"
    ]
    assert participants
    assert participants[0].value["display_name"] == "inventory"
    assert participants[0].details["critical_rank"] == 0
    assert breakdown.status == ToolStatus.EMPTY
