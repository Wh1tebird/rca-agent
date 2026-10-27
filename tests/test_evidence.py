from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from rca_agent.environment.rca100 import Rca100Environment
from rca_agent.evidence import (
    EvidenceCapability,
    EvidenceQuery,
    EvidenceResult,
    LogEvidenceProvider,
    MetricEvidenceProvider,
    TopologyEvidenceProvider,
    TraceEvidenceProvider,
    UnavailableEvidenceProvider,
)
from rca_agent.models import (
    CallToolAction,
    DataStatus,
    GetMetricSeriesRequest,
    Modality,
    Observation,
    QueryCost,
    TimeWindow,
    ToolName,
    ToolStatus,
)
from rca_agent.runner import EpisodeRunner

WINDOW = TimeWindow(
    start=datetime(2026, 1, 1, tzinfo=UTC),
    end=datetime(2026, 1, 1, 0, 5, tzinfo=UTC),
)


def observation(modality: Modality, signal: str = "request_count") -> Observation:
    return Observation(
        observation_id="obs-" + "a" * 16,
        modality=modality,
        entity_ref="ent-service",
        signal=signal,
        value=None,
        data_status=DataStatus.UNKNOWN,
    )


def test_typed_query_is_bounded_and_opaque() -> None:
    query = EvidenceQuery(
        operation="metric_series",
        modality=Modality.METRIC,
        entity_ref="ent-service",
        signal="request_count",
        window=WINDOW,
        limit=5,
    )
    assert query.query_ref.startswith("qry-")
    assert len(query.query_ref) == 20
    assert "sql" not in query.model_dump_json().lower()
    with pytest.raises(ValidationError):
        EvidenceQuery(
            operation="metric_series",
            modality=Modality.TRACE,
            entity_ref="ent-service",
            signal="request_count",
            window=WINDOW,
        )
    with pytest.raises(ValidationError):
        EvidenceQuery(
            operation="metric_series",
            modality=Modality.METRIC,
            entity_ref="ent-service",
            signal="request_count",
            window=WINDOW,
            limit=101,
        )


@pytest.mark.parametrize(
    ("provider", "operation", "modality", "signal"),
    [
        (MetricEvidenceProvider([observation(Modality.METRIC)]), "metric_series", Modality.METRIC, "request_count"),
        (LogEvidenceProvider([observation(Modality.LOG, "error_marker")]), "log_markers", Modality.LOG, "error_marker"),
        (TraceEvidenceProvider([observation(Modality.TRACE, "trace_duration")]), "trace_summary", Modality.TRACE, None),
        (TopologyEvidenceProvider([observation(Modality.TOPOLOGY, "topology_edge")]), "topology", Modality.TOPOLOGY, None),
    ],
)
def test_deterministic_providers_return_neutral_bounded_observations(
    provider, operation: str, modality: Modality, signal: str | None,
) -> None:
    query = EvidenceQuery(
        operation=operation,
        modality=modality,
        entity_ref="ent-service",
        signal=signal,
        window=WINDOW if operation in {"metric_series", "log_markers", "trace_summary"} else None,
        limit=1,
    )
    result = provider.execute(query, budget_remaining=2)
    assert result.status is ToolStatus.SUCCESS
    assert result.cost.budget_after == 1
    assert result.query_ref == query.query_ref
    assert all("candidate_id" not in item.details for item in result.observations)


def test_missing_data_is_preserved_and_budget_is_enforced() -> None:
    provider = MetricEvidenceProvider([observation(Modality.METRIC)])
    query = EvidenceQuery(
        operation="metric_series", modality=Modality.METRIC,
        entity_ref="ent-service", signal="request_count", window=WINDOW,
    )
    result = provider.execute(query, budget_remaining=0)
    assert result.status is ToolStatus.BUDGET_EXHAUSTED
    assert result.observations == ()
    available = provider.execute(query, budget_remaining=1)
    assert available.observations[0].data_status is DataStatus.UNKNOWN


def test_observation_neutrality_rejects_candidate_interpretation() -> None:
    with pytest.raises(ValidationError, match="interpretation"):
        EvidenceResult(
            query_ref="qry-" + "b" * 16,
            status=ToolStatus.SUCCESS,
            observations=(observation(Modality.METRIC).model_copy(
                update={"details": {"candidate_id": "h-1"}}
            ),),
            cost=QueryCost(units=1, budget_before=2, budget_after=1),
            message="fixture",
        )


def test_cmdb_and_database_are_explicit_unavailable_extensions() -> None:
    provider = UnavailableEvidenceProvider("database")
    query = EvidenceQuery(
        operation="database", modality=Modality.METRIC,
        entity_ref="ent-service",
    )
    result = provider.execute(query, budget_remaining=3)
    assert result.status is ToolStatus.CAPABILITY_UNAVAILABLE
    assert result.observations == ()


def test_capability_permission_and_bounds_are_host_enforced() -> None:
    capability = EvidenceCapability(
        capability_ref="evcap-" + "d" * 16,
        modality=Modality.METRIC,
        operation="metric_series",
        max_results=1,
        required_permission="metrics.read",
    )
    query = EvidenceQuery(
        operation="metric_series", modality=Modality.METRIC,
        entity_ref="ent-service", signal="request_count", window=WINDOW, limit=2,
    )
    denied = MetricEvidenceProvider([observation(Modality.METRIC)])
    denied._capability = capability  # deterministic host fixture setup
    assert denied.execute(query, budget_remaining=2).status is ToolStatus.REJECTED
    allowed = type(denied)([observation(Modality.METRIC)])
    allowed._capability = capability
    allowed._granted_permissions = frozenset({"metrics.read"})
    assert allowed.execute(query, budget_remaining=2).status is ToolStatus.REJECTED
    bounded = query.model_copy(update={"limit": 1})
    assert allowed.execute(bounded, budget_remaining=2).status is ToolStatus.SUCCESS


def test_runner_records_host_owned_capability_metadata(cases_root) -> None:
    runner = EpisodeRunner(Rca100Environment.from_case(cases_root, "t025"))
    bootstrap = runner.start()
    action = CallToolAction(
        tool_name=ToolName.GET_METRIC_SERIES,
        request=GetMetricSeriesRequest(
            entity_ref=bootstrap.alert_entity.entity_ref,
            signal="request_count",
            window=bootstrap.observation_horizon,
            max_points=1,
        ),
    )
    runner.handle(action)
    request = next(event for event in runner.transcript.events
                   if event.event_type.value == "tool_request")
    assert request.payload["evidence_capability"].startswith("evcap-")
    assert request.payload["evidence_provider"] == "environment"
    assert len(request.payload["evidence_scope_digest"]) == 64
    assert bootstrap.alert_entity.entity_ref not in request.payload["evidence_scope_digest"]
    assert request.payload["evidence_time_range"] is not None
    assert request.payload["evidence_cost_units"] == 1
