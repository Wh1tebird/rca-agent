from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from rca_agent.environment.rca100 import Rca100Environment
from rca_agent.models import (
    CallToolAction,
    GetMetricSeriesRequest,
    RequestStopAction,
    TimeWindow,
    ToolName,
)
from rca_agent.runner import (
    EpisodeRecovery,
    EpisodeRunner,
    OperationalMetricName,
    OperationalMetrics,
    RecoveryState,
    RecoveryValidationError,
    audit_transcript,
)
from rca_agent.runner.component_worker import ComponentWorkerHost, ComponentWorkerSpec
from rca_agent.runner.transcript import EventType
from rca_agent.tools.fingerprint import query_fingerprint


def _runner(cases_root: Path, metrics: OperationalMetrics | None = None) -> EpisodeRunner:
    return EpisodeRunner(
        Rca100Environment.from_case(cases_root, "t025"),
        operational_metrics=metrics,
    )


def _pending_query(runner: EpisodeRunner) -> None:
    assert runner.bootstrap is not None
    request = GetMetricSeriesRequest(
        entity_ref=runner.bootstrap.alert_entity.entity_ref,
        signal="request_latency_seconds",
        window=TimeWindow(
            start=datetime(2024, 1, 1, tzinfo=UTC),
            end=datetime(2024, 1, 1, 0, 1, tzinfo=UTC),
        ),
    )
    action = CallToolAction(tool_name=ToolName.GET_METRIC_SERIES, request=request)
    runner.transcript.append(EventType.TOOL_REQUEST, {
        "tool_name": action.tool_name.value,
        "request": request.model_dump(mode="json"),
        "query_ref": query_fingerprint(action.tool_name, request),
        "budget_before": runner.bootstrap.query_budget.remaining,
    })


def test_recovery_of_frozen_episode_keeps_all_integrity_bindings(cases_root: Path) -> None:
    runner = _runner(cases_root)
    runner.start()
    answer = runner.handle(RequestStopAction(summary="freeze for recovery"))

    snapshot = EpisodeRecovery.inspect(runner.transcript.events)

    assert snapshot.state is RecoveryState.FROZEN
    assert snapshot.episode_id == runner.episode_id
    assert snapshot.next_sequence == len(runner.transcript.events) + 1
    assert snapshot.head_digest == runner.transcript.head_digest
    assert snapshot.answer_digest == answer.artifact_digest
    assert snapshot.replay_valid is True
    with pytest.raises(RecoveryValidationError, match="episode_is_frozen"):
        EpisodeRecovery.attach(runner)


def test_active_runner_can_be_attached_without_creating_a_second_state_source(
    cases_root: Path,
) -> None:
    runner = _runner(cases_root)
    runner.start()

    snapshot = EpisodeRecovery.attach(runner)

    assert snapshot.state is RecoveryState.RESUMABLE
    assert snapshot.episode_id == runner.episode_id
    assert snapshot.next_sequence == len(runner.transcript.events) + 1


def test_pending_query_is_retryable_without_replaying_a_provider(cases_root: Path) -> None:
    runner = _runner(cases_root)
    runner.start()
    _pending_query(runner)

    snapshot = EpisodeRecovery.inspect(runner.transcript.events)

    assert snapshot.state is RecoveryState.RETRYABLE
    assert len(snapshot.pending_operations) == 1
    assert snapshot.pending_operations[0].kind == "query"
    assert snapshot.pending_operations[0].disposition == "retryable"
    assert snapshot.replay_valid is False


def test_interrupted_handoff_is_classified_without_model_or_provider_calls(cases_root: Path) -> None:
    runner = _runner(cases_root)
    runner.start()
    runner.transcript.append(EventType.DISCOVERY_STARTED, {
        "operation_kind": "handoff",
        "operation_status": "started",
    })

    snapshot = EpisodeRecovery.inspect(runner.transcript.events)

    assert snapshot.state is RecoveryState.RETRYABLE
    assert {item.kind for item in snapshot.pending_operations} == {"handoff"}
    assert snapshot.failure_kind == "pending_handoff"


def test_recovery_rejects_digest_mismatch_and_duplicate_sequence(cases_root: Path) -> None:
    runner = _runner(cases_root)
    runner.start()
    events = runner.transcript.events

    tampered = events[:-1] + (events[-1].model_copy(update={"digest": "f" * 64}),)
    assert EpisodeRecovery.inspect(tampered).state is RecoveryState.CORRUPTED
    assert EpisodeRecovery.inspect(events + (events[-1],)).failure_kind == (
        "duplicate_or_noncontiguous_event"
    )


def test_event_record_roundtrip_is_recoverable_only_with_integrity_fields(cases_root: Path) -> None:
    runner = _runner(cases_root)
    runner.start()
    runner.handle(RequestStopAction(summary="event record recovery"))
    records = tuple(event.as_event_record() for event in runner.transcript.events)

    snapshot = EpisodeRecovery.inspect(records)

    assert snapshot.state is RecoveryState.FROZEN
    assert snapshot.head_digest == runner.transcript.head_digest


def test_audit_projection_is_redacted_and_contains_host_metadata(cases_root: Path) -> None:
    runner = _runner(cases_root)
    runner.start()
    runner.transcript.append(EventType.TOOL_REQUEST, {
        "tool_name": "get_metric_series",
        "request": {"secret": "should-not-be-copied", "path": "/private/case"},
        "query_ref": "qry-0123456789abcdef",
        "evidence_capability": "metrics.request_latency",
        "evidence_provider": "offline_fixture",
        "evidence_scope_digest": "a" * 64,
        "evidence_time_range": {
            "start": "2024-01-01T00:00:00Z",
            "end": "2024-01-01T00:01:00Z",
        },
        "evidence_cost_units": 1,
        "prompt": "do not copy",
    })

    records = audit_transcript(runner.transcript.events)
    serialized = " ".join(record.model_dump_json() for record in records)
    request_record = records[-1]

    assert request_record.capability == "metrics.request_latency"
    assert request_record.provider == "offline_fixture"
    assert request_record.entity_scope_digest == "a" * 64
    assert request_record.cost_units == 1
    assert "should-not-be-copied" not in serialized
    assert "/private/case" not in serialized
    assert "prompt" not in serialized

    frozen_runner = _runner(cases_root)
    frozen_runner.start()
    answer = frozen_runner.handle(RequestStopAction(summary="audit freeze"))
    frozen_records = audit_transcript(
        frozen_runner.transcript.events,
        replay_result="valid",
    )
    frozen_record = frozen_records[-1]
    assert frozen_record.answer_artifact_digest == answer.artifact_digest
    assert frozen_record.stop_reason == answer.termination_reason.value
    assert frozen_record.replay_result == "valid"


def test_metrics_are_observational_and_do_not_change_decision(cases_root: Path) -> None:
    metrics = OperationalMetrics()
    runner = _runner(cases_root, metrics)
    runner.start()
    answer = runner.handle(RequestStopAction(summary="metrics are passive"))

    snapshot = metrics.snapshot()

    assert snapshot.counters[OperationalMetricName.EPISODE_STARTED.value] == 1
    assert snapshot.counters[OperationalMetricName.EPISODE_COMPLETED.value] == 1
    assert answer.authoritative_decision.status.value == "inconclusive"


def test_worker_metrics_cover_ready_and_reap_without_external_calls() -> None:
    metrics = OperationalMetrics()
    worker = ComponentWorkerHost(
        ComponentWorkerSpec(worker_type="fake", factory_id="test.fake"),
        metrics=metrics,
    )
    worker.start()
    worker.request({"audit": "offline"})
    worker.terminate()

    snapshot = metrics.snapshot()

    assert snapshot.counters[OperationalMetricName.WORKER_READY.value] == 1
    assert snapshot.counters[OperationalMetricName.WORKER_REAP.value] >= 1
