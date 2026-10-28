from __future__ import annotations

from pathlib import Path

from rca_agent.environment.rca100 import Rca100Environment
from rca_agent.investigation import InvestigationWorkspaceReducer
from rca_agent.langgraph_orchestrator import RoutingProjection
from rca_agent.models import (
    AuthoritativeDecision,
    CandidateFrontier,
    CoreAssessmentPreview,
    DecisionSource,
    DecisionStatus,
    EpisodeStatus,
    RequestStopAction,
    StopReason,
)
from rca_agent.runner import EpisodeRunner, EventRecord, RecordedReplay
from rca_agent.runner.transcript import EventType


def _runner(cases_root: Path, *, budget: int = 20) -> EpisodeRunner:
    return EpisodeRunner(
        Rca100Environment.from_case(cases_root, "t025", query_budget=budget),
        investigation_workspace_mode="active",
    )


def test_transcript_event_record_has_stable_identity_and_runner_provenance(cases_root: Path) -> None:
    runner = _runner(cases_root)
    runner.start()
    runner.handle(RequestStopAction(summary="state convergence"))

    records = [event.as_event_record() for event in runner.transcript.events]
    assert all(isinstance(record, EventRecord) for record in records)
    assert len({record.event_id for record in records}) == len(records)
    assert [record.sequence for record in records] == list(range(1, len(records) + 1))
    assert all(record.provenance == "episode_runner" for record in records)
    assert all(record.schema_version == 1 for record in records)


def test_replay_reconstructs_stop_budget_frontier_and_answer_digest(cases_root: Path) -> None:
    runner = _runner(cases_root, budget=1)
    runner.start()
    answer = runner.handle(RequestStopAction(summary="freeze convergence"))
    report = RecordedReplay.verify(runner.transcript.events)

    assert runner.status is EpisodeStatus.ANSWER_FROZEN
    assert answer is runner.answer  # a non-LangGraph driver uses the same Runner authority
    assert answer.termination_reason is StopReason.POLICY_REQUEST
    assert report.stop_reason is answer.termination_reason
    assert report.answer_digest == answer.artifact_digest
    assert report.budget_remaining == 1
    assert report.candidate_assessment_digest is not None
    assert report.candidate_frontier_digest == report.candidate_assessment_digest
    assert report.head_digest == runner.transcript.head_digest


def test_workspace_is_only_the_transcript_derived_projection(cases_root: Path) -> None:
    runner = _runner(cases_root)
    runner.start()
    answer = runner.handle(RequestStopAction(summary="projection only"))
    assert answer.investigation_workspace is not None

    events = runner.transcript.events
    reducer = InvestigationWorkspaceReducer(runner.episode_id)
    for event in events[:-1]:
        reducer.apply(event)
    expected = reducer.workspace.snapshot(
        transcript_head_sequence=events[-2].sequence,
        transcript_head_digest=events[-2].digest,
        registry_digest=answer.investigation_workspace.registry_digest,
    )
    assert answer.investigation_workspace == expected


def test_routing_projection_is_derived_and_cannot_authorize_state_changes() -> None:
    projection = RoutingProjection(
        stage="active",
        needs_continue=True,
        candidate_frontier=CandidateFrontier(),
        core_assessment=CoreAssessmentPreview(overall_decision=AuthoritativeDecision(
            status=DecisionStatus.INCONCLUSIVE,
            source=DecisionSource.CORE,
            reason="no evidence",
        )),
    )
    assert projection.needs_continue is True
    assert projection.stop_signal is None
    assert EventType.ANSWER_FROZEN.value not in projection.model_dump_json()
