from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from rca_agent.environment.rca100 import Rca100Environment
from rca_agent.models import (
    AuthoritativeDecision,
    CallToolAction,
    DecisionSource,
    DecisionStatus,
    EpisodeStatus,
    GetMetricSeriesRequest,
    HypothesisProposal,
    RequestStopAction,
    StopReason,
    SubmitHypothesisAction,
    TimeWindow,
    ToolName,
    parse_policy_action,
)
from rca_agent.runner import EpisodeRunner, EpisodeStateError, EventType, RecordedReplay


def _environment(cases_root: Path, budget: int = 20):
    return Rca100Environment.from_case(cases_root, "t025", query_budget=budget)


def _incident_window(bootstrap) -> TimeWindow:
    duration = (bootstrap.observation_horizon.end - bootstrap.observation_horizon.start) / 2
    return TimeWindow(
        start=bootstrap.observation_horizon.start + duration,
        end=bootstrap.observation_horizon.end,
    )


def _metric_action(bootstrap, max_points: int = 5) -> CallToolAction:
    return CallToolAction(
        tool_name=ToolName.GET_METRIC_SERIES,
        request=GetMetricSeriesRequest(
            entity_ref=bootstrap.alert_entity.entity_ref,
            signal="request_count",
            window=_incident_window(bootstrap),
            max_points=max_points,
        ),
    )


def test_policy_action_schema_is_closed_and_tool_specific():
    with pytest.raises(ValidationError):
        parse_policy_action(
            {
                "action_type": "call_tool",
                "tool_name": "run_sql",
                "request": {"sql": "select 1"},
            }
        )
    with pytest.raises(ValidationError):
        parse_policy_action(
            {
                "action_type": "call_tool",
                "tool_name": "get_metric_series",
                "request": {"path": "/tmp/data", "threshold": 2.0},
            }
        )
    with pytest.raises(ValidationError):
        parse_policy_action(
            {
                "action_type": "submit_hypothesis",
                "proposal": {
                    "hypothesis_id": "h-x",
                    "mechanism": "synthetic mechanism",
                    "causal_entity_ref": "ent-alert",
                    "rationale": "test",
                    "decision": "confirmed_root_cause",
                },
            }
        )


def test_non_core_decision_cannot_confirm_root_cause():
    with pytest.raises(ValidationError, match="requires domain-core authority"):
        AuthoritativeDecision(
            status=DecisionStatus.CONFIRMED_ROOT_CAUSE,
            source=DecisionSource.PHASE2_LIFECYCLE_DEFAULT,
            winning_hypothesis_id="h-x",
            reason="not authoritative",
        )


def test_tool_call_records_request_response_cost_and_budget(cases_root: Path, tmp_path: Path):
    runner = EpisodeRunner(
        _environment(cases_root),
        transcript_path=tmp_path / "episode.jsonl",
    )
    bootstrap = runner.start()

    response = runner.handle(_metric_action(bootstrap))

    assert runner.status == EpisodeStatus.ACTIVE
    assert response.cost.budget_before == 20
    assert response.cost.budget_after == 19
    events = runner.transcript.events
    request_event = next(item for item in events if item.event_type == EventType.TOOL_REQUEST)
    response_event = next(item for item in events if item.event_type == EventType.TOOL_RESPONSE)
    cost_event = next(item for item in events if item.event_type == EventType.QUERY_COST)
    assert request_event.payload["query_ref"] == response_event.payload["query_ref"]
    assert cost_event.payload == {
        "query_ref": response.query_ref,
        **response.cost.model_dump(mode="json"),
    }
    assert (tmp_path / "episode.jsonl").read_text().count("\n") == len(events)


def test_max_rounds_freezes_without_executing_another_tool(cases_root: Path):
    runner = EpisodeRunner(_environment(cases_root), max_rounds=1)
    bootstrap = runner.start()
    runner.handle(_metric_action(bootstrap))

    answer = runner.handle(_metric_action(bootstrap))

    assert runner.rounds == 1
    assert runner.status == EpisodeStatus.ANSWER_FROZEN
    assert answer.termination_reason == StopReason.MAX_ROUNDS
    assert sum(item.event_type == EventType.TOOL_REQUEST for item in runner.transcript.events) == 1


def test_budget_exhaustion_freezes_and_blocks_more_queries(cases_root: Path):
    runner = EpisodeRunner(_environment(cases_root, budget=1), max_rounds=5)
    bootstrap = runner.start()
    runner.handle(_metric_action(bootstrap))

    answer = runner.handle(_metric_action(bootstrap))

    assert answer.termination_reason == StopReason.BUDGET_EXHAUSTED
    assert runner.status == EpisodeStatus.ANSWER_FROZEN
    with pytest.raises(EpisodeStateError):
        runner.handle(_metric_action(bootstrap))


def test_answer_freeze_is_single_use_and_blocks_mutation(cases_root: Path):
    runner = EpisodeRunner(_environment(cases_root))
    runner.start()
    answer = runner.handle(RequestStopAction(summary="no more bounded questions"))

    assert answer.verify_digest()
    assert answer.authoritative_decision.status == DecisionStatus.INCONCLUSIVE
    assert answer.authoritative_decision.source == DecisionSource.CORE
    with pytest.raises(EpisodeStateError):
        runner.freeze_answer(StopReason.POLICY_REQUEST)
    with pytest.raises(EpisodeStateError):
        runner.start()


def test_hypothesis_is_proposal_and_uses_only_issued_observations(cases_root: Path):
    runner = EpisodeRunner(_environment(cases_root))
    bootstrap = runner.start()
    response = runner.handle(_metric_action(bootstrap))
    observation_ref = response.observations[0].observation_id
    proposal = HypothesisProposal(
        hypothesis_id="h-latency",
        mechanism="bounded test mechanism",
        causal_entity_ref=bootstrap.alert_entity.entity_ref,
        rationale="one issued observation motivates another investigation step",
        observation_refs=(observation_ref,),
        missing_evidence=("independent direct evidence",),
    )
    runner.handle(SubmitHypothesisAction(proposal=proposal))
    answer = runner.handle(RequestStopAction(summary="freeze the phase 2 lifecycle artifact"))

    assert answer.policy_proposals == (proposal,)
    assert answer.observation_refs == (observation_ref,)
    assert answer.missing_evidence == ("independent direct evidence",)
    assert answer.authoritative_decision.status == DecisionStatus.OUT_OF_SCOPE
    assert answer.authoritative_decision.source == DecisionSource.CORE


def test_illegal_lifecycle_transitions_are_rejected(cases_root: Path):
    runner = EpisodeRunner(_environment(cases_root))
    with pytest.raises(EpisodeStateError):
        runner.freeze_answer(StopReason.POLICY_REQUEST)
    with pytest.raises(EpisodeStateError):
        runner.begin_evaluation()


def test_out_of_horizon_action_is_recorded_as_rejected_tool_response(cases_root: Path):
    runner = EpisodeRunner(_environment(cases_root))
    bootstrap = runner.start()
    outside = TimeWindow(
        start=bootstrap.observation_horizon.start - timedelta(seconds=2),
        end=bootstrap.observation_horizon.start - timedelta(seconds=1),
    )
    response = runner.handle(
        CallToolAction(
            tool_name=ToolName.GET_METRIC_SERIES,
            request=GetMetricSeriesRequest(
                entity_ref=bootstrap.alert_entity.entity_ref,
                signal="request_count",
                window=outside,
            ),
        )
    )

    assert response.status.value == "rejected"
    assert any(item.event_type == EventType.QUERY_COST for item in runner.transcript.events)


def test_unexpected_tool_exception_is_recorded_and_safely_frozen(cases_root: Path):
    class FailingEnvironment:
        def __init__(self):
            self.base = _environment(cases_root)

        def reset(self):
            return self.base.reset()

        def get_metric_series(self, request):
            del request
            raise RuntimeError("private backend detail")

    runner = EpisodeRunner(FailingEnvironment())
    bootstrap = runner.start()

    answer = runner.handle(_metric_action(bootstrap))

    assert answer.termination_reason == StopReason.TOOL_FAILURE
    assert runner.status == EpisodeStatus.ANSWER_FROZEN
    assert EventType.TOOL_EXCEPTION in {item.event_type for item in runner.transcript.events}
    transcript = " ".join(str(item.payload) for item in runner.transcript.events)
    assert "private backend detail" not in transcript
    assert RecordedReplay.verify(runner.transcript.events).valid
