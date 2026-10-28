from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from langchain_core.runnables import RunnableLambda
from pydantic import ValidationError

from rca_agent.langchain_policy import (
    LangChainPolicyAdapter,
    PolicyEnvelope,
    ScriptedPolicyRunnable,
)
from rca_agent.langgraph_orchestrator import (
    COMPATIBILITY_CACHE_FIELDS,
    EPHEMERAL_ROUTING_FIELDS,
    ROUTING_PROJECTION_FIELDS,
    InvestigationGraphState,
    LangGraphInvestigation,
    RoutingProjection,
)
from rca_agent.models import (
    AuthoritativeDecision,
    CallToolAction,
    CapabilityDescriptor,
    CoreAssessmentPreview,
    DataStatus,
    DecisionSource,
    DecisionStatus,
    EntityRef,
    EpisodeStatus,
    GetTraceStageBreakdownRequest,
    HypothesisProposal,
    Modality,
    Observation,
    QueryBudgetView,
    QueryCost,
    ResetResponse,
    StopReason,
    SubmitHypothesisAction,
    TimeWindow,
    ToolName,
    ToolResponse,
    ToolStatus,
    UpdateCandidateEvidenceAction,
)
from rca_agent.reasoning import ReasoningRequest, ReasoningResponse
from rca_agent.runner import EpisodeRunner, EventType, RecordedReplay
from rca_agent.tools.fingerprint import query_fingerprint


class FakeTraceEnvironment:
    def __init__(self, budget: int = 4) -> None:
        self.budget = budget
        self.remaining = budget
        self.calls = 0
        self.now = datetime.now(UTC)

    def reset(self) -> ResetResponse:
        return ResetResponse(
            event_time=self.now,
            alert_summary="slow operation",
            alert_entity=EntityRef(
                entity_ref="ent-service",
                display_name="service",
                entity_type="apm.operation",
            ),
            observation_horizon=TimeWindow(
                start=self.now - timedelta(minutes=5),
                end=self.now,
            ),
            capabilities=(
                CapabilityDescriptor(
                    tool_name="get_trace_stage_breakdown",
                    modality=Modality.TRACE,
                    available=True,
                    query_cost=1,
                    max_window_seconds=3600,
                    max_results=8,
                ),
            ),
            query_budget=QueryBudgetView(initial=self.budget, remaining=self.remaining),
        )

    def get_trace_stage_breakdown(
        self, request: GetTraceStageBreakdownRequest
    ) -> ToolResponse:
        before = self.remaining
        self.remaining -= 1
        self.calls += 1
        return ToolResponse(
            tool_name=ToolName.GET_TRACE_STAGE_BREAKDOWN,
            query_ref=query_fingerprint(ToolName.GET_TRACE_STAGE_BREAKDOWN, request),
            status=ToolStatus.SUCCESS,
            observations=(
                Observation(
                    observation_id="obs-0000000000000001",
                    modality=Modality.TRACE,
                    entity_ref="ent-service",
                    signal="sql_duration_share",
                    value=0.8,
                    data_status=DataStatus.DERIVED,
                ),
                Observation(
                    observation_id="obs-0000000000000002",
                    modality=Modality.TRACE,
                    entity_ref="ent-service",
                    signal="sql_shape",
                    value="complex_cross_join",
                    data_status=DataStatus.DERIVED,
                ),
            ),
            cost=QueryCost(units=1, budget_before=before, budget_after=self.remaining),
            message="bounded trace-stage facts returned",
        )


class MultiCandidateTraceEnvironment(FakeTraceEnvironment):
    def get_trace_stage_breakdown(
        self, request: GetTraceStageBreakdownRequest
    ) -> ToolResponse:
        before = self.remaining
        self.remaining -= 1
        self.calls += 1
        return ToolResponse(
            tool_name=ToolName.GET_TRACE_STAGE_BREAKDOWN,
            query_ref=query_fingerprint(ToolName.GET_TRACE_STAGE_BREAKDOWN, request),
            status=ToolStatus.SUCCESS,
            observations=(
                Observation(
                    observation_id="obs-0000000000000001",
                    modality=Modality.TRACE,
                    entity_ref="ent-service",
                    signal="sql_duration_share",
                    value=0.1,
                    data_status=DataStatus.DERIVED,
                ),
                Observation(
                    observation_id="obs-0000000000000002",
                    modality=Modality.TRACE,
                    entity_ref="ent-service",
                    signal="sql_shape",
                    value="simple_select",
                    data_status=DataStatus.DERIVED,
                ),
                Observation(
                    observation_id="obs-0000000000000003",
                    modality=Modality.TRACE,
                    entity_ref="ent-service",
                    signal="connection_duration_ms",
                    value=180.0,
                    data_status=DataStatus.DERIVED,
                ),
            ),
            cost=QueryCost(units=1, budget_before=before, budget_after=self.remaining),
            message="bounded differential trace-stage facts returned",
        )


class RecordingScriptedPolicyRunnable(ScriptedPolicyRunnable):
    def __init__(self, actions):
        super().__init__(actions)
        self.contexts: list[dict] = []

    def invoke(self, input, config=None, **kwargs):
        self.contexts.append(input["context"])
        return super().invoke(input, config, **kwargs)


class RecordingReasoningAdapter:
    def __init__(self, actions):
        self.actions = list(actions)
        self.requests: list[ReasoningRequest] = []

    def invoke(self, request: ReasoningRequest) -> ReasoningResponse:
        self.requests.append(request)
        action = self.actions.pop(0)
        return ReasoningResponse(
            role=request.role,
            operation=request.operation,
            kind="next_action",
            proposal={"action": action.model_dump(mode="json")},
        )


def _tool_action() -> CallToolAction:
    return CallToolAction(
        tool_name=ToolName.GET_TRACE_STAGE_BREAKDOWN,
        request=GetTraceStageBreakdownRequest(trace_ref="trace-0000000000000001"),
    )


def _hypothesis_action() -> SubmitHypothesisAction:
    return SubmitHypothesisAction(
        proposal=HypothesisProposal(
            hypothesis_id="h-slowsql",
            candidate_id="slowSQL",
            mechanism="slowSQL",
            causal_entity_ref="ent-service",
            rationale="SQL dominates the bounded trace and has the trusted structural shape",
            observation_refs=("obs-0000000000000001", "obs-0000000000000002"),
        )
    )


def _register_candidate(candidate_id: str, hypothesis_id: str) -> SubmitHypothesisAction:
    return SubmitHypothesisAction(
        proposal=HypothesisProposal(
            hypothesis_id=hypothesis_id,
            candidate_id=candidate_id,
            mechanism=candidate_id,
            causal_entity_ref="ent-service",
            rationale="keep this candidate active until discriminating trace evidence is acquired",
        )
    )


def test_fake_langchain_policy_drives_graph_to_core_confirmation():
    environment = FakeTraceEnvironment()
    runner = EpisodeRunner(environment)
    policy = LangChainPolicyAdapter(
        ScriptedPolicyRunnable((_tool_action(), _hypothesis_action()))
    )

    report = LangGraphInvestigation(runner, policy).run()

    assert runner.status == EpisodeStatus.ANSWER_FROZEN
    assert environment.calls == 1
    assert report.answer.authoritative_decision.status == DecisionStatus.CONFIRMED_ROOT_CAUSE
    assert report.answer.authoritative_decision.source == DecisionSource.CORE
    assert report.answer.authoritative_decision.winning_hypothesis_id == "h-slowsql"
    assert (
        report.final_preview.overall_decision.status
        == DecisionStatus.CONFIRMED_ROOT_CAUSE
    )
    assert len(report.history) == 2
    assert RecordedReplay.verify(runner.transcript.events).valid


def test_graph_uses_typed_reasoning_facade_and_runner_projection():
    environment = FakeTraceEnvironment()
    runner = EpisodeRunner(environment)
    reasoning = RecordingReasoningAdapter((_tool_action(), _hypothesis_action()))
    compatibility_policy = LangChainPolicyAdapter(ScriptedPolicyRunnable(()))

    report = LangGraphInvestigation(
        runner,
        compatibility_policy,
        reasoning_adapter=reasoning,
    ).run()

    assert len(reasoning.requests) == 2
    assert all(request.role == "inquiry" for request in reasoning.requests)
    assert all(request.operation == "next_action" for request in reasoning.requests)
    assert all(isinstance(request.context, dict) for request in reasoning.requests)
    assert report.routing_projection is not None
    projection = report.routing_projection
    assert projection.stage == runner.status.value
    assert projection.needs_continue is False
    assert projection.stop_signal == runner.status.value
    assert projection.reasoning_request_ref is not None
    assert projection.reasoning_response_ref is not None
    assert report.answer == runner.answer
    with pytest.raises(ValidationError):
        projection.stage = "active"
    assert set(RoutingProjection.model_fields) == ROUTING_PROJECTION_FIELDS
    assert not {"answer", "workspace", "budget", "permission"} & set(
        projection.model_dump(mode="json")
    )


def test_stale_graph_preview_is_overridden_by_runner_projection():
    runner = EpisodeRunner(FakeTraceEnvironment())
    runner.start()
    investigation = LangGraphInvestigation(
        runner,
        LangChainPolicyAdapter(ScriptedPolicyRunnable(())),
    )
    investigation._last_preview = CoreAssessmentPreview(overall_decision=AuthoritativeDecision(
        status=DecisionStatus.CONFIRMED_ROOT_CAUSE,
        source=DecisionSource.CORE,
        winning_hypothesis_id="h-stale",
        reason="stale cache",
    ))

    projection = investigation._routing_projection()

    assert projection.core_assessment.overall_decision.status != DecisionStatus.CONFIRMED_ROOT_CAUSE
    assert projection.needs_continue is True


def test_langgraph_state_marks_legacy_fields_as_cache_only():
    annotations = set(InvestigationGraphState.__annotations__)
    assert COMPATIBILITY_CACHE_FIELDS <= annotations
    assert EPHEMERAL_ROUTING_FIELDS <= annotations
    assert "routing_projection" in annotations
    source = Path(__file__).parents[1].joinpath(
        "src", "rca_agent", "langgraph_orchestrator.py"
    ).read_text()
    assert "from rca_agent.environment" not in source
    assert "from rca_agent.evidence" not in source
    assert ".transcript.append" not in source


def test_core_preview_is_read_only_and_does_not_freeze_or_append_transcript():
    runner = EpisodeRunner(FakeTraceEnvironment())
    runner.start()
    response = runner.handle(_tool_action())
    runner.handle(_hypothesis_action())
    event_count = len(runner.transcript.events)

    first = runner.preview_core_assessment()
    second = runner.preview_core_assessment()

    assert response.status == ToolStatus.SUCCESS
    assert first == second
    assert first.overall_decision.status == DecisionStatus.CONFIRMED_ROOT_CAUSE
    assert runner.status == EpisodeStatus.ACTIVE
    assert runner.answer is None
    assert len(runner.transcript.events) == event_count


def test_multi_candidate_graph_uses_differential_signal_and_scoped_evidence():
    environment = MultiCandidateTraceEnvironment(budget=1)
    runner = EpisodeRunner(environment, max_rounds=5)
    scripted = RecordingScriptedPolicyRunnable(
        (
            _register_candidate("slowSQL", "h-slowsql"),
            _register_candidate("dbNetworkLatency", "h-dbnetwork"),
            _tool_action(),
            UpdateCandidateEvidenceAction(
                hypothesis_id="h-slowsql",
                observation_refs=("obs-0000000000000001", "obs-0000000000000002"),
            ),
            UpdateCandidateEvidenceAction(
                hypothesis_id="h-dbnetwork",
                observation_refs=("obs-0000000000000003",),
            ),
        )
    )

    report = LangGraphInvestigation(
        runner,
        LangChainPolicyAdapter(scripted),
    ).run()

    differential_context = scripted.contexts[2]
    differential_signals = {
        item["signal"] for item in differential_context["core_assessment"]["discriminating_signals"]
    }
    assert differential_signals == {
        "connection_duration_ms",
        "sql_duration_share",
        "sql_shape",
    }
    assert report.history[2]["action"]["tool_name"] == "get_trace_stage_breakdown"
    final_states = {
        item.candidate_id: item.evidence_status
        for item in report.final_preview.candidate_assessments
    }
    assert final_states["slowSQL"].value == "contradicted"
    assert final_states["dbNetworkLatency"].value == "confirmed"
    pre_confirmation_frontier = scripted.contexts[4]["candidate_frontier"]
    assert pre_confirmation_frontier["contradicted_candidate_ids"] == ["slowSQL"]
    assert pre_confirmation_frontier["active_candidate_ids"] == ["dbNetworkLatency"]
    assert report.final_frontier.confirmed_candidate_ids == ("dbNetworkLatency",)
    assert report.final_frontier.contradicted_candidate_ids == ("slowSQL",)
    assert (
        report.answer.authoritative_decision.winning_hypothesis_id
        == "h-dbnetwork"
    )
    assert report.answer.authoritative_decision.source == DecisionSource.CORE
    assert environment.calls == 1
    assert RecordedReplay.verify(runner.transcript.events).valid


def test_runner_max_rounds_stops_graph_before_second_environment_call():
    environment = FakeTraceEnvironment()
    runner = EpisodeRunner(environment, max_rounds=1)
    policy = LangChainPolicyAdapter(ScriptedPolicyRunnable((_tool_action(), _tool_action())))

    report = LangGraphInvestigation(runner, policy).run()

    assert report.answer.termination_reason == StopReason.MAX_ROUNDS
    assert environment.calls == 1
    assert RecordedReplay.verify(runner.transcript.events).valid


def test_invalid_structured_model_output_fails_before_any_tool_execution():
    environment = FakeTraceEnvironment()
    runner = EpisodeRunner(environment)
    invalid = RunnableLambda(
        lambda _: {
            "action": {
                "action_type": "request_stop",
                "reason": "policy_request",
                "summary": "attempted direct decision",
                "decision": "confirmed_root_cause",
            }
        }
    )

    report = LangGraphInvestigation(runner, LangChainPolicyAdapter(invalid)).run()

    assert report.answer.termination_reason == StopReason.POLICY_INVALID_ACTION
    assert environment.calls == 0
    event_types = {item.event_type for item in runner.transcript.events}
    assert EventType.ACTION_REJECTED in event_types
    assert EventType.TOOL_REQUEST not in event_types
    assert RecordedReplay.verify(runner.transcript.events).valid


def test_policy_structured_schema_has_no_decision_or_evidence_authority():
    schema = PolicyEnvelope.model_json_schema()
    property_names: set[str] = set()

    def collect_property_names(value: object) -> None:
        if isinstance(value, dict):
            properties = value.get("properties")
            if isinstance(properties, dict):
                property_names.update(properties)
            for child in value.values():
                collect_property_names(child)
        elif isinstance(value, list):
            for child in value:
                collect_property_names(child)

    collect_property_names(schema)

    assert "decision" not in property_names
    assert "threshold" not in property_names
    assert "evidence_weight" not in property_names


def test_core_preview_schema_exposes_no_condition_rule_or_score():
    schema = CoreAssessmentPreview.model_json_schema()
    property_names: set[str] = set()

    def collect_property_names(value: object) -> None:
        if isinstance(value, dict):
            properties = value.get("properties")
            if isinstance(properties, dict):
                property_names.update(properties)
            for child in value.values():
                collect_property_names(child)
        elif isinstance(value, list):
            for child in value:
                collect_property_names(child)

    collect_property_names(schema)

    assert not {
        "condition_id",
        "comparison",
        "threshold",
        "strength",
        "evidence_weight",
        "winner_margin",
    } & property_names
