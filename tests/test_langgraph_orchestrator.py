from datetime import UTC, datetime, timedelta

from langchain_core.runnables import RunnableLambda

from rca_agent.langchain_policy import (
    LangChainPolicyAdapter,
    PolicyEnvelope,
    ScriptedPolicyRunnable,
)
from rca_agent.langgraph_orchestrator import LangGraphInvestigation
from rca_agent.models import (
    CallToolAction,
    CapabilityDescriptor,
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
)
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
    assert report.final_preview.status == DecisionStatus.CONFIRMED_ROOT_CAUSE
    assert len(report.history) == 2
    assert RecordedReplay.verify(runner.transcript.events).valid


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
    assert first.status == DecisionStatus.CONFIRMED_ROOT_CAUSE
    assert runner.status == EpisodeStatus.ACTIVE
    assert runner.answer is None
    assert len(runner.transcript.events) == event_count


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
