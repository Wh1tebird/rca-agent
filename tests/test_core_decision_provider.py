from datetime import UTC, datetime, timedelta

from rca_agent.models import (
    CallToolAction,
    CapabilityDescriptor,
    DataStatus,
    DecisionSource,
    DecisionStatus,
    EntityRef,
    GetTraceStageBreakdownRequest,
    HypothesisProposal,
    Modality,
    Observation,
    QueryBudgetView,
    QueryCost,
    RequestStopAction,
    ResetResponse,
    StopReason,
    SubmitHypothesisAction,
    TimeWindow,
    ToolName,
    ToolResponse,
    ToolStatus,
)
from rca_agent.runner import CoreDecisionProvider, EpisodeRunner, IssuedObservation
from rca_agent.tools.fingerprint import query_fingerprint


def _proposal(candidate_id: str) -> HypothesisProposal:
    return HypothesisProposal(
        hypothesis_id=f"h-{candidate_id.casefold()}",
        candidate_id=candidate_id,
        mechanism=candidate_id,
        causal_entity_ref="ent-service",
        rationale="bounded evidence should be assessed by the domain core",
        observation_refs=("obs-0000000000000001", "obs-0000000000000002"),
    )


def _record(
    observation_id: str,
    signal: str,
    value,
    status: DataStatus = DataStatus.DERIVED,
) -> IssuedObservation:
    return IssuedObservation(
        observation=Observation(
            observation_id=observation_id,
            modality=Modality.TRACE,
            entity_ref="ent-service",
            signal=signal,
            value=value,
            data_status=status,
        ),
        query_ref=f"qry-{observation_id.removeprefix('obs-')}",
    )


def test_complete_candidate_owned_trace_evidence_confirms():
    decision = CoreDecisionProvider().decide(
        (_proposal("slowSQL"),),
        (
            _record("obs-0000000000000001", "sql_duration_share", 0.8),
            _record("obs-0000000000000002", "sql_shape", "complex_cross_join"),
        ),
        StopReason.POLICY_REQUEST,
    )

    assert decision.status == DecisionStatus.CONFIRMED_ROOT_CAUSE
    assert decision.source == DecisionSource.CORE
    assert decision.winning_hypothesis_id == "h-slowsql"


def test_partial_positive_evidence_is_supported_but_not_confirmed():
    decision = CoreDecisionProvider().decide(
        (_proposal("slowSQL"),),
        (_record("obs-0000000000000001", "sql_duration_share", 0.8),),
        StopReason.POLICY_REQUEST,
    )

    assert decision.status == DecisionStatus.SUPPORTED_HYPOTHESIS
    assert decision.winning_hypothesis_id is None


def test_model_inferred_values_cannot_become_confirmation_evidence():
    decision = CoreDecisionProvider().decide(
        (_proposal("slowSQL"),),
        (
            _record(
                "obs-0000000000000001",
                "sql_duration_share",
                0.8,
                DataStatus.MODEL_INFERRED,
            ),
            _record(
                "obs-0000000000000002",
                "sql_shape",
                "complex_cross_join",
                DataStatus.MODEL_INFERRED,
            ),
        ),
        StopReason.POLICY_REQUEST,
    )

    assert decision.status == DecisionStatus.INCONCLUSIVE
    assert decision.winning_hypothesis_id is None


def test_unknown_candidate_fails_closed_as_out_of_scope():
    proposal = _proposal("unknownMechanism")

    decision = CoreDecisionProvider().decide(
        (proposal,),
        (),
        StopReason.POLICY_REQUEST,
    )

    assert decision.status == DecisionStatus.OUT_OF_SCOPE
    assert decision.winning_hypothesis_id is None


def test_single_condition_catalog_candidate_can_confirm():
    proposal = _proposal("dbNetworkLatency").model_copy(
        update={"observation_refs": ("obs-0000000000000001",)}
    )
    decision = CoreDecisionProvider().decide(
        (proposal,),
        (_record("obs-0000000000000001", "connection_duration_ms", 180.0),),
        StopReason.POLICY_REQUEST,
    )

    assert decision.status == DecisionStatus.CONFIRMED_ROOT_CAUSE
    assert decision.winning_hypothesis_id == "h-dbnetworklatency"


def test_episode_runner_uses_issued_observation_ledger_for_core_decision():
    now = datetime.now(UTC)

    class TraceEnvironment:
        def reset(self):
            return ResetResponse(
                event_time=now,
                alert_summary="slow operation",
                alert_entity=EntityRef(
                    entity_ref="ent-service",
                    display_name="service",
                    entity_type="apm.operation",
                ),
                observation_horizon=TimeWindow(
                    start=now - timedelta(minutes=5),
                    end=now,
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
                query_budget=QueryBudgetView(initial=2, remaining=2),
            )

        def get_trace_stage_breakdown(self, request):
            query_ref = query_fingerprint(ToolName.GET_TRACE_STAGE_BREAKDOWN, request)
            return ToolResponse(
                tool_name=ToolName.GET_TRACE_STAGE_BREAKDOWN,
                query_ref=query_ref,
                status=ToolStatus.SUCCESS,
                observations=(
                    _record("obs-0000000000000001", "sql_duration_share", 0.8).observation,
                    _record(
                        "obs-0000000000000002", "sql_shape", "complex_cross_join"
                    ).observation,
                ),
                cost=QueryCost(units=1, budget_before=2, budget_after=1),
                message="bounded trace-stage facts returned",
            )

    runner = EpisodeRunner(TraceEnvironment())
    runner.start()
    response = runner.handle(
        CallToolAction(
            tool_name=ToolName.GET_TRACE_STAGE_BREAKDOWN,
            request=GetTraceStageBreakdownRequest(trace_ref="trace-0000000000000001"),
        )
    )
    proposal = _proposal("slowSQL")
    runner.handle(SubmitHypothesisAction(proposal=proposal))
    answer = runner.handle(RequestStopAction(summary="let the domain core decide"))

    assert response.status == ToolStatus.SUCCESS
    assert answer.authoritative_decision.status == DecisionStatus.CONFIRMED_ROOT_CAUSE
    assert answer.authoritative_decision.winning_hypothesis_id == proposal.hypothesis_id
