from datetime import UTC, datetime, timedelta
from time import sleep

import pytest
from langchain_core.runnables import RunnableLambda
from pydantic import ValidationError

from rca_agent.core.dynamic import (
    AlternativeDraft,
    DiscoveryPayload,
    DynamicHypothesis,
    EvidenceRegistry,
    MechanismClass,
    MechanismDiscoveryPayload,
    MechanismDraft,
    PlannedPrediction,
    Prediction,
    RuleSpec,
    SignalSpec,
    VerificationPlanDraft,
    VerificationPlanExtension,
    VerificationPlanPayload,
    assess_dynamic,
    freeze_mechanism,
    normalize_verification_plan,
    repair_mechanical_contract,
    verify_dynamic,
)
from rca_agent.core.v3.models import ConditionPolarity, MechanismFamily
from rca_agent.langchain_discovery import DiscoveryPayloadError, LangChainDiscoveryAdapter
from rca_agent.langchain_policy import LangChainPolicyAdapter, ScriptedPolicyRunnable
from rca_agent.langgraph_orchestrator import LangGraphInvestigation
from rca_agent.models import (
    CallToolAction,
    CapabilityDescriptor,
    DataStatus,
    DecisionStatus,
    EntityRef,
    GetMetricSeriesRequest,
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
    ToolResponse,
    ToolStatus,
    UpdateCandidateEvidenceAction,
)
from rca_agent.runner import CoreDecisionProvider, EpisodeRunner, IssuedObservation, RecordedReplay
from rca_agent.runner.plan_extension import added_predictions
from rca_agent.runner.transcript import EventType
from rca_agent.tools.fingerprint import query_fingerprint, sha256_digest

NOW = datetime(2026, 9, 14, tzinfo=UTC)
WINDOW = TimeWindow(start=NOW - timedelta(minutes=5), end=NOW)


def query(signal):
    return CallToolAction(
        tool_name="get_metric_series",
        request=GetMetricSeriesRequest(
            entity_ref="ent-service",
            signal=signal,
            window=WINDOW,
        ),
    )


def registry():
    return EvidenceRegistry(
        signals=tuple(
            SignalSpec(
                signal=s,
                entity_ref="ent-service",
                data_type="number",
                unit="ratio",
                description=description,
                independent_group=s,
                query=query(s),
                minimum=0,
                maximum=1,
            )
            for s, description in (
                ("active_ratio", "busy workers divided by maximum workers"),
                ("queue_ratio", "queued tasks divided by queue capacity"),
                ("downstream_wait_ratio", "fraction of worker time blocked on downstream"),
                ("gc_pause_ratio", "fraction of time paused for garbage collection"),
                ("traffic_growth_ratio", "incoming traffic relative to baseline"),
            )
        ),
        rules=tuple(
            RuleSpec(
                rule_id=s,
                signal=s,
                comparison="gte",
                threshold=threshold,
                source="configuration",
                provenance="test-only load-test contract",
                refutes=tuple(MechanismClass),
            )
            for s, threshold in (
                ("active_ratio", 0.9),
                ("queue_ratio", 0.8),
                ("downstream_wait_ratio", 0.1),
                ("gc_pause_ratio", 0.1),
                ("traffic_growth_ratio", 0.1),
            )
        ),
    )


def draft(refute=False):
    return DynamicHypothesis(
        source_hypothesis_id="h-unknown",
        mechanism="thread pool exhaustion",
        family=MechanismFamily.APPLICATION,
        causal_explanation="Workers are occupied and tasks queue in this specific pool.",
        predictions=tuple(
            Prediction(
                rule_id=s,
                polarity=ConditionPolarity.SUPPORT,
                rationale="Expected capacity pressure",
            )
            for s in ("active_ratio", "queue_ratio")
        )
        + (
            (
                Prediction(
                    rule_id="downstream_wait_ratio",
                    polarity=ConditionPolarity.CONTRADICT,
                    rationale="Downstream blocking would explain worker occupancy instead",
                    distinguishes=("downstream blocking",),
                ),
            )
            if refute
            else ()
        ),
        alternatives=("downstream blocking",),
    )


class Environment:
    def __init__(self, values=None, budget=20):
        self.values = values or {
            "active_ratio": 0.95,
            "queue_ratio": 0.9,
            "downstream_wait_ratio": 0,
            "gc_pause_ratio": 0,
            "traffic_growth_ratio": 0,
        }
        self.remaining = budget
        self.calls = []

    def reset(self):
        return ResetResponse(
            event_time=NOW,
            alert_summary="request queue delay",
            alert_entity=EntityRef(
                entity_ref="ent-service", display_name="service", entity_type="service"
            ),
            observation_horizon=WINDOW,
            capabilities=(
                CapabilityDescriptor(
                    tool_name="get_metric_series",
                    modality=Modality.METRIC,
                    available=True,
                    query_cost=1,
                    max_window_seconds=300,
                    max_results=100,
                ),
            ),
            query_budget=QueryBudgetView(initial=self.remaining, remaining=self.remaining),
        )

    def get_metric_series(self, request):
        before = self.remaining
        self.remaining = max(0, before - 1)
        self.calls.append(request.signal)
        ref = query_fingerprint("get_metric_series", request)
        obs = Observation(
            observation_id="obs-" + sha256_digest(request.signal)[:16],
            modality=Modality.METRIC,
            entity_ref=request.entity_ref,
            signal=request.signal,
            value=self.values.get(request.signal),
            unit="ratio",
            data_status=DataStatus.OBSERVED,
        )
        return ToolResponse(
            tool_name="get_metric_series",
            query_ref=ref,
            status=ToolStatus.SUCCESS if before else ToolStatus.BUDGET_EXHAUSTED,
            observations=(obs,) if before else (),
            cost=QueryCost(
                units=int(before > 0), budget_before=before, budget_after=self.remaining
            ),
            message="bounded facts",
        )


class Discoverer:
    def __init__(self, hypothesis=None):
        self.calls = 0
        self.hypothesis = hypothesis or draft()

    def discover(self, context):
        self.calls += 1
        self.context = context
        return DiscoveryPayload(hypotheses=(self.hypothesis,))


def unknown_action():
    return SubmitHypothesisAction(
        proposal=HypothesisProposal(
            hypothesis_id="h-unknown",
            candidate_id="unknownMechanism",
            mechanism="unknown queue bottleneck",
            causal_entity_ref="ent-service",
            rationale="initial bounded observation warrants exploration",
        )
    )


def run(discoverer=None, reg=None, environment=None, *, enable_core_feedback=False, **kwargs):
    discoverer = discoverer or Discoverer()
    environment = environment or Environment()
    runner = EpisodeRunner(
        environment, discovery=discoverer, evidence_registry=reg or registry(), **kwargs
    )
    policy = LangChainPolicyAdapter(
        ScriptedPolicyRunnable(
            (
                query("active_ratio"),
                unknown_action(),
                RequestStopAction(summary="investigation complete"),
            )
        )
    )
    return (runner, LangGraphInvestigation(
        runner, policy, enable_core_feedback=enable_core_feedback,
    ).run(), discoverer, environment)


def test_unknown_candidate_full_runner_query_loop_and_replay():
    runner, report, discoverer, environment = run()
    assert discoverer.calls == 1
    assert environment.calls == ["active_ratio", "active_ratio", "queue_ratio"]
    assert report.answer.authoritative_decision.status == DecisionStatus.SUPPORTED_HYPOTHESIS
    assert len(report.answer.policy_proposals) == 2
    assert report.answer.policy_proposals[0].observation_refs == ()
    assert len(report.answer.policy_proposals[1].observation_refs) == 2
    assert RecordedReplay.verify(runner.transcript.events).valid
    assert "threshold" not in str(discoverer.context)
    assert "comparison" not in str(discoverer.context)
    assert len(runner.dynamic_candidate_options) == 1


def test_post_freeze_coverage_respects_candidate_bindings_and_completeness():
    from copy import deepcopy

    from evaluation.evidence_coverage import evidence_coverage

    runner, report, _, _ = run(Discoverer(draft(refute=True)))
    result = {
        "tool_responses": [e.payload for e in runner.transcript.events
                           if e.event_type == EventType.TOOL_RESPONSE],
        "registered_candidates": [e.payload for e in runner.transcript.events
                                  if e.event_type == EventType.DYNAMIC_CANDIDATE_REGISTERED],
        "policy_proposals": [p.model_dump(mode="json") for p in report.answer.policy_proposals],
    }
    full = evidence_coverage(result)
    assert full["checks_with_eligible_evidence"] == full["required_checks"] == 3
    detached = deepcopy(result)
    for proposal in detached["policy_proposals"]:
        proposal["observation_refs"] = []
    assert evidence_coverage(detached)["checks_with_eligible_evidence"] == 0
    partial = deepcopy(result)
    for response in partial["tool_responses"]:
        response["truncated"] = True
    assert evidence_coverage(partial)["checks_with_eligible_evidence"] == 0


def test_generic_verification_without_per_mechanism_configuration():
    runner, report, _, _ = run(Discoverer(draft(refute=True)), registry())
    assert report.answer.authoritative_decision.status == DecisionStatus.DISCOVERED_VERIFIED
    assert report.final_frontier.discovered_verified_candidate_ids
    assert not report.final_frontier.confirmed_candidate_ids
    assert report.final_preview.overall_decision == report.answer.authoritative_decision
    assert RecordedReplay.verify(runner.transcript.events).valid
    _, contradicted, _, _ = run(
        Discoverer(draft(refute=True)),
        registry(),
        Environment({"active_ratio": 0.95, "queue_ratio": 0.9, "downstream_wait_ratio": 0.2}),
    )
    assert contradicted.answer.authoritative_decision.status != DecisionStatus.DISCOVERED_VERIFIED
    assert contradicted.final_frontier.contradicted_candidate_ids


def test_missing_data_does_not_become_refutation():
    _, report, _, _ = run(environment=Environment({"active_ratio": 0.95}))
    candidate = report.final_preview.candidate_assessments[0]
    assert not candidate.contradicting_signals
    assert "queue_ratio" in (*candidate.missing_signals, *candidate.unavailable_signals)


@pytest.mark.parametrize(
    "change",
    [
        {"signal": "invented"},
        {"threshold": -1},
        {"threshold": "high"},
        {"comparison": "contains"},
        {"source": "baseline"},
    ],
)
def test_bad_rule_contract_rejected(change):
    raw = registry().model_dump()
    raw["rules"][0].update(change)
    with pytest.raises(ValueError):
        EvidenceRegistry.model_validate(raw)


def test_model_cannot_author_threshold_and_host_groups_limit_verification():
    raw = draft().model_dump()
    raw["predictions"][0]["threshold"] = -100
    with pytest.raises(ValidationError):
        DynamicHypothesis.model_validate(raw)
    raw = registry().model_dump()
    raw["signals"][1]["independent_group"] = "active_ratio"
    reg = EvidenceRegistry.model_validate(raw)
    compiled = reg.compile(draft(), "ent-service")
    env = Environment()
    records = tuple(
        IssuedObservation(
            response.observations[0], response.query_ref,
        )
        for response in (
            env.get_metric_series(query("active_ratio").request),
            env.get_metric_series(query("queue_ratio").request),
        )
    )
    verification = verify_dynamic(
        compiled, tuple(assess_dynamic(compiled, records)), records
    )
    assert "insufficient_independent_support" in verification.blockers


def test_noncausal_symptom_rule_cannot_count_as_root_cause_support():
    base = registry()
    restricted = EvidenceRegistry(
        signals=base.signals,
        rules=(base.rules[0].model_copy(update={"support_eligible": False}), *base.rules[1:]),
    )

    with pytest.raises(ValueError, match="not eligible as root-cause support"):
        restricted.compile(draft(), "ent-service")


def test_generic_gc_or_other_generic_rules_cannot_claim_specific_full_gc():
    specific = draft().model_copy(update={"mechanism": "Full GC pause storm"})

    with pytest.raises(ValueError, match="cannot establish the specific Full GC subtype"):
        registry().compile(specific, "ent-service")


def test_nonfinite_wrong_unit_entity_and_query_are_not_evidence():
    compiled = registry().compile(draft(), "ent-service")
    env = Environment()
    response = env.get_metric_series(query("active_ratio").request)
    observation = response.observations[0]
    for change in (
        {"unit": "ms"},
        {"entity_ref": "ent-another"},
        {"value": float("nan")},
        {"data_status": DataStatus.MODEL_INFERRED},
        {"value": True},
    ):
        rows = (IssuedObservation(observation.model_copy(update=change), response.query_ref),)
        assert assess_dynamic(compiled, rows)[0].observation is None
    rows = (IssuedObservation(observation, "qry-0000000000000000"),)
    assert assess_dynamic(compiled, rows)[0].observation is None


def test_registration_is_immutable_and_episode_scoped():
    compiled = registry().compile(draft(), "ent-service")
    provider = CoreDecisionProvider()
    provider.register_dynamic(compiled)
    with pytest.raises(ValueError):
        provider.register_dynamic(compiled)
    compiled.candidate.conditions[0].threshold = -100
    assert (
        provider._dynamic[compiled.candidate.candidate_id].candidate.conditions[0].threshold == 0.9
    )
    proposal = unknown_action().proposal.model_copy(
        update={"candidate_id": compiled.candidate.candidate_id}
    )
    assert (
        CoreDecisionProvider().decide((proposal,), (), StopReason.POLICY_REQUEST).status
        == DecisionStatus.OUT_OF_SCOPE
    )


def test_core_feedback_only_offers_unattempted_frozen_queries():
    runner = EpisodeRunner(
        Environment(), discovery=Discoverer(hypothesis=draft(refute=True)),
        evidence_registry=registry(), max_rounds=20,
    )
    runner.start()
    runner.handle(query("active_ratio"))
    runner.handle(unknown_action())
    runner.discover_candidates()
    assert runner.has_verification_queries
    assert not runner.feedback_query_options()
    # Simulate a queued verification query that was not run before handoff.
    runner._EpisodeRunner__verification_queries.clear()
    options = runner.feedback_query_options()
    assert {item[1] for item in options} >= {"queue_ratio", "downstream_wait_ratio"}
    assert "active_ratio" not in {item[1] for item in options}
    hypothesis_id, _, action = options[0]
    response = runner.handle(action)
    runner.handle(UpdateCandidateEvidenceAction(
        hypothesis_id=hypothesis_id,
        observation_refs=tuple(item.observation_id for item in response.observations),
    ))
    assert action not in {item[2] for item in runner.feedback_query_options()}
    runner.handle(RequestStopAction(summary="bounded feedback test complete"))
    assert RecordedReplay.verify(runner.transcript.events).valid


def test_failure_and_timeout_are_bounded_and_replayable():
    class Broken:
        def discover(self, context):
            raise RuntimeError("secret provider payload")

    runner, report, _, _ = run(Broken())
    assert report.answer.authoritative_decision.status == DecisionStatus.OUT_OF_SCOPE
    assert "secret provider payload" not in str(runner.transcript.events)
    assert RecordedReplay.verify(runner.transcript.events).valid

    class InvalidPayload:
        def discover(self, context):
            raise DiscoveryPayloadError({
                "hypotheses": [{"mechanism": "possible heap pressure"}],
            })

    runner, report, _, _ = run(InvalidPayload())
    rejected = next(
        event.payload for event in runner.transcript.events
        if event.event_type == EventType.DISCOVERY_PAYLOAD_REJECTED
    )
    assert rejected["raw_payload"]["hypotheses"][0]["mechanism"] == (
        "possible heap pressure"
    )
    assert report.answer.authoritative_decision.status == DecisionStatus.OUT_OF_SCOPE
    assert RecordedReplay.verify(runner.transcript.events).valid

    class Slow:
        def discover(self, context):
            sleep(0.05)
            return DiscoveryPayload(hypotheses=(draft(),))

    runner, _, _, _ = run(Slow(), discovery_timeout_seconds=0.001)
    sleep(0.06)
    assert not runner.dynamic_candidate_options
    assert any(e.payload.get("status") == "timeout" for e in runner.transcript.events)


def test_round_and_budget_bounds_stop_verification():
    runner, report, _, _ = run(max_rounds=3)
    assert report.answer.termination_reason == StopReason.MAX_ROUNDS
    assert RecordedReplay.verify(runner.transcript.events).valid
    runner, report, _, _ = run(environment=Environment(budget=2))
    assert report.answer.termination_reason == StopReason.BUDGET_EXHAUSTED
    assert RecordedReplay.verify(runner.transcript.events).valid


def test_langchain_discovery_adapter_validates_model_output():
    adapter = LangChainDiscoveryAdapter(
        RunnableLambda(lambda _: {"hypotheses": [draft().model_dump()]})
    )
    assert adapter.discover({}).hypotheses[0].mechanism == "thread pool exhaustion"


def test_langchain_discovery_adapter_preserves_invalid_structured_json():
    class RawMessage:
        content = '{"hypotheses":[{"mechanism":"heap pressure"}]}'

    adapter = LangChainDiscoveryAdapter(RunnableLambda(lambda _: {
        "raw": RawMessage(), "parsed": None, "parsing_error": ValueError("private detail"),
    }))
    with pytest.raises(DiscoveryPayloadError) as exc_info:
        adapter.discover({})
    assert exc_info.value.raw_payload == {
        "hypotheses": [{"mechanism": "heap pressure"}],
    }


def test_mechanical_repair_only_canonicalizes_exact_alternative_references():
    original = draft(refute=True).model_copy(update={
        "alternatives": ("  Downstream   Blocking  ",),
    })
    repaired, actions = repair_mechanical_contract(original)
    assert actions == ("canonicalized_alternative_references",)
    assert repaired.predictions[-1].distinguishes == ("  Downstream   Blocking  ",)
    assert repaired.mechanism == original.mechanism
    assert repaired.causal_explanation == original.causal_explanation
    assert repaired.predictions[:-1] == original.predictions[:-1]

    semantic_mismatch = draft(refute=True).model_copy(update={
        "alternatives": ("downstream database latency",),
    })
    unchanged, actions = repair_mechanical_contract(semantic_mismatch)
    assert unchanged == semantic_mismatch
    assert not actions


def test_runner_records_raw_draft_and_auditable_mechanical_repair():
    repairable = draft(refute=True).model_copy(update={
        "alternatives": ("Downstream Blocking",),
    })
    runner, _, _, _ = run(Discoverer(repairable))
    evaluated = next(
        event.payload for event in runner.transcript.events
        if event.event_type == EventType.DISCOVERY_DRAFT_EVALUATED
    )
    assert evaluated["raw_draft"]["alternatives"] == ["Downstream Blocking"]
    assert evaluated["initial_status"] == "rejected"
    assert evaluated["repair_actions"] == ["canonicalized_alternative_references"]
    assert evaluated["final_status"] == "accepted"
    assert evaluated["final_draft"]["mechanism"] == repairable.mechanism
    assert RecordedReplay.verify(runner.transcript.events).valid


class TwoStageDiscoverer:
    def discover(self, context):
        assert "signals" not in context
        assert "rules" not in context
        assert all("available_rule_ids" not in node for node in context["entity_graph"]["nodes"])
        return MechanismDiscoveryPayload(mechanisms=(MechanismDraft(
            source_hypothesis_id="h-unknown",
            causal_entity_ref="ent-service",
            mechanism="thread pool exhaustion",
            mechanism_class=MechanismClass.THREAD_POOL_SATURATION,
            family=MechanismFamily.APPLICATION,
            causal_explanation="Workers are occupied and incoming tasks wait in the queue.",
            alternatives=(AlternativeDraft(
                alternative_id="alt-downstream",
                mechanism="downstream blocking",
                mechanism_class=MechanismClass.DOWNSTREAM_DEPENDENCY,
            ),),
        ),))


class TwoStagePlanner:
    def __init__(self, *, fail_first=False, always_fail=False):
        self.calls = 0
        self.fail_first = fail_first
        self.always_fail = always_fail

    def plan(self, context):
        self.calls += 1
        mechanism_id = context["frozen_mechanisms"][0]["mechanism_id"]
        handles = {item["rule_id"]: item["capability_ref"]
                   for item in context["capabilities"]}
        invalid = self.always_fail or (self.fail_first and self.calls == 1)
        active = "cap-" + "0" * 20 if invalid else handles["active_ratio"]
        return VerificationPlanPayload(plans=(VerificationPlanDraft(
            mechanism_id=mechanism_id,
            predictions=(
                PlannedPrediction(
                    capability_ref=active,
                    polarity=ConditionPolarity.SUPPORT,
                    rationale="Workers should be busy.",
                ),
                PlannedPrediction(
                    capability_ref=handles["queue_ratio"],
                    polarity=ConditionPolarity.SUPPORT,
                    rationale="Tasks should wait in the queue.",
                ),
                PlannedPrediction(
                    capability_ref=handles["downstream_wait_ratio"],
                    polarity=ConditionPolarity.CONTRADICT,
                    rationale="Downstream blocking would refute local pool exhaustion.",
                    distinguishes=("alt-downstream",),
                ),
            ),
        ),))


class CompetingMechanismsDiscoverer:
    def discover(self, context):
        return MechanismDiscoveryPayload(mechanisms=(
            MechanismDraft(
                source_hypothesis_id="h-unknown",
                causal_entity_ref="ent-service",
                mechanism="local thread pool exhaustion",
                mechanism_class=MechanismClass.THREAD_POOL_SATURATION,
                family=MechanismFamily.APPLICATION,
                causal_explanation="Local workers are occupied while work queues locally.",
                alternatives=(AlternativeDraft(
                    alternative_id="alt-load-spike",
                    mechanism="transient incoming load spike",
                    mechanism_class=MechanismClass.TRAFFIC_SURGE,
                ),),
            ),
            MechanismDraft(
                source_hypothesis_id="h-unknown",
                causal_entity_ref="ent-service",
                mechanism="downstream wait induced worker starvation",
                mechanism_class=MechanismClass.DOWNSTREAM_DEPENDENCY,
                family=MechanismFamily.DATABASE_DEPENDENCY,
                causal_explanation="Workers remain occupied because downstream calls block them.",
                alternatives=(AlternativeDraft(
                    alternative_id="alt-lock-contention",
                    mechanism="local lock contention",
                    mechanism_class=MechanismClass.LOCK_CONTENTION,
                ),),
            ),
        ))


class CompetingMechanismsPlanner:
    def plan(self, context):
        handles = {
            item["rule_id"]: item["capability_ref"]
            for item in context["capabilities"]
        }
        plans = []
        for mechanism in context["frozen_mechanisms"]:
            alternatives = tuple(
                item["alternative_id"] for item in mechanism["alternatives"]
            )
            refutation_rules = ("downstream_wait_ratio", "gc_pause_ratio")
            plans.append(VerificationPlanDraft(
                mechanism_id=mechanism["mechanism_id"],
                predictions=(
                    PlannedPrediction(
                        capability_ref=handles["active_ratio"],
                        polarity=ConditionPolarity.SUPPORT,
                        rationale="Workers should be occupied.",
                    ),
                    PlannedPrediction(
                        capability_ref=handles["queue_ratio"],
                        polarity=ConditionPolarity.SUPPORT,
                        rationale="Queued work should accumulate.",
                    ),
                ) + tuple(
                    PlannedPrediction(
                        capability_ref=handles[refutation_rules[index]],
                        polarity=ConditionPolarity.CONTRADICT,
                        rationale="This observation must separate the peer explanation.",
                        distinguishes=(alternative,),
                    )
                    for index, alternative in enumerate(alternatives)
                ),
            ))
        return VerificationPlanPayload(plans=tuple(plans))


class FailedCompetitorPlanner:
    def plan(self, context):
        handles = {
            item["rule_id"]: item["capability_ref"]
            for item in context["capabilities"]
        }
        plans = []
        for index, mechanism in enumerate(context["frozen_mechanisms"]):
            alternatives = tuple(
                item["alternative_id"] for item in mechanism["alternatives"]
            )
            support_rules = (
                ("active_ratio", "queue_ratio")
                if index == 0
                else ("active_ratio", "downstream_wait_ratio")
            )
            contradict_rules = (
                ("downstream_wait_ratio", "gc_pause_ratio")
                if index == 0
                else ("queue_ratio", "traffic_growth_ratio")
            )
            plans.append(VerificationPlanDraft(
                mechanism_id=mechanism["mechanism_id"],
                predictions=tuple(
                    PlannedPrediction(
                        capability_ref=handles[rule_id],
                        polarity=ConditionPolarity.SUPPORT,
                        rationale="The mechanism requires this independent support group.",
                    )
                    for rule_id in support_rules
                ) + tuple(
                    PlannedPrediction(
                        capability_ref=handles[contradict_rules[alternative_index]],
                        polarity=ConditionPolarity.CONTRADICT,
                        rationale="This observation separates the peer explanation.",
                        distinguishes=(alternative,),
                    )
                    for alternative_index, alternative in enumerate(alternatives)
                ),
            ))
        return VerificationPlanPayload(plans=tuple(plans))


def test_two_stage_discovery_freezes_mechanism_and_compiles_capability_handles():
    planner = TwoStagePlanner()
    runner, _, _, _ = run(TwoStageDiscoverer(), verification_planner=planner)
    assert planner.calls == 1
    events = runner.transcript.events
    assert sum(event.event_type == EventType.MECHANISM_DRAFT_FROZEN for event in events) == 1
    assert sum(event.event_type == EventType.VERIFICATION_PLAN_COMPILED for event in events) == 1
    assert EventType.VERIFICATION_PLAN_RETRIED not in {event.event_type for event in events}
    assert runner.dynamic_candidate_options
    assert RecordedReplay.verify(events).valid

    compiled = next(event.payload for event in events
                    if event.event_type == EventType.VERIFICATION_PLAN_COMPILED)
    with pytest.raises(ValidationError):
        VerificationPlanDraft.model_validate({
            "mechanism_id": compiled["plan"]["mechanism_id"],
            "mechanism": "attempted mutation",
            "predictions": compiled["plan"]["predictions"],
        })


def test_core_feedback_adds_a_distinguishing_plan_and_replays():
    class FeedbackPlanner(TwoStagePlanner):
        supports_plan_extension = True

        def plan(self, context):
            initial = super().plan(context)
            item = initial.plans[0]
            predictions = tuple(
                prediction.model_copy(update={"distinguishes": ()})
                if prediction.polarity == ConditionPolarity.CONTRADICT else prediction
                for prediction in item.predictions
            )
            return VerificationPlanPayload(plans=(item.model_copy(update={
                "predictions": predictions,
            }),))

        def plan_extension(self, context):
            handle = next(item["capability_ref"] for item in context["capabilities"]
                          if item["rule_id"] == "traffic_growth_ratio")
            return VerificationPlanExtension(
                mechanism_id=context["frozen_mechanisms"][0]["mechanism_id"],
                added_predictions=(PlannedPrediction(
                    capability_ref=handle,
                    polarity=ConditionPolarity.CONTRADICT,
                    rationale="No traffic rise distinguishes local pool pressure.",
                    distinguishes=("alt-downstream",),
                ),),
            )

    runner, report, _, _ = run(
        TwoStageDiscoverer(), verification_planner=FeedbackPlanner(),
        max_rounds=30, enable_core_feedback=True,
    )
    answer = report.answer
    assert answer.authoritative_decision.status == DecisionStatus.DISCOVERED_VERIFIED
    assert any(item.get("kind") == "core_feedback_plan"
               and item["result"]["status"] == "extended" for item in report.history)
    for previous_event, following_event in zip(
        runner.transcript.events, runner.transcript.events[1:], strict=False,
    ):
        assert following_event.previous_digest == previous_event.digest, (
            previous_event.sequence, following_event.sequence,
        )
    assert RecordedReplay.verify(runner.transcript.events).valid
    assert sum(event.event_type == EventType.VERIFICATION_PLAN_EXTENDED
               for event in runner.transcript.events) == 1


def test_feedback_plan_cannot_remove_or_rewrite_prior_predictions():
    old = VerificationPlanDraft(
        mechanism_id="mech-" + "a" * 20,
        predictions=(
            PlannedPrediction(capability_ref="cap-" + "a" * 20,
                              polarity=ConditionPolarity.SUPPORT, rationale="first"),
            PlannedPrediction(capability_ref="cap-" + "b" * 20,
                              polarity=ConditionPolarity.CONTRADICT, rationale="second"),
        ),
    )
    added = PlannedPrediction(capability_ref="cap-" + "c" * 20,
                              polarity=ConditionPolarity.CONTRADICT, rationale="third")
    with pytest.raises(ValueError, match="removed or changed"):
        added_predictions(old, old.model_copy(update={
            "predictions": (old.predictions[0].model_copy(update={"rationale": "rewritten"}),
                            old.predictions[1], added),
        }))


def test_competing_dynamic_mechanisms_cannot_both_be_promoted_as_root_cause():
    runner, report, _, _ = run(
        CompetingMechanismsDiscoverer(),
        verification_planner=CompetingMechanismsPlanner(),
    )
    verification = next(
        event.payload["candidates"]
        for event in runner.transcript.events
        if event.event_type == EventType.DYNAMIC_VERIFICATION
    )
    assert len(verification) == 2
    assert all(not item["passed"] for item in verification.values())
    assert all(
        "competing_dynamic_mechanism_unresolved" in item["blockers"]
        for item in verification.values()
    )
    decision = report.answer.authoritative_decision
    assert decision.status == DecisionStatus.MULTIPLE_SUPPORTED
    assert len(decision.supported_hypothesis_ids) == 2
    assert RecordedReplay.verify(runner.transcript.events).valid


def test_decisively_failed_dynamic_competitor_does_not_block_verified_mechanism():
    runner, report, _, _ = run(
        CompetingMechanismsDiscoverer(),
        verification_planner=FailedCompetitorPlanner(),
    )
    verification = next(
        event.payload["candidates"]
        for event in runner.transcript.events
        if event.event_type == EventType.DYNAMIC_VERIFICATION
    )
    states = sorted(item["state"] for item in verification.values())
    assert states == ["failed", "verified"]
    assert report.answer.authoritative_decision.status == DecisionStatus.DISCOVERED_VERIFIED
    assert RecordedReplay.verify(runner.transcript.events).valid


def test_plan_normalization_merges_same_capability_and_polarity_only():
    base = PlannedPrediction(
        capability_ref="cap-" + "1" * 20,
        polarity=ConditionPolarity.CONTRADICT,
        rationale="One bounded refutation test.",
        distinguishes=("alt-one",),
    )
    plan = VerificationPlanDraft(
        mechanism_id="mech-" + "2" * 20,
        predictions=(
            base,
            base.model_copy(update={"distinguishes": ("alt-two",)}),
            base.model_copy(update={
                "capability_ref": "cap-" + "3" * 20,
                "polarity": ConditionPolarity.SUPPORT,
                "distinguishes": (),
            }),
        ),
    )
    normalized, repairs = normalize_verification_plan(plan)
    assert repairs == ("merged_duplicate_capability_predictions",)
    assert len(normalized.predictions) == 2
    assert normalized.predictions[0].distinguishes == ("alt-one", "alt-two")


def test_two_stage_refutation_must_be_dedicated_to_one_alternative():
    discoverer = CompetingMechanismsDiscoverer()
    payload = discoverer.discover({})
    mechanism = payload.mechanisms[0]
    peer = payload.mechanisms[1]
    frozen = freeze_mechanism(mechanism.model_copy(update={
        "alternatives": (
            *mechanism.alternatives,
            AlternativeDraft(
                alternative_id="alt-peer-mechanism",
                mechanism=peer.mechanism,
                mechanism_class=peer.mechanism_class,
            ),
        ),
    }))
    handles = {item.rule_id: item.capability_ref for item in registry().capability_handles()}
    plan = VerificationPlanDraft(
        mechanism_id=frozen.mechanism_id,
        predictions=(
            PlannedPrediction(
                capability_ref=handles["active_ratio"],
                polarity=ConditionPolarity.SUPPORT,
                rationale="Independent support one.",
            ),
            PlannedPrediction(
                capability_ref=handles["queue_ratio"],
                polarity=ConditionPolarity.SUPPORT,
                rationale="Independent support two.",
            ),
            PlannedPrediction(
                capability_ref=handles["downstream_wait_ratio"],
                polarity=ConditionPolarity.CONTRADICT,
                rationale="One signal cannot discharge unrelated alternatives.",
                distinguishes=("alt-load-spike", "alt-peer-mechanism"),
            ),
        ),
    )
    with pytest.raises(ValueError, match="at most one alternative"):
        registry().compile_plan(frozen, plan)


def test_two_stage_refutation_scope_must_match_alternative_class():
    mechanism = TwoStageDiscoverer().discover({
        "entity_graph": {"nodes": []},
    }).mechanisms[0]
    frozen = freeze_mechanism(mechanism)
    base = registry()
    restricted = base.model_copy(update={
        "rules": tuple(
            rule.model_copy(update={
                "refutes": (MechanismClass.GC_PAUSE,)
                if rule.rule_id == "downstream_wait_ratio"
                else rule.refutes,
            })
            for rule in base.rules
        ),
    })
    handles = {
        item.rule_id: item.capability_ref for item in restricted.capability_handles()
    }
    plan = VerificationPlanDraft(
        mechanism_id=frozen.mechanism_id,
        predictions=(
            PlannedPrediction(
                capability_ref=handles["active_ratio"],
                polarity=ConditionPolarity.SUPPORT,
                rationale="Independent support one.",
            ),
            PlannedPrediction(
                capability_ref=handles["queue_ratio"],
                polarity=ConditionPolarity.SUPPORT,
                rationale="Independent support two.",
            ),
            PlannedPrediction(
                capability_ref=handles["downstream_wait_ratio"],
                polarity=ConditionPolarity.CONTRADICT,
                rationale="The capability scope does not match this alternative.",
                distinguishes=("alt-downstream",),
            ),
        ),
    )
    with pytest.raises(ValueError, match="cannot refute"):
        restricted.compile_plan(frozen, plan)

    normalized, repairs = restricted.normalize_plan(frozen, plan)
    assert "removed_unvalidated_refutation_claim" in repairs
    assert normalized.predictions[-1].distinguishes == ()
    compiled = restricted.compile_plan(frozen, normalized)
    assert compiled.alternatives == ("downstream blocking",)


def test_two_stage_planner_gets_one_bounded_retry_without_changing_mechanism():
    planner = TwoStagePlanner(fail_first=True)
    runner, _, _, _ = run(TwoStageDiscoverer(), verification_planner=planner)
    assert planner.calls == 2
    events = runner.transcript.events
    frozen = next(event.payload["mechanism"] for event in events
                  if event.event_type == EventType.MECHANISM_DRAFT_FROZEN)
    compiled = next(event.payload for event in events
                    if event.event_type == EventType.VERIFICATION_PLAN_COMPILED)
    assert compiled["attempt"] == 2
    assert compiled["mechanism_digest"] == frozen["digest"]
    assert sum(event.event_type == EventType.VERIFICATION_PLAN_RETRIED for event in events) == 1
    assert RecordedReplay.verify(events).valid

    failed = TwoStagePlanner(always_fail=True)
    runner, _, _, _ = run(TwoStageDiscoverer(), verification_planner=failed)
    assert failed.calls == 2
    assert not runner.dynamic_candidate_options
    assert sum(event.event_type == EventType.VERIFICATION_PLAN_RETRIED
               for event in runner.transcript.events) == 1
    assert RecordedReplay.verify(runner.transcript.events).valid


def test_two_stage_planner_retries_one_transient_invocation_failure():
    class TransientPlanner(TwoStagePlanner):
        def plan(self, context):
            if self.calls == 0:
                self.calls += 1
                raise RuntimeError("private provider failure")
            return super().plan(context)

    planner = TransientPlanner()
    runner, _, _, _ = run(TwoStageDiscoverer(), verification_planner=planner)
    assert planner.calls == 2
    assert runner.dynamic_candidate_options
    assert "private provider failure" not in str(runner.transcript.events)
    assert sum(event.event_type == EventType.VERIFICATION_PLAN_RETRIED
               for event in runner.transcript.events) == 1
    assert RecordedReplay.verify(runner.transcript.events).valid


def test_unknown_signal_recipe_is_rejected_before_query():
    raw = registry().model_dump()
    raw["signals"][0]["query"]["request"]["window"] = {
        "start": NOW - timedelta(days=1),
        "end": NOW,
    }
    reg = EvidenceRegistry.model_validate(raw)
    runner, report, _, env = run(reg=reg)
    assert env.calls == ["active_ratio"]
    assert report.answer.authoritative_decision.status == DecisionStatus.OUT_OF_SCOPE
    assert RecordedReplay.verify(runner.transcript.events).valid


def test_capacity_reference_and_unit_validation():
    raw = registry().model_dump()
    raw["rules"][0].update(
        threshold=None, reference_signal="queue_ratio", source="capacity", factor=2
    )
    reg = EvidenceRegistry.model_validate(raw)
    compiled = reg.compile(draft(), "ent-service")
    env = Environment({"active_ratio": 0.95, "queue_ratio": 0.4})
    records = []
    for signal in ("active_ratio", "queue_ratio"):
        response = env.get_metric_series(query(signal).request)
        records.append(IssuedObservation(response.observations[0], response.query_ref))
    assert assess_dynamic(compiled, tuple(records))[0].matched
    assert assess_dynamic(compiled, tuple(records[:1]))[0].observation is None
    raw["signals"][1]["unit"] = "ms"
    with pytest.raises(ValueError):
        EvidenceRegistry.model_validate(raw)


def test_sustained_requires_coverage_and_complete_results():
    raw = registry().model_dump()
    raw["rules"][0].update(
        aggregation="sustained",
        min_samples=3,
        min_duration_seconds=60,
        max_gap_seconds=30,
        minimum_fraction=0.8,
    )
    compiled = EvidenceRegistry.model_validate(raw).compile(draft(), "ent-service")
    response = Environment().get_metric_series(query("active_ratio").request)
    rows = tuple(
        IssuedObservation(
            response.observations[0].model_copy(
                update={
                    "observation_id": f"obs-{i:016x}",
                    "observed_at": NOW - timedelta(seconds=30 * i),
                }
            ),
            response.query_ref,
        )
        for i in range(3)
    )
    assert assess_dynamic(compiled, rows)[0].matched
    assert assess_dynamic(compiled, rows[:2])[0].observation is None
    incomplete = tuple(IssuedObservation(r.observation, r.query_ref, False) for r in rows)
    assert assess_dynamic(compiled, incomplete)[0].observation is None
    sparse = (
        *rows[:2],
        IssuedObservation(
            rows[2].observation.model_copy(
                update={
                    "observed_at": NOW - timedelta(seconds=120),
                }
            ),
            response.query_ref,
        ),
    )
    assert assess_dynamic(compiled, sparse)[0].observation is None


def test_stop_request_gets_one_discovery_attempt_then_freezes():
    class EmptyDiscoverer:
        def discover(self, context):
            return DiscoveryPayload()

    runner = EpisodeRunner(Environment(), discovery=EmptyDiscoverer(), evidence_registry=registry())
    # Existing Catalog candidate with incomplete observations: only the stop triggers discovery.
    proposal = unknown_action().proposal.model_copy(update={"candidate_id": "slowSQL"})
    policy = LangChainPolicyAdapter(
        ScriptedPolicyRunnable(
            (
                query("active_ratio"),
                SubmitHypothesisAction(proposal=proposal),
                RequestStopAction(summary="no further planned queries"),
            )
        )
    )
    report = LangGraphInvestigation(runner, policy).run()
    assert report.answer.termination_reason == StopReason.POLICY_REQUEST
    assert sum(e.event_type == EventType.DISCOVERY_STARTED for e in runner.transcript.events) == 1
    assert RecordedReplay.verify(runner.transcript.events).valid


def test_factory_uses_issued_trace_and_reuses_existing_candidate():
    from rca_agent.core.runtime_registry import rca100_evidence_registry

    bootstrap = Environment().reset()
    bootstrap = bootstrap.model_copy(
        update={
            "capabilities": (
                CapabilityDescriptor(
                    tool_name="get_trace_stage_breakdown",
                    modality=Modality.TRACE,
                    available=True,
                    query_cost=1,
                    max_window_seconds=300,
                    max_results=8,
                ),
            )
        }
    )
    summary = Observation(
        observation_id="obs-0000000000000001",
        modality=Modality.TRACE,
        entity_ref="ent-service",
        signal="operation_trace_duration_ms",
        value=300,
        unit="ms",
        data_status=DataStatus.OBSERVED,
        details={"trace_ref": "trace-0000000000000001"},
    )
    verification_summary = summary.model_copy(update={
        "observation_id": "obs-0000000000000002",
        "details": {"trace_ref": "trace-0000000000000002"},
    })
    stage = Observation(
        observation_id="obs-0000000000000003",
        modality=Modality.TRACE,
        entity_ref="ent-root-service",
        signal="sql_duration_share",
        value=0.9,
        unit="ratio",
        data_status=DataStatus.DERIVED,
        details={"trace_ref": "trace-0000000000000001"},
    )
    reg = rca100_evidence_registry(bootstrap, (summary, verification_summary, stage))
    assert {s.signal for s in reg.signals} == {
        "operation_duration_ms",
        "sql_shape",
        "sql_duration_ms",
        "sql_duration_share",
        "connection_duration_ms",
        "connection_acquisition_ms",
        "db_client_span_duration_ms",
    }
    assert {s.entity_ref for s in reg.signals} == {"ent-root-service"}
    assert {
        s.query.request.trace_ref for s in reg.signals
    } == {"trace-0000000000000002"}
    assert not rca100_evidence_registry(bootstrap, ()).signals


def test_factory_registers_jvm_thread_rules_for_an_issued_service():
    from rca_agent.core.runtime_registry import rca100_evidence_registry

    bootstrap = Environment().reset()
    observations = (
        Observation(
            observation_id="obs-0000000000000001",
            modality=Modality.TOPOLOGY,
            entity_ref="ent-service",
            signal="related_entity",
            value={
                "entity_ref": "ent-inventory",
                "display_name": "inventory",
                "entity_type": "apm.service",
            },
            data_status=DataStatus.DERIVED,
        ),
    )
    registry = rca100_evidence_registry(bootstrap, observations)

    assert {spec.signal for spec in registry.signals} >= {
        "thread_pool_active", "thread_pool_max", "thread_pool_queue_depth",
        "thread_pool_rejected_tasks", "jvm_runnable_threads", "jvm_thread_count",
        "jvm_thread_cpu_time_ns",
        "jvm_heap_used_bytes", "jvm_heap_max_bytes", "jvm_gc_collection_delta",
        "jvm_gc_pause_seconds_delta",
    }
    rules = {rule.rule_id: rule for rule in registry.rules}
    assert "thread_pool_active" not in rules
    rejected = next(
        spec for spec in registry.signals if spec.signal == "thread_pool_rejected_tasks"
    )
    assert rejected.independent_group == "thread_pool_saturation"
    gc_groups = {
        spec.independent_group for spec in registry.signals
        if spec.signal in {"jvm_gc_collection_delta", "jvm_gc_pause_seconds_delta"}
    }
    assert gc_groups == {"jvm_gc_activity"}
    assert {spec.entity_ref for spec in registry.signals} == {"ent-inventory"}
    assert "thread_pool_max" not in rules
    assert "jvm_heap_max_bytes" not in rules


def test_trace_symptoms_and_exclusion_rules_are_not_causal_support():
    from rca_agent.core.runtime_registry import rca100_evidence_registry

    bootstrap = Environment().reset().model_copy(update={
        "capabilities": (
            CapabilityDescriptor(
                tool_name="get_trace_stage_breakdown",
                modality=Modality.TRACE,
                available=True,
                query_cost=1,
                max_window_seconds=300,
                max_results=12,
            ),
        )
    })
    summary = Observation(
        observation_id="obs-0000000000000001",
        modality=Modality.TRACE,
        entity_ref="ent-source",
        signal="operation_trace_duration_ms",
        value=300,
        unit="ms",
        data_status=DataStatus.OBSERVED,
        details={"trace_ref": "trace-0000000000000001"},
    )
    stage = Observation(
        observation_id="obs-0000000000000002",
        modality=Modality.TRACE,
        entity_ref="ent-target",
        signal="sql_duration_share",
        value=0.1,
        unit="ratio",
        data_status=DataStatus.DERIVED,
        details={"trace_ref": "trace-0000000000000001"},
    )
    rules = {
        rule.rule_id: rule for rule in rca100_evidence_registry(
            bootstrap, (summary, stage)
        ).rules
    }

    assert not rules["operation_duration_ms"].support_eligible
    assert not rules["sql_duration_ms"].support_eligible
    assert not rules["db_client_span_duration_ms"].support_eligible
    assert not rules["sql_duration_share_low"].support_eligible
    assert rules["sql_duration_share"].support_eligible
    assert rules["sql_shape"].support_eligible


def test_provider_factory_no_network(monkeypatch):
    from rca_agent.deepseek_policy import create_deepseek_discovery

    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-only-key")
    assert isinstance(create_deepseek_discovery(), LangChainDiscoveryAdapter)


def test_compiled_registration_after_freeze_is_rejected_by_replay():
    from rca_agent.runner.replay import ReplayValidationError

    runner, _, _, _ = run()
    registration = next(
        e
        for e in runner.transcript.events
        if e.event_type == EventType.DYNAMIC_CANDIDATE_REGISTERED
    )
    runner.transcript.append(registration.event_type, registration.payload)
    with pytest.raises(ReplayValidationError):
        RecordedReplay.verify(runner.transcript.events)


def test_discovery_aligns_known_mechanism_without_replacing_catalog():
    specs = (
        SignalSpec(
            signal="sql_duration_share",
            entity_ref="ent-service",
            data_type="number",
            unit="ratio",
            description="SQL execution share",
            independent_group="duration",
            query=query("sql_duration_share"),
        ),
        SignalSpec(
            signal="sql_shape",
            entity_ref="ent-service",
            data_type="string",
            unit="ratio",
            description="query structure",
            independent_group="structure",
            query=query("sql_shape"),
        ),
    )
    reg = EvidenceRegistry(
        signals=specs,
        rules=(
            RuleSpec(
                rule_id="sql_duration_share",
                signal="sql_duration_share",
                comparison="gte",
                threshold=0.8,
                source="configuration",
                provenance="test fixture",
            ),
            RuleSpec(
                rule_id="sql_shape",
                signal="sql_shape",
                comparison="eq",
                threshold="complex_cross_join",
                source="configuration",
                provenance="test fixture",
            ),
        ),
    )
    hypothesis = draft().model_copy(
        update={
            "mechanism": "slowSQL",
            "family": MechanismFamily.DATABASE_DEPENDENCY,
            "predictions": tuple(
                Prediction(rule_id=s.signal, polarity="support", rationale="test prediction")
                for s in specs
            ),
        }
    )
    env = Environment(
        {"active_ratio": 0.95, "sql_duration_share": 0.9, "sql_shape": "complex_cross_join"}
    )
    runner, report, _, _ = run(Discoverer(hypothesis), reg, env)
    assert not runner.dynamic_candidate_options
    assert report.answer.authoritative_decision.status == DecisionStatus.CONFIRMED_ROOT_CAUSE
    assert report.answer.policy_proposals[-1].candidate_id == "slowSQL"
    assert RecordedReplay.verify(runner.transcript.events).valid


def test_dynamic_evidence_cannot_be_attached_to_another_hypothesis():
    compiled = registry().compile(draft(), "ent-service")
    provider = CoreDecisionProvider()
    provider.register_dynamic(compiled)
    other = unknown_action().proposal.model_copy(
        update={
            "hypothesis_id": "h-other",
            "candidate_id": compiled.candidate.candidate_id,
        }
    )
    assert (
        provider.decide((other,), (), StopReason.POLICY_REQUEST).status
        == DecisionStatus.OUT_OF_SCOPE
    )
    with pytest.raises(ValueError):
        EpisodeRunner(Environment(), decision_provider=provider)


@pytest.mark.parametrize("mechanism", ["unregistered worker saturation", "new mechanism 42"])
def test_mechanism_name_does_not_gate_generic_verification(mechanism):
    hypothesis = draft(refute=True).model_copy(update={"mechanism": mechanism})
    runner, report, _, _ = run(Discoverer(hypothesis))
    assert report.answer.authoritative_decision.status == DecisionStatus.DISCOVERED_VERIFIED
    assert report.answer.policy_proposals[-1].mechanism == mechanism
    assert "this investigation's root cause" in report.answer.authoritative_decision.reason
    assert RecordedReplay.verify(runner.transcript.events).valid


def verification_event(runner):
    event = next(
        e for e in runner.transcript.events if e.event_type == EventType.DYNAMIC_VERIFICATION
    )
    return next(iter(event.payload["candidates"].values()))


def test_missing_refutation_data_blocks_verified_root_cause():
    runner, report, _, _ = run(
        Discoverer(draft(refute=True)),
        environment=Environment(
            {
                "active_ratio": 0.95,
                "queue_ratio": 0.9,
            }
        ),
    )
    assert report.answer.authoritative_decision.status == DecisionStatus.SUPPORTED_HYPOTHESIS
    assert "required_observations_unavailable" in verification_event(runner)["blockers"]
    assert verification_event(runner)["state"] == "unresolved"
    assert report.final_preview.candidate_assessments[0].verification_pending


def test_unresolved_alternative_blocks_result_even_with_positive_support():
    hypothesis = draft(refute=True).model_copy(
        update={
            "alternatives": ("downstream blocking", "scheduler pause"),
        }
    )
    runner, report, _, _ = run(Discoverer(hypothesis))
    assert report.answer.authoritative_decision.status == DecisionStatus.SUPPORTED_HYPOTHESIS
    assert "alternative_explanations_unresolved" in verification_event(runner)["blockers"]


def test_undeclared_alternative_is_rejected_by_compiler():
    raw = draft(refute=True).model_dump()
    raw["predictions"][-1]["distinguishes"] = ["made up alternative"]
    with pytest.raises(ValueError, match="undeclared alternative"):
        registry().compile(DynamicHypothesis.model_validate(raw), "ent-service")


def test_requerying_discovery_facts_does_not_count_as_new_verification():
    runner = EpisodeRunner(
        Environment(), discovery=Discoverer(draft(refute=True)), evidence_registry=registry()
    )
    policy = LangChainPolicyAdapter(
        ScriptedPolicyRunnable(
            (
                query("active_ratio"),
                query("queue_ratio"),
                query("downstream_wait_ratio"),
                unknown_action(),
                RequestStopAction(summary="all available data checked"),
            )
        )
    )
    report = LangGraphInvestigation(runner, policy).run()
    assert report.answer.authoritative_decision.status == DecisionStatus.SUPPORTED_HYPOTHESIS
    assert "no_new_support_verification" in verification_event(runner)["blockers"]
    assert RecordedReplay.verify(runner.transcript.events).valid


def test_observation_id_reuse_is_not_new_verification():
    env = Environment()
    records = []
    for signal in ("active_ratio", "queue_ratio", "downstream_wait_ratio"):
        response = env.get_metric_series(query(signal).request)
        records.append(IssuedObservation(response.observations[0], response.query_ref))
    compiled = registry().compile(
        draft(refute=True),
        "ent-service",
        discovery_observation_refs=tuple(r.observation.observation_id for r in records),
    )
    assessments = tuple(assess_dynamic(compiled, tuple(records)))
    assert (
        "no_new_support_verification"
        in verify_dynamic(compiled, assessments, tuple(records)).blockers
    )


def test_verified_status_requires_core_and_winner():
    from rca_agent.models import AuthoritativeDecision

    for source, winner in (("rca_agent_core", None), ("phase2_lifecycle_default", "h-test")):
        with pytest.raises(ValidationError):
            AuthoritativeDecision(
                status=DecisionStatus.DISCOVERED_VERIFIED,
                source=source,
                winning_hypothesis_id=winner,
                reason="not authoritative",
            )


def test_replay_recomputes_generic_verification_instead_of_trusting_pass_flag():
    from rca_agent.runner.replay import ReplayValidationError
    from rca_agent.runner.transcript import Transcript

    runner, _, _, _ = run()
    rebuilt = Transcript(runner.episode_id)
    for event in runner.transcript.events:
        if event.event_type == EventType.ANSWER_FROZEN:
            break
        payload = event.payload
        if event.event_type == EventType.DYNAMIC_VERIFICATION:
            payload = {
                "candidates": {
                    k: {**v, "passed": True, "blockers": []}
                    for k, v in payload["candidates"].items()
                }
            }
        rebuilt.append(event.event_type, payload)
    with pytest.raises(ReplayValidationError, match="differs from recorded evidence"):
        RecordedReplay.verify(rebuilt.events)


def test_rule_definitions_and_discovery_snapshot_are_recorded_before_queries():
    runner, report, _, _ = run(Discoverer(draft(refute=True)))
    registration = next(
        e
        for e in runner.transcript.events
        if e.event_type == EventType.DYNAMIC_CANDIDATE_REGISTERED
    )
    compiled = registration.payload["compiled"]
    assert compiled["discovery_query_refs"] == [
        query_fingerprint("get_metric_series", query("active_ratio").request)
    ]
    assert len(compiled["predictions"]) == 3
    assert compiled["alternatives"] == ["downstream blocking"]
    assert "confirmation" not in compiled
    assert report.final_preview.overall_decision == report.answer.authoritative_decision
    proof = verification_event(runner)
    assert proof["passed"] and proof["standard"] == "generic-root-cause-v1"
    assert proof["distinguished_alternatives"] == ["downstream blocking"]


def test_same_facts_with_new_ids_and_query_limits_do_not_verify():
    from rca_agent.core.dynamic import observation_fact_digest

    original_records = []
    env = Environment()
    for signal in ("active_ratio", "queue_ratio", "downstream_wait_ratio"):
        response = env.get_metric_series(query(signal).request)
        original_records.append(IssuedObservation(response.observations[0], response.query_ref))
    raw = registry().model_dump()
    for spec in raw["signals"]:
        spec["query"]["request"]["max_points"] = 51
    reg = EvidenceRegistry.model_validate(raw)
    compiled = reg.compile(draft(refute=True), "ent-service",
                           discovery_fact_digests=tuple(observation_fact_digest(r.observation)
                                                       for r in original_records))
    specs_by_signal = {spec.signal: spec for spec in reg.signals}
    records = tuple(IssuedObservation(
        original.observation.model_copy(update={"observation_id": f"obs-{i:016x}"}),
        query_fingerprint(
            specs_by_signal[original.observation.signal].query.tool_name,
            specs_by_signal[original.observation.signal].query.request,
        ),
    ) for i, original in enumerate(original_records))
    report = verify_dynamic(compiled, tuple(assess_dynamic(compiled, records)), records)
    assert not report.passed
    assert "no_new_support_verification" in report.blockers


def test_a_new_but_unsatisfied_sample_cannot_certify_old_maximum():
    from rca_agent.core.dynamic import observation_fact_digest

    raw = registry().model_dump()
    for rule in raw["rules"]:
        rule["aggregation"] = "max"
    reg = EvidenceRegistry.model_validate(raw)
    env = Environment()
    old_records = []
    for signal in ("active_ratio", "queue_ratio", "downstream_wait_ratio"):
        response = env.get_metric_series(query(signal).request)
        old_records.append(IssuedObservation(response.observations[0], response.query_ref))
    compiled = reg.compile(draft(refute=True), "ent-service",
                           discovery_fact_digests=tuple(observation_fact_digest(r.observation)
                                                       for r in old_records))
    new = IssuedObservation(old_records[0].observation.model_copy(update={
        "observation_id": "obs-1111111111111111", "value": .1,
    }), old_records[0].query_ref)
    records = (*old_records, new)
    report = verify_dynamic(compiled, tuple(assess_dynamic(compiled, records)), records)
    assert not report.passed
    assert "no_new_support_verification" in report.blockers


@pytest.mark.parametrize("alternatives", [("",), ("downstream blocking", "Downstream blocking")])
def test_blank_or_duplicate_alternatives_are_not_valid_tests(alternatives):
    raw = draft().model_dump()
    raw["alternatives"] = alternatives
    with pytest.raises(ValidationError):
        DynamicHypothesis.model_validate(raw)


def test_catalog_holdout_is_enforced_in_policy_discovery_core_and_replay():
    from rca_agent.langgraph_orchestrator import _candidate_options

    excluded = frozenset({"dbNetworkLatency"})
    provider = CoreDecisionProvider(excluded)
    assert "dbNetworkLatency" not in {c.candidate_id for c in provider.catalog("ent-service")}
    assert "dbNetworkLatency" not in {
        c.candidate_id for c in _candidate_options("ent-service", excluded)
    }
    proposal = unknown_action().proposal.model_copy(update={
        "candidate_id": "dbNetworkLatency", "mechanism": "dbNetworkLatency",
    })
    assert provider.decide((proposal,), (), StopReason.POLICY_REQUEST).status == DecisionStatus.OUT_OF_SCOPE

    discoverer = Discoverer(draft(refute=True))
    runner = EpisodeRunner(Environment(), decision_provider=provider, discovery=discoverer,
                           evidence_registry=registry())
    policy = LangChainPolicyAdapter(ScriptedPolicyRunnable((
        query("active_ratio"), unknown_action(), RequestStopAction(summary="done"),
    )))
    LangGraphInvestigation(runner, policy).run()
    assert "dbNetworkLatency" not in {
        c["candidate_id"] for c in discoverer.context["existing_candidates"]
    }
    assert runner.transcript.events[0].payload["excluded_catalog_candidate_ids"] == [
        "dbNetworkLatency"
    ]
    assert RecordedReplay.verify(runner.transcript.events).valid


def test_discovery_can_return_distinct_mechanisms_for_one_source_hypothesis():
    second = draft(refute=True).model_copy(update={"mechanism": "another queue mechanism"})

    class Multiple:
        def discover(self, context):
            return DiscoveryPayload(hypotheses=(draft(refute=True), second))

    runner = EpisodeRunner(Environment(), discovery=Multiple(), evidence_registry=registry())
    policy = LangChainPolicyAdapter(ScriptedPolicyRunnable((
        query("active_ratio"), unknown_action(), RequestStopAction(summary="done"),
    )))
    report = LangGraphInvestigation(runner, policy).run()
    assert len(runner.dynamic_candidate_options) == 2
    assert len([p for p in report.answer.policy_proposals if p.hypothesis_id.startswith(
        "h-discovered-")]) == 2
    assert RecordedReplay.verify(runner.transcript.events).valid
