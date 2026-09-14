from datetime import UTC, datetime, timedelta
from time import sleep

import pytest
from langchain_core.runnables import RunnableLambda
from pydantic import ValidationError

from rca_agent.core.dynamic import (
    DiscoveryPayload,
    DynamicHypothesis,
    EvidenceRegistry,
    Prediction,
    RuleSpec,
    SignalSpec,
    assess_dynamic,
    verify_dynamic,
)
from rca_agent.core.v3.models import ConditionPolarity, MechanismFamily
from rca_agent.langchain_discovery import LangChainDiscoveryAdapter
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
)
from rca_agent.runner import CoreDecisionProvider, EpisodeRunner, IssuedObservation, RecordedReplay
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
            )
            for s, threshold in (
                ("active_ratio", 0.9),
                ("queue_ratio", 0.8),
                ("downstream_wait_ratio", 0.1),
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


def run(discoverer=None, reg=None, environment=None, **kwargs):
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
    return runner, LangGraphInvestigation(runner, policy).run(), discoverer, environment


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


def test_model_cannot_author_threshold_or_groups():
    raw = draft().model_dump()
    raw["predictions"][0]["threshold"] = -100
    with pytest.raises(ValidationError):
        DynamicHypothesis.model_validate(raw)
    raw = registry().model_dump()
    raw["signals"][1]["independent_group"] = "active_ratio"
    with pytest.raises(ValueError):
        EvidenceRegistry.model_validate(raw).compile(draft(), "ent-service")


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


def test_failure_and_timeout_are_bounded_and_replayable():
    class Broken:
        def discover(self, context):
            raise RuntimeError("secret provider payload")

    runner, report, _, _ = run(Broken())
    assert report.answer.authoritative_decision.status == DecisionStatus.OUT_OF_SCOPE
    assert "secret provider payload" not in str(runner.transcript.events)
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
    observation = Observation(
        observation_id="obs-0000000000000001",
        modality=Modality.TRACE,
        entity_ref="ent-service",
        signal="operation_trace_duration_ms",
        value=300,
        unit="ms",
        data_status=DataStatus.OBSERVED,
        details={"trace_ref": "trace-0000000000000001"},
    )
    reg = rca100_evidence_registry(bootstrap, (observation,))
    assert {s.signal for s in reg.signals} == {
        "sql_shape",
        "sql_duration_share",
        "connection_duration_ms",
    }
    assert not rca100_evidence_registry(bootstrap, ()).signals


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
    records = tuple(IssuedObservation(
        original.observation.model_copy(update={"observation_id": f"obs-{i:016x}"}),
        query_fingerprint(spec.query.tool_name, spec.query.request),
    ) for i, (original, spec) in enumerate(zip(original_records, reg.signals, strict=True)))
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
