"""Host-owned alternative identity and full synthetic pipeline acceptance."""
from copy import deepcopy

import pytest
from test_bound_formalization import BoundDiscovery
from test_dynamic_investigation import (
    Environment,
    TwoStageDiscoverer,
    TwoStagePlanner,
    query,
    registry,
    unknown_action,
)

from rca_agent.core.dynamic import MechanismClass
from rca_agent.investigation.bound_formalization import (
    AlternativeDetails,
    BoundCandidateResponseV7,
    CauseDetails,
    bind_candidate,
    candidate_binding,
)
from rca_agent.langchain_acquisition import InquiryHandoff, InquiryHypothesis
from rca_agent.models import RequestStopAction
from rca_agent.runner import EpisodeRunner, RecordedReplay, ReplayValidationError
from rca_agent.runner.transcript import EventType, Transcript


class DiscoveryV7(BoundDiscovery):
    def formalize_candidate(self, context):
        self.calls.append(context["binding"]["hypothesis_id"])
        if self.fail_first and context["binding"]["hypothesis_id"] == "h1":
            raise ValueError("private failure")
        template = TwoStageDiscoverer().discover(context).mechanisms[0]
        return BoundCandidateResponseV7(status="formalized", reason_code="FORMALIZED",
            reason="Typed original cause and its distinct competing explanations",
            cause_details=CauseDetails.model_validate(template.model_dump(exclude={
                "mechanism", "source_hypothesis_id", "alternatives"})),
            alternative_details=tuple(AlternativeDetails(alternative_id=a["alternative_id"],
                mechanism_class=MechanismClass.UNKNOWN) for a in context["binding"]["alternative_bindings"]))


def run_v7(*, fail_first=False, alternatives=None):
    discovery = DiscoveryV7(fail_first)
    runner = EpisodeRunner(Environment(), discovery=discovery, verification_planner=TwoStagePlanner(),
                           evidence_registry=registry(), max_rounds=20)
    runner.start()
    runner.handle(query("active_ratio"))
    runner.handle(unknown_action())
    originals = tuple(InquiryHypothesis(hypothesis_id=f"h{i}", mechanism=(
        "worker exhaustion" if alternatives else f"worker exhaustion mechanism {i}"),
        alternative=text) for i, text in enumerate(alternatives or
            ["connection pool exhaustion", "frontend worker exhaustion"], 1))
    handoff = InquiryHandoff(revision=1, hypotheses=originals, needs=(), query_links=())
    result = runner.discover_candidates(inquiry_handoff=handoff.model_dump(mode="json"))
    runner.handle(RequestStopAction(summary="new host-bound contract"))
    assert RecordedReplay.verify(runner.transcript.events).valid
    return runner, discovery, result


def test_original_alternatives_keep_identity_exact_text_and_provenance():
    runner, discovery, result = run_v7()
    assert discovery.calls == ["h1", "h2"] and result["registered"] == 2
    records = [e.payload for e in runner.transcript.events
               if e.event_type == EventType.BOUND_CANDIDATE_FORMALIZATION_RECORDED]
    frozen = [e.payload["mechanism"] for e in runner.transcript.events
              if e.event_type == EventType.MECHANISM_DRAFT_FROZEN]
    for record, mechanism in zip(records, frozen, strict=True):
        original = record["binding"]["alternative_bindings"][0]
        alternative = next(a for a in mechanism["alternatives"] if a["alternative_id"] == original["alternative_id"])
        assert alternative["mechanism"] == original["mechanism"]
        assert alternative["inquiry_provenance"] == original["provenance"]
        assert mechanism["mechanism"] == record["binding"]["mechanism"]
    assert all(c.alternative for c in runner.answer.inquiry_candidates)


def test_same_cause_retains_multiple_round_alternatives():
    runner, _, _ = run_v7(alternatives=["connection pool exhaustion", "frontend worker exhaustion"])
    frozen = [e.payload["mechanism"] for e in runner.transcript.events
              if e.event_type == EventType.MECHANISM_DRAFT_FROZEN]
    assert len(frozen) == 1
    assert {a["mechanism"] for a in frozen[0]["alternatives"]} == {
        "connection pool exhaustion", "frontend worker exhaustion"}


def test_v7_failed_candidate_retry_is_isolated_and_original_stays_visible():
    runner, discovery, result = run_v7(fail_first=True)
    assert discovery.calls == ["h1", "h1", "h2"]
    assert result["registered"] == 1
    assert len(runner.answer.inquiry_candidates) == 2
    assert runner.answer.inquiry_candidates[0].discovery_disposition == "deferred"
    assert "private failure" not in str([e.payload for e in runner.transcript.events])


@pytest.mark.parametrize("alternative,rejected", [
    ("  WORKER   exhaustion  ", True), ("worker exhaustion upstream", False),
    ("frontend worker exhaustion", False),
])
def test_self_check_is_strict_normalized_equality(alternative, rejected):
    candidate = {"hypothesis_id": "h1", "mechanism": "worker exhaustion", "alternative": alternative}
    binding = candidate_binding(candidate, "h-unknown", version=7)
    response = DiscoveryV7().formalize_candidate({"binding": binding, "entity_graph": {"nodes": []}})
    if rejected:
        with pytest.raises(ValueError, match="SELF_ALTERNATIVE"):
            bind_candidate(binding, response, {"ent-service"})
    else:
        assert bind_candidate(binding, response, {"ent-service"})[0].alternatives[0].mechanism == alternative


@pytest.mark.parametrize("target", ["alternative_id", "text", "cause_role", "binding", "missing"])
def test_v7_rejects_model_replacement_and_rehashed_tampering(target):
    runner, _, _ = run_v7()
    rebuilt = Transcript(runner.episode_id)
    changed = False
    for event in runner.transcript.events:
        payload = deepcopy(event.payload)
        if event.event_type == EventType.BOUND_CANDIDATE_FORMALIZATION_RECORDED and not changed:
            if target == "alternative_id":
                payload["response"]["alternative_details"][0]["alternative_id"] = "alt-fake"
            elif target == "text":
                payload["response"]["alternative_details"][0]["mechanism"] = "replacement"
            elif target == "cause_role":
                payload["response"]["cause_details"]["mechanism_class"] = "database_connection_pool"
            elif target == "missing":
                payload["response"]["alternative_details"] = []
            else:
                payload["binding"]["alternative_bindings"][0]["mechanism"] = "replacement"
            changed = True
        rebuilt.append(event.event_type, payload)
    with pytest.raises((ReplayValidationError, ValueError)):
        RecordedReplay.verify(rebuilt.events)


def test_new_external_self_alternative_is_deferred_before_freeze():
    runner, _, result = run_v7(alternatives=["WORKER exhaustion"])
    assert result["registered"] == 0
    assert not any(e.event_type == EventType.MECHANISM_DRAFT_FROZEN for e in runner.transcript.events)
    assert runner.answer.inquiry_candidates[0].alternative == "WORKER exhaustion"


def test_full_synthetic_inquiry_to_answer_and_replay_without_extra_calls():
    import json

    from langchain_core.runnables import RunnableLambda

    from rca_agent.inquiry_policy import InquiryInvestigationPolicy
    from rca_agent.langchain_acquisition import LangChainInquiryAdapter, query_options
    from rca_agent.models.contract import MetricDescriptor

    class PublicEnvironment(Environment):
        def reset(self):
            base = super().reset()
            capability = base.capabilities[0].model_copy(update={"max_results": 50, "metrics": tuple(
                MetricDescriptor(entity_type="service", signal=s, unit="ratio", description="bounded occupancy ratio")
                for s in ("active_ratio", "queue_ratio"))})
            return base.model_copy(update={"capabilities": (capability,)})

    calls = []
    def respond(payload):
        calls.append(payload)
        menu = json.loads(payload["context"])["queries"]
        choices = [{"need_id": "n1", "query_ref": ref, "mode": "direct", "purpose": "Measure occupancy",
                    "measurement_target": {k: item[k] for k in ("entity_ref", "signal", "measurement_kind", "windows")}}
                   for ref, item in menu.items()]
        return {"hypotheses": [{"hypothesis_id": "h1", "mechanism": "worker exhaustion",
                                "alternative": "connection pool exhaustion"}],
                "needs": [{"need_id": "n1", "hypothesis_id": "h1", "question": "Measure occupancy",
                           "observable_signals": ["active_ratio", "queue_ratio"], "entity_ref": "ent-service",
                           "expected_if_true": "busy", "expected_if_alternative": "not busy"}],
                "choices": choices}
    discovery = DiscoveryV7()
    runner = EpisodeRunner(PublicEnvironment(), discovery=discovery, verification_planner=TwoStagePlanner(),
                           evidence_registry=registry(), investigation_workspace_mode="active", max_rounds=20)
    bootstrap = runner.start()
    policy = InquiryInvestigationPolicy(LangChainInquiryAdapter(RunnableLambda(respond)), query_limit=2)
    policy.selection_calls = 1
    options = query_options({"bootstrap": bootstrap.model_dump(mode="json")},
                            [bootstrap.alert_entity.model_dump(mode="json")])
    policy._plan_inquiry_round({}, options)
    runner.record_inquiry_round(policy.inquiry_drafts[0])
    history = []
    for action in policy.selection_queue:
        routed = runner.route_active_inquiry_action(action)
        response = runner.handle(routed)
        policy.record_routed_action(action, routed)
        history.append({"action": routed.model_dump(mode="json"),
                        "result": {"kind": "tool_response", "value": response.model_dump(mode="json")}})
    handoff = policy.discovery_handoff(tuple(history))
    assert len(handoff["query_links"]) == 2
    runner.handle(unknown_action())
    result = runner.discover_candidates(inquiry_handoff=handoff)
    runner.handle(RequestStopAction(summary="synthetic full chain"))
    assert result["registered"] == 1 and len(calls) == 1 and discovery.calls == ["h1"]
    assert runner.answer.verify_digest() and RecordedReplay.verify(runner.transcript.events).valid
    assert all(t.status.value != "resolved" for t in runner.answer.investigation_workspace.tasks)
    original_id = policy.inquiry_drafts[0]["draft"]["hypotheses"][0]["alternative_id"]
    frozen = next(e.payload["mechanism"] for e in runner.transcript.events if e.event_type == EventType.MECHANISM_DRAFT_FROZEN)
    assert frozen["alternatives"][0]["alternative_id"] == original_id
    assert runner.answer.inquiry_candidates[0].alternative_id == original_id

    tampered = Transcript(runner.episode_id)
    for event in runner.transcript.events:
        payload = deepcopy(event.payload)
        if event.event_type == EventType.DISCOVERY_STARTED:
            payload["inquiry_handoff"]["query_links"][0]["observation_refs"] = []
        tampered.append(event.event_type, payload)
    with pytest.raises(ReplayValidationError):
        RecordedReplay.verify(tampered.events)
