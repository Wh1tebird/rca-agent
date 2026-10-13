from copy import deepcopy

import pytest
from test_dynamic_investigation import (
    Environment,
    TwoStageDiscoverer,
    TwoStagePlanner,
    query,
    registry,
    unknown_action,
)

from rca_agent.core.dynamic import MechanismDiscoveryPayload, NewMechanismBasis
from rca_agent.investigation.bound_formalization import (
    BoundCandidateResponse,
    CandidateDetails,
    admit_novel,
    bind_candidate,
    candidate_binding,
    compatible_classes,
)
from rca_agent.langchain_acquisition import InquiryHandoff, InquiryHypothesis
from rca_agent.langchain_discovery import LangChainMechanismDiscoveryAdapter
from rca_agent.models import RequestStopAction
from rca_agent.runner import EpisodeRunner, RecordedReplay, ReplayValidationError
from rca_agent.runner.transcript import EventType, Transcript


class BoundDiscovery:
    supports_bound_candidates = True

    def __init__(self, fail_first=False):
        self.fail_first = fail_first
        self.calls = []

    def formalize_candidate(self, context):
        self.calls.append(context["binding"]["hypothesis_id"])
        if self.fail_first and context["binding"]["hypothesis_id"] == "h1":
            raise ValueError("private provider diagnostic")
        template = TwoStageDiscoverer().discover(context).mechanisms[0]
        details = CandidateDetails.model_validate(template.model_dump(exclude={"mechanism", "source_hypothesis_id"}))
        return BoundCandidateResponse(status="formalized", details=details,
                                      reason_code="FORMALIZED", reason="Typed original causal mechanism.")

    def discover(self, context):
        assert context["new_discovery_only"]
        assert "inquiry_handoff" not in context
        return MechanismDiscoveryPayload()


def run_bound(fail_first=False, count=2):
    discovery = BoundDiscovery(fail_first)
    runner = EpisodeRunner(Environment(), discovery=discovery, verification_planner=TwoStagePlanner(),
                           evidence_registry=registry(), max_rounds=20, bound_handoff_version=6)
    runner.start()
    runner.handle(query("active_ratio"))
    runner.handle(unknown_action())
    handoff = InquiryHandoff(revision=1, hypotheses=tuple(
        InquiryHypothesis(hypothesis_id=f"h{i}", mechanism=f"worker exhaustion mechanism {i}",
                          alternative="connection pool exhaustion") for i in range(1, count + 1)
    ), needs=(), query_links=())
    result = runner.discover_candidates(inquiry_handoff=handoff.model_dump(mode="json"))
    runner.handle(RequestStopAction(summary="bound candidate test"))
    assert RecordedReplay.verify(runner.transcript.events).valid
    return runner, discovery, result


def test_bound_identity_cannot_be_written_by_model():
    response = BoundCandidateResponse.model_json_schema()
    assert "mechanism" not in response["$defs"]["CandidateDetails"]["properties"]
    assert "source_hypothesis_id" not in response["$defs"]["CandidateDetails"]["properties"]
    with pytest.raises(ValueError):
        BoundCandidateResponse.model_validate({"status": "deferred", "reason_code": "MISSING_FIELDS",
                                               "reason": "cannot type", "mechanism": "pool exhaustion"})


def test_candidate_adapter_rejects_empty_parsed_output():
    class EmptyRunnable:
        def invoke(self, context):
            return {"raw": {}, "parsed": None}

    adapter = LangChainMechanismDiscoveryAdapter(None, EmptyRunnable())
    assert adapter.supports_bound_candidates
    with pytest.raises(ValueError, match="candidate structured output failed"):
        adapter.formalize_candidate({})


def test_candidate_adapter_preserves_explicit_deferred_response():
    class DeferredRunnable:
        def invoke(self, context):
            return {"raw": {}, "parsed": {"status": "deferred", "details": None,
                    "reason_code": "AMBIGUOUS_ENTITY", "reason": "No issued service identity."}}

    adapter = LangChainMechanismDiscoveryAdapter(None, DeferredRunnable())
    assert adapter.formalize_candidate({}).reason_code == "AMBIGUOUS_ENTITY"


def test_type_guard_uses_original_not_alternative_and_never_assigns_class():
    template = TwoStageDiscoverer().discover({"entity_graph": {"nodes": []}}).mechanisms[0]
    details = CandidateDetails.model_validate(template.model_dump(exclude={"mechanism", "source_hypothesis_id"}))
    response = BoundCandidateResponse(status="formalized", details=details,
                                     reason_code="FORMALIZED", reason="Model typed candidate.")
    candidate = {"hypothesis_id": "h1", "mechanism": "slow SQL query execution",
                 "alternative": "thread pool exhaustion"}
    binding = candidate_binding(candidate, "h-unknown")
    with pytest.raises(ValueError, match="class conflicts"):
        bind_candidate(binding, response, {"ent-service"}, check_class=True)
    # Historical v5 replay does not retroactively apply the new guard.
    assert bind_candidate(binding, response, {"ent-service"})[0].mechanism == candidate["mechanism"]
    assert compatible_classes("unclassified application fault") == ()
    assert set(compatible_classes("memory pressure causing GC pauses")) == {"memory_pressure", "gc_pause"}


def test_all_original_candidates_reach_planner_without_model_renaming():
    runner, discovery, result = run_bound()
    assert discovery.calls == ["h1", "h2"]
    assert result["registered"] == 2
    drafts = [e.payload["draft"]["mechanism"] for e in runner.transcript.events
              if e.event_type == EventType.MECHANISM_DRAFT_GENERATED]
    assert drafts == ["worker exhaustion mechanism 1", "worker exhaustion mechanism 2"]


def test_failed_candidate_does_not_block_other_and_errors_stay_safe():
    runner, discovery, result = run_bound(fail_first=True)
    assert discovery.calls == ["h1", "h1", "h2"]
    assert result["registered"] == 1
    assert "private provider diagnostic" not in str([e.payload for e in runner.transcript.events])


def test_formalization_capacity_preserves_unprocessed_originals():
    runner, discovery, result = run_bound(count=5)
    assert discovery.calls == ["h1", "h2", "h3", "h4"]
    assert len([e for e in runner.transcript.events
                if e.event_type == EventType.MECHANISM_DRAFT_GENERATED]) == 4
    disposition = next(d for d in result["inquiry_dispositions"] if d["hypothesis_id"] == "h5")
    assert disposition["disposition"] == "deferred"
    assert disposition["reason"] == "FORMALIZATION_CAPACITY"


@pytest.mark.parametrize("target", ["binding", "details", "aggregate", "missing_record"])
def test_bound_replay_rejects_rehashed_tampering(target):
    runner, _, _ = run_bound()
    transcript = Transcript(runner.episode_id)
    changed = False
    for event in runner.transcript.events:
        raw = deepcopy(event.payload)
        if event.event_type == EventType.BOUND_CANDIDATE_FORMALIZATION_RECORDED and not changed:
            if target == "binding":
                raw["binding"]["mechanism"] = "replacement pool cause"
                changed = True
            elif target == "details":
                raw["response"]["details"]["causal_entity_ref"] = "ent-unissued"
                changed = True
            elif target == "missing_record":
                changed = True
                continue
        if event.event_type == EventType.DISCOVERY_HANDOFF_VALIDATED and target == "aggregate":
            raw["payload"]["mechanisms"] = []
        transcript.append(event.event_type, raw)
    with pytest.raises((ReplayValidationError, ValueError)):
        RecordedReplay.verify(transcript.events)


def test_novel_discovery_cannot_replace_original_or_use_unissued_basis():
    candidate = {"hypothesis_id": "h1", "mechanism": "worker exhaustion", "alternative": "pool exhaustion"}
    binding = candidate_binding(candidate, "h-unknown")
    template = TwoStageDiscoverer().discover({"entity_graph": {"nodes": []}}).mechanisms[0]
    details = CandidateDetails.model_validate(template.model_dump(exclude={"mechanism", "source_hypothesis_id"}))
    response = BoundCandidateResponse(status="formalized", details=details,
                                      reason_code="FORMALIZED", reason="Typed original cause.")
    draft, disposition = bind_candidate(binding, response, {"ent-service"})
    base = MechanismDiscoveryPayload(mechanisms=(draft,), inquiry_dispositions=(disposition,))
    novel = MechanismDiscoveryPayload(mechanisms=(draft,), new_mechanisms=(NewMechanismBasis(
        mechanism=draft.mechanism, source_hypothesis_id=draft.source_hypothesis_id,
        basis_observation_refs=("obs-missing",),
    ),))
    assert admit_novel(base, novel, {"hypotheses": [candidate]}, [], {"h-unknown"}, {"ent-service"}) == base
