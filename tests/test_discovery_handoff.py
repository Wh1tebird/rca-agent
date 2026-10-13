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

from rca_agent.core.dynamic import (
    AlternativeDraft,
    InquiryDisposition,
    MechanismDiscoveryPayload,
    MechanismDraft,
    NewMechanismBasis,
    VerificationPlanPayload,
)
from rca_agent.investigation.discovery_handoff import (
    merge_partial_handoff,
    pending_formalizations,
    quarantine_handoff,
    repair_errors,
    validate_handoff,
)
from rca_agent.langchain_acquisition import (
    InquiryHandoff,
    InquiryHypothesis,
    inquiry_candidate_records,
)
from rca_agent.models import RequestStopAction
from rca_agent.runner import EpisodeRunner, RecordedReplay, ReplayValidationError
from rca_agent.runner.transcript import EventType, Transcript

ORIGINAL = "complex cross join causes slow SQL"
ALTERNATIVE = "database connection pool exhaustion"
FACTS = [{"observation_id": "obs-one", "data_status": "observed", "value": 2.8}]
HANDOFF = {"hypotheses": [{"hypothesis_id": "h1", "mechanism": ORIGINAL,
                            "alternative": ALTERNATIVE}]}


def mechanism(text=ORIGINAL):
    return MechanismDraft(
        source_hypothesis_id="h-unknown", causal_entity_ref="ent-service",
        mechanism=text, mechanism_class="database_execution", family="database_dependency",
        causal_explanation="The measured SQL client span dominates operation latency.",
        alternatives=(AlternativeDraft(alternative_id="alt-pool", mechanism=ALTERNATIVE,
                                       mechanism_class="database_connection_pool"),),
    )


def disposition(**changes):
    return InquiryDisposition.model_validate({
        "hypothesis_id": "h1", "disposition": "kept", "reason": "Preserve original causal idea.",
        "formal_mechanism": ORIGINAL, **changes,
    })


def payload(d=None, mechanisms=None, **changes):
    return MechanismDiscoveryPayload(mechanisms=mechanisms or (mechanism(),),
                                     inquiry_dispositions=(d or disposition(),), **changes)


def codes(p, facts=FACTS):
    return {code for code, _, _ in validate_handoff(p, HANDOFF, facts)}


def test_kept_cannot_promote_its_alternative():
    bad = payload(disposition(formal_mechanism=ALTERNATIVE), (mechanism(ALTERNATIVE),))
    assert "KEPT_CAUSAL_SEMANTICS_CHANGED" in codes(bad)
    safe = quarantine_handoff(bad, HANDOFF, FACTS)
    assert not safe.mechanisms
    assert safe.inquiry_dispositions[0].disposition == "deferred"
    assert not codes(safe)


def test_exact_original_draft_survives_bad_disposition_without_semantic_rewrite():
    bad = payload(disposition(formal_mechanism=ALTERNATIVE, original_disposition="retained"),
                  (mechanism(ALTERNATIVE), mechanism()))
    safe = quarantine_handoff(bad, HANDOFF, FACTS, recover_exact=True)
    assert safe.mechanisms == (mechanism(),)
    assert safe.inquiry_dispositions[0].formal_mechanism == ORIGINAL
    assert safe.inquiry_dispositions[0].disposition == "kept"
    assert not codes(safe)
    # Historical v2 keeps its original fail-closed result.
    assert not quarantine_handoff(bad, HANDOFF, FACTS).mechanisms
    dropping = MechanismDiscoveryPayload(inquiry_dispositions=(disposition(
        disposition="deferred", formal_mechanism=None),))
    assert "REPAIR_CHANGED_VALID_MECHANISM" in {
        e[0] for e in repair_errors(bad, dropping, HANDOFF, FACTS, recover_exact=True)
    }


@pytest.mark.parametrize("kind", ["discarded", "revised"])
def test_exact_link_recovery_does_not_override_explicit_rejection_or_revision(kind):
    bad = payload(disposition(disposition=kind, formal_mechanism=None))
    safe = quarantine_handoff(bad, HANDOFF, FACTS, recover_exact=True)
    assert not safe.mechanisms


def test_exact_link_recovery_does_not_guess_paraphrases_or_duplicate_drafts():
    bad = payload(disposition(formal_mechanism=ALTERNATIVE),
                  (mechanism("a complex query is slow"),))
    assert not quarantine_handoff(bad, HANDOFF, FACTS, recover_exact=True).mechanisms
    bad = payload(disposition(formal_mechanism=ALTERNATIVE), (mechanism(), mechanism()))
    assert not quarantine_handoff(bad, HANDOFF, FACTS, recover_exact=True).mechanisms


def test_partial_retry_adds_independent_valid_candidate_and_preserves_initial_draft():
    handoff = deepcopy(HANDOFF)
    handoff["hypotheses"].append({"hypothesis_id": "h2", "mechanism": "GC pause", "alternative": "CPU load"})
    initial = payload().model_copy(update={"inquiry_dispositions": (
        disposition(), disposition(hypothesis_id="h2", formal_mechanism="not submitted"),
    )})
    retry = payload(mechanisms=(mechanism(ALTERNATIVE), mechanism("GC pause"))).model_copy(update={
        "inquiry_dispositions": (
            disposition(formal_mechanism=ALTERNATIVE),
            disposition(hypothesis_id="h2", formal_mechanism="GC pause"),
        ),
    })
    merged = merge_partial_handoff(initial, retry, handoff, FACTS)
    assert merged.mechanisms == (mechanism(), mechanism("GC pause"))
    assert not validate_handoff(merged, handoff, FACTS)
    assert not pending_formalizations(merged, handoff)


def test_partial_retry_cannot_override_valid_discard_or_rewrite_first_draft():
    initial = payload(disposition(disposition="discarded", formal_mechanism=None,
                                  basis_observation_refs=("obs-one",)))
    initial = initial.model_copy(update={"mechanisms": ()})
    merged = merge_partial_handoff(initial, payload(), HANDOFF, FACTS)
    assert not merged.mechanisms
    assert merged.inquiry_dispositions[0].disposition == "discarded"
    rewritten = payload(mechanisms=(mechanism().model_copy(update={"causal_entity_ref": "ent-other"}),))
    assert merge_partial_handoff(payload(), rewritten, HANDOFF, FACTS).mechanisms == payload().mechanisms


def test_partial_retry_accepts_provenanced_new_mechanism_but_not_unbound_one():
    deferred = MechanismDiscoveryPayload(inquiry_dispositions=(disposition(
        disposition="deferred", formal_mechanism=None),))
    new = mechanism("transport delay")
    retry = deferred.model_copy(update={"mechanisms": (new,), "new_mechanisms": (
        NewMechanismBasis(mechanism=new.mechanism, source_hypothesis_id=new.source_hypothesis_id,
                          basis_observation_refs=("obs-one",)),
    )})
    assert merge_partial_handoff(deferred, retry, HANDOFF, FACTS).mechanisms == (new,)
    retry = retry.model_copy(update={"new_mechanisms": ()})
    assert not merge_partial_handoff(deferred, retry, HANDOFF, FACTS).mechanisms


def test_partial_retry_can_complete_previously_deferred_candidate():
    deferred = MechanismDiscoveryPayload(inquiry_dispositions=(disposition(
        disposition="deferred", formal_mechanism=None),))
    assert merge_partial_handoff(deferred, payload(), HANDOFF, FACTS).mechanisms == payload().mechanisms


def test_exact_kept_with_whitespace_and_case_only_is_valid():
    text = " COMPLEX   CROSS JOIN CAUSES SLOW SQL "
    assert not codes(payload(disposition(formal_mechanism=text), (mechanism(text),)))


@pytest.mark.parametrize("change,code", [
    ({"hypothesis_id": "unknown"}, "UNKNOWN_INQUIRY_CANDIDATE"),
    ({"formal_mechanism": "not submitted"}, "FORMAL_MECHANISM_NOT_UNIQUE"),
    ({"disposition": "discarded", "formal_mechanism": None}, "MISSING_OBSERVATION_BASIS"),
    ({"disposition": "deferred"}, "INACTIVE_CANDIDATE_HAS_FORMAL_LINK"),
    ({"basis_observation_refs": ["obs-unissued"]}, "INVALID_OBSERVATION_BASIS"),
    ({"original_disposition": "deferred"}, "UNEXPECTED_ORIGINAL_DISPOSITION"),
])
def test_disposition_contract_errors(change, code):
    assert code in codes(payload(disposition(**change)))


def test_every_candidate_accounted_exactly_once():
    p = payload().model_copy(update={"inquiry_dispositions": ()})
    assert "CANDIDATE_NOT_ACCOUNTED_ONCE" in codes(p)
    p = payload().model_copy(update={"inquiry_dispositions": (disposition(), disposition())})
    assert "CANDIDATE_NOT_ACCOUNTED_ONCE" in codes(p)


@pytest.mark.parametrize("status", ["missing", "unknown", "model_inferred", "imputed"])
def test_missing_or_inferred_fact_cannot_justify_discard_or_revision(status):
    d = disposition(disposition="revised", formal_mechanism=ALTERNATIVE,
                    basis_observation_refs=["obs-one"], original_disposition="deferred")
    assert "INVALID_OBSERVATION_BASIS" in codes(
        payload(d, (mechanism(ALTERNATIVE),)), [{**FACTS[0], "data_status": status}],
    )


@pytest.mark.parametrize("original_state", ["retained", "deferred"])
def test_revision_accounts_for_original_explicitly(original_state):
    d = disposition(disposition="revised", formal_mechanism=ALTERNATIVE,
                    basis_observation_refs=["obs-one"], original_disposition=original_state)
    drafts = (mechanism(ALTERNATIVE),) + ((mechanism(),) if original_state == "retained" else ())
    assert not codes(payload(d, drafts))
    if original_state == "retained":
        assert "REVISION_DROPPED_ORIGINAL" in codes(payload(d, (mechanism(ALTERNATIVE),)))


def test_new_mechanism_is_additive_and_requires_issued_basis():
    new = mechanism("network transport delay")
    p = payload(mechanisms=(mechanism(), new))
    assert "MECHANISM_WITHOUT_PROVENANCE" in codes(p)
    p = p.model_copy(update={"new_mechanisms": (NewMechanismBasis(
        mechanism=new.mechanism, source_hypothesis_id=new.source_hypothesis_id,
        basis_observation_refs=("obs-one",),
    ),)})
    assert not codes(p)
    p = p.model_copy(update={"inquiry_dispositions": ()})
    assert "CANDIDATE_NOT_ACCOUNTED_ONCE" in codes(p)


def test_quarantine_keeps_valid_independent_candidate_and_repair_cannot_drop_it():
    handoff = deepcopy(HANDOFF)
    handoff["hypotheses"].append({"hypothesis_id": "h2", "mechanism": "GC pause",
                                   "alternative": "memory allocation"})
    good = mechanism("GC pause")
    bad = payload(disposition(formal_mechanism=ALTERNATIVE), (mechanism(ALTERNATIVE), good))
    bad = bad.model_copy(update={"inquiry_dispositions": (*bad.inquiry_dispositions,
        disposition(hypothesis_id="h2", formal_mechanism="GC pause"))})
    safe = quarantine_handoff(bad, handoff, FACTS)
    assert safe.mechanisms == (good,)
    assert not validate_handoff(safe, handoff, FACTS)
    repaired = payload().model_copy(update={"inquiry_dispositions": (*payload().inquiry_dispositions,
        disposition(hypothesis_id="h2", disposition="deferred", formal_mechanism=None))})
    assert "REPAIR_CHANGED_VALID_MECHANISM" in {e[0] for e in repair_errors(bad, repaired, handoff, FACTS)}


class ScriptedDiscovery:
    def __init__(self, mode):
        self.calls = 0
        self.mode = mode

    def discover(self, context):
        self.calls += 1
        original = context["inquiry_handoff"]["hypotheses"][0]["mechanism"]
        template = TwoStageDiscoverer().discover(context).mechanisms[0]
        if self.mode == "valid" or self.calls == 2 and self.mode == "repair":
            text = original
        else:
            text = "connection pool exhaustion"
        assert context["handoff_contract"]["version"] == 4
        if self.calls == 2:
            assert context["handoff_repair"]["errors"][0]["code"] == "KEPT_CAUSAL_SEMANTICS_CHANGED"
        return payload(disposition(formal_mechanism=text), (template.model_copy(update={"mechanism": text}),))


def run_handoff(mode, *, custom_discovery=None, record_round=False, custom_handoff=None):
    discovery, planner = custom_discovery or ScriptedDiscovery(mode), TwoStagePlanner()
    runner = EpisodeRunner(Environment(), inquiry_contract_version=1, discovery=discovery, verification_planner=planner,
                           evidence_registry=registry(), max_rounds=20)
    runner.start()
    runner.handle(query("active_ratio"))
    runner.handle(unknown_action())
    handoff = InquiryHandoff(revision=1, hypotheses=(InquiryHypothesis(
        hypothesis_id="h1", mechanism="thread pool exhaustion", alternative="connection pool exhaustion",
    ),), needs=(), query_links=())
    if custom_handoff is not None:
        handoff = custom_handoff
    if record_round:
        record = {"round": 1, "draft": {
            "hypotheses": [h.model_dump(mode="json") for h in handoff.hypotheses],
            "needs": [{"need_id": "n1", "hypothesis_id": "h1", "question": "Are workers occupied?",
                       "expected_if_true": "active threads increase",
                       "expected_if_alternative": "active threads stay low"}],
            "choices": [{"need_id": "n1", "query_ref": "aq-legacy"}],
        }}
        runner.record_inquiry_round(record)
        handoff = handoff.model_copy(update={"candidate_history": inquiry_candidate_records([record])})
    result = runner.discover_candidates(inquiry_handoff=handoff.model_dump(mode="json"))
    answer = runner.handle(RequestStopAction(summary="semantic handoff test"))
    assert RecordedReplay.verify(runner.transcript.events).valid
    return runner, discovery, planner, result, answer


def test_partially_repaired_candidate_reaches_planner_and_replays():
    class PartialDiscovery:
        def discover(self, context):
            hypotheses = context["inquiry_handoff"]["hypotheses"]
            first, second = [h["mechanism"] for h in hypotheses]
            template = TwoStageDiscoverer().discover(context).mechanisms[0]
            if "handoff_repair" not in context:
                drafts = (template.model_copy(update={"mechanism": first}),)
                links = (disposition(formal_mechanism=first),
                         disposition(hypothesis_id="h2", formal_mechanism=second))
            else:
                assert context["handoff_repair"]["pending_formalizations"][0]["hypothesis_id"] == "h2"
                drafts = (template.model_copy(update={"mechanism": ALTERNATIVE}),
                          template.model_copy(update={"mechanism": second}))
                links = (disposition(formal_mechanism=ALTERNATIVE),
                         disposition(hypothesis_id="h2", formal_mechanism=second))
            return MechanismDiscoveryPayload(mechanisms=drafts, inquiry_dispositions=links)

    handoff = InquiryHandoff(revision=1, hypotheses=(
        InquiryHypothesis(hypothesis_id="h1", mechanism="thread pool exhaustion", alternative=ALTERNATIVE),
        InquiryHypothesis(hypothesis_id="h2", mechanism="worker queue accumulation", alternative="CPU load"),
    ), needs=(), query_links=())
    runner, _, _, result, _ = run_handoff("partial", custom_discovery=PartialDiscovery(), custom_handoff=handoff)
    validated = next(e.payload for e in runner.transcript.events
                     if e.event_type == EventType.DISCOVERY_HANDOFF_VALIDATED)
    assert validated["outcome"] == "partially_repaired"
    assert result["registered"] == 2
    generated = [e.payload["draft"]["mechanism"] for e in runner.transcript.events
                 if e.event_type == EventType.MECHANISM_DRAFT_GENERATED]
    assert generated == ["thread pool exhaustion", "worker queue accumulation"]
    # Rehashing a changed merge outcome cannot bypass semantic replay checks.
    transcript = Transcript(runner.episode_id)
    for event in runner.transcript.events:
        raw = deepcopy(event.payload)
        if event.event_type == EventType.DISCOVERY_HANDOFF_VALIDATED:
            raw["outcome"] = "quarantined"
        transcript.append(event.event_type, raw)
    with pytest.raises(ReplayValidationError):
        RecordedReplay.verify(transcript.events)


def test_recovered_original_reaches_planner_even_when_retry_repeats_bad_link():
    class BadLinkDiscovery:
        def discover(self, context):
            original = context["inquiry_handoff"]["hypotheses"][0]["mechanism"]
            template = TwoStageDiscoverer().discover(context).mechanisms[0]
            if "handoff_repair" in context:
                assert context["handoff_repair"]["safe_projection"]["mechanisms"][0]["mechanism"] == original
            return payload(
                disposition(formal_mechanism=ALTERNATIVE, original_disposition="retained"),
                (template.model_copy(update={"mechanism": original}),
                 template.model_copy(update={"mechanism": ALTERNATIVE})),
            )

    runner, _, _, result, _ = run_handoff("invalid", custom_discovery=BadLinkDiscovery())
    assert result["registered"] == 1
    generated = [e.payload["draft"]["mechanism"] for e in runner.transcript.events
                 if e.event_type == EventType.MECHANISM_DRAFT_GENERATED]
    assert generated == ["thread pool exhaustion"]
    assert any(e.event_type == EventType.VERIFICATION_PLAN_COMPILED for e in runner.transcript.events)


def test_planner_omission_is_retried_and_explicit_for_each_attempt():
    class EmptyPlanner:
        def plan(self, context):
            return VerificationPlanPayload(plans=())

    runner = EpisodeRunner(Environment(), inquiry_contract_version=1, discovery=ScriptedDiscovery("valid"),
                           verification_planner=EmptyPlanner(), evidence_registry=registry())
    runner.start()
    runner.handle(query("active_ratio"))
    runner.handle(unknown_action())
    handoff = InquiryHandoff(revision=1, hypotheses=(InquiryHypothesis(
        hypothesis_id="h1", mechanism="thread pool exhaustion", alternative=ALTERNATIVE,
    ),), needs=(), query_links=())
    result = runner.discover_candidates(inquiry_handoff=handoff.model_dump(mode="json"))
    failures = [e.payload for e in runner.transcript.events
                if e.event_type == EventType.VERIFICATION_PLAN_REJECTED]
    assert [f["attempt"] for f in failures] == [1, 2]
    assert all(f["error_code"] == "MISSING_VERIFICATION_PLAN" for f in failures)
    assert result["registered"] == 0
    runner.handle(RequestStopAction(summary="planner omission recorded"))
    assert RecordedReplay.verify(runner.transcript.events).valid


@pytest.mark.parametrize("original_state", ["retained", "deferred"])
def test_revised_cause_does_not_supersede_original_workspace_or_fake_its_core_link(original_state):
    class RevisionDiscovery:
        def discover(self, context):
            original = TwoStageDiscoverer().discover(context).mechanisms[0]
            revision = original.model_copy(update={"mechanism": "connection pool exhaustion"})
            facts = [o["observation_id"] for o in context["observations"]]
            return payload(disposition(
                disposition="revised", formal_mechanism=revision.mechanism,
                basis_observation_refs=facts, original_disposition=original_state,
            ), (original, revision) if original_state == "retained" else (revision,))

    runner, _, _, _, answer = run_handoff("valid", custom_discovery=RevisionDiscovery(), record_round=True)
    original = answer.investigation_workspace.candidates[0]
    assert original.mechanism_summary == "thread pool exhaustion"
    assert original.stage.value == ("active" if original_state == "retained" else "deferred")
    assert not any(e.event_type.value == "investigation_candidate_superseded" for e in runner.transcript.events)
    outcome = answer.inquiry_candidates[0]
    formalized = {e.payload["mechanism"]["mechanism_id"]: e.payload["mechanism"]["mechanism"]
                  for e in runner.transcript.events if e.event_type == EventType.MECHANISM_DRAFT_FROZEN}
    assert all(formalized[ref] == "thread pool exhaustion" for ref in outcome.formal_mechanism_ids)


def test_failed_retry_preserves_original_workspace_without_formalization():
    runner, *_ = run_handoff("invalid", record_round=True)
    workspace = runner.investigation_workspace_snapshot
    assert workspace.candidates[0].mechanism_summary == "thread pool exhaustion"
    assert workspace.candidates[0].stage.value == "deferred"
    assert workspace.candidates[0].formal_mechanism_id is None


@pytest.mark.parametrize("mode,calls,registered,outcome", [
    ("valid", 1, 1, "accepted"), ("repair", 2, 1, "repaired"),
    ("invalid", 2, 0, "quarantined"),
])
def test_runner_checks_handoff_before_planner(mode, calls, registered, outcome):
    runner, discovery, planner, result, answer = run_handoff(mode)
    assert discovery.calls == calls and result["registered"] == registered
    event = next(e for e in runner.transcript.events if e.event_type == EventType.DISCOVERY_HANDOFF_VALIDATED)
    assert event.payload["outcome"] == outcome
    assert answer.inquiry_candidates[0].mechanism == "thread pool exhaustion"
    if mode == "invalid":
        assert planner.calls == 0
        assert answer.inquiry_candidates[0].discovery_disposition == "deferred"
        assert not answer.inquiry_candidates[0].formal_mechanism_ids


@pytest.mark.parametrize("target", ["initial", "errors", "final", "draft", "disposition", "missing_validation"])
def test_rehashed_handoff_tampering_is_rejected(target):
    runner, *_ = run_handoff("repair")
    rebuilt = Transcript(runner.episode_id)
    for event in runner.transcript.events:
        data = deepcopy(event.payload)
        if event.event_type == EventType.DISCOVERY_HANDOFF_VALIDATED:
            if target == "initial":
                data["initial_payload"]["mechanisms"][0]["mechanism"] = "forged cause"
            elif target == "errors":
                data["retry_errors"] = [{"code": "invented", "hypothesis_id": None, "mechanism": None}]
            elif target == "final":
                data["payload"]["mechanisms"][0]["mechanism"] = "forged cause"
            elif target == "missing_validation":
                continue
        if target == "draft" and event.event_type == EventType.MECHANISM_DRAFT_GENERATED:
            data["draft"]["mechanism"] = "forged cause"
        if target == "disposition" and event.event_type == EventType.DISCOVERY_COMPLETED:
            data["inquiry_dispositions"][0]["formal_mechanism"] = "forged cause"
        rebuilt.append(event.event_type, data)
    with pytest.raises(ReplayValidationError):
        RecordedReplay.verify(rebuilt.events)
