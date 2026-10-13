"""GT-free fixed-input acceptance for the active query scheduler."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from rca_agent.core.dynamic import EvidenceRegistry
from rca_agent.core.runtime_registry import rca100_evidence_registry
from rca_agent.environment.rca100 import Rca100Environment
from rca_agent.inquiry_policy import InquiryInvestigationPolicy
from rca_agent.investigation.query_strategy import task_score
from rca_agent.langchain_acquisition import InquiryDraft, query_options
from rca_agent.langchain_policy import LangChainPolicyAdapter
from rca_agent.langgraph_orchestrator import LangGraphInvestigation
from rca_agent.models import RequestStopAction
from rca_agent.runner import EpisodeRunner, RecordedReplay, ReplayValidationError
from rca_agent.runner.transcript import EventType, Transcript
from rca_agent.tools.fingerprint import query_fingerprint


def setup_runner(cases_root, *, registry=rca100_evidence_registry, budget=20, inquiry_contract_version=1):
    runner = EpisodeRunner(
        Rca100Environment.from_case(cases_root, "t030", query_budget=budget),
        evidence_registry=registry, investigation_workspace_mode="active",
        inquiry_contract_version=inquiry_contract_version,
    )
    bootstrap = runner.start()
    return runner, bootstrap


def round_record(bootstrap, signals, *, round_number=1, needs_signals=None):
    entity = bootstrap.alert_entity.model_dump(mode="json")
    options = query_options({"bootstrap": bootstrap.model_dump(mode="json")}, [entity])
    choices, selected, needs = [], [], []
    for index, (tool, signal) in enumerate(signals):
        ref, item = next((ref, item) for ref, item in options.items()
                         if item["action"].tool_name.value == tool
                         and getattr(item["action"].request, "signal", "related_entity") == signal)
        need_id = f"n{index}"
        target = needs_signals[index] if needs_signals else signal
        needs.append({
            "need_id": need_id, "hypothesis_id": "h1", "question": f"Measure {target}",
            "expected_if_true": f"{target} rises", "expected_if_alternative": f"{target} stays low",
            "observable_signals": [target], "entity_ref": entity["entity_ref"],
        })
        choices.append({"need_id": need_id, "query_ref": ref})
        selected.append({"need_id": need_id, "query_ref": ref, "signal": signal,
                         "entity_ref": entity["entity_ref"],
                         "action": item["action"].model_dump(mode="json")})
    record = {"round": round_number, "draft": {
        "hypotheses": [{"hypothesis_id": "h1", "mechanism": "workload change",
                        "alternative": "downstream delay"}], "needs": needs, "choices": choices,
    }, "selected_actions": selected}
    return record, [item["action"] for ref, item in options.items()
                    if any(c["query_ref"] == ref for c in choices)]


def offered_actions(record):
    from rca_agent.models import CallToolAction

    return [CallToolAction.model_validate(item["action"]) for item in record["selected_actions"]]


def finish(runner):
    runner.handle(RequestStopAction(summary="fixed-input scheduler acceptance"))
    assert RecordedReplay.verify(runner.transcript.events).valid


def test_both_useful_queries_execute_and_second_is_not_suppressed(cases_root):
    runner, bootstrap = setup_runner(cases_root)
    record, _ = round_record(bootstrap, [
        ("compare_metric_windows", "request_latency_seconds"),
        ("compare_metric_windows", "request_count"),
    ])
    runner.record_inquiry_round(record)
    first, second = offered_actions(record)
    routed = runner.route_active_inquiry_action(first)
    assert routed.request.signal == "request_count"  # useful > symptom, despite offer order
    runner.handle(routed)
    pending = [t for t in runner.investigation_workspace_snapshot.tasks if t.status.value == "open"]
    assert len(pending) == 1
    routed = runner.route_active_inquiry_action(second)
    assert routed.request.signal == "request_latency_seconds"
    runner.handle(routed)
    selections = [e.payload for e in runner.transcript.events
                  if e.event_type == EventType.DISCRIMINATION_ACTIVE_SELECTION_RECORDED]
    assert len(selections) == 2
    assert all(not s["suppressed_query_fingerprints"] for s in selections)
    assert len(selections[0]["pending_query_fingerprints"]) == 1
    assert selections[1]["pending_query_fingerprints"] == []
    finish(runner)


def test_context_only_query_runs_without_any_evidence_rules(cases_root):
    runner, bootstrap = setup_runner(cases_root, registry=EvidenceRegistry(signals=(), rules=()))
    record, _ = round_record(bootstrap, [("get_metric_series", "request_count")])
    runner.record_inquiry_round(record)
    task = runner.investigation_workspace_snapshot.tasks[0]
    assert task.capability_options[0].rule_id == "query_only"
    assert not task.capability_options[0].support_eligible
    assert not task.capability_options[0].refutes
    action = runner.route_active_inquiry_action(offered_actions(record)[0])
    runner.handle(action)
    assert not runner.dynamic_candidate_options
    finish(runner)


def test_need_query_mismatch_is_exploration_not_a_discriminator(cases_root):
    runner, bootstrap = setup_runner(cases_root)
    record, _ = round_record(bootstrap, [("compare_metric_windows", "request_count")],
                             needs_signals=["sql_duration_ms"])
    runner.record_inquiry_round(record)
    task = runner.investigation_workspace_snapshot.tasks[0]
    assert task.scheduling["alignment"] == "mismatch"
    matrix = runner.investigation_workspace_snapshot.discrimination_matrix
    # Selected tasks no longer occupy the pending Matrix; inspect recorded attempt reasons.
    assert not matrix.entries
    assert "NEED_QUERY_MISMATCH" in task.attempts[0].selection_reason_codes
    assert "EXPLICIT_CONTRASTING_PREDICTIONS" not in task.attempts[0].selection_reason_codes
    runner.handle(runner.route_active_inquiry_action(offered_actions(record)[0]))
    finish(runner)


def test_pending_query_survives_new_round(cases_root):
    runner, bootstrap = setup_runner(cases_root)
    record, _ = round_record(bootstrap, [
        ("compare_metric_windows", "request_count"),
        ("compare_metric_windows", "request_latency_seconds"),
    ])
    runner.record_inquiry_round(record)
    runner.handle(runner.route_active_inquiry_action(offered_actions(record)[0]))
    following, _ = round_record(bootstrap, [("get_metric_series", "error_rate")], round_number=2)
    runner.record_inquiry_round(following)
    runner.handle(runner.route_active_inquiry_action(offered_actions(following)[0]))
    remaining = runner.route_active_inquiry_action(RequestStopAction(summary="drain pending"))
    assert remaining is not None
    runner.handle(remaining)
    assert len([e for e in runner.transcript.events if e.event_type == EventType.TOOL_REQUEST]) == 3
    finish(runner)


def test_duplicate_and_budget_cancellations_are_explicit(cases_root):
    runner, bootstrap = setup_runner(cases_root, budget=1)
    record, _ = round_record(bootstrap, [
        ("compare_metric_windows", "request_count"),
        ("get_metric_series", "request_count"),
    ])
    runner.record_inquiry_round(record)
    runner.handle(runner.route_active_inquiry_action(offered_actions(record)[0]))
    assert runner.route_active_inquiry_action(offered_actions(record)[1]) is None
    cancellations = [e.payload for e in runner.transcript.events
                     if e.event_type == EventType.INQUIRY_QUERY_CANCELLED]
    assert cancellations[0]["reason_code"] == "QUERY_BUDGET_UNAVAILABLE"
    finish(runner)

    runner, bootstrap = setup_runner(cases_root)
    record, _ = round_record(bootstrap, [("compare_metric_windows", "request_count")])
    runner.record_inquiry_round(record)
    runner.handle(runner.route_active_inquiry_action(offered_actions(record)[0]))
    record["round"] = 2
    runner.record_inquiry_round(record)
    assert runner.route_active_inquiry_action(offered_actions(record)[0]) is None
    assert any(e.payload.get("reason_code") == "DUPLICATE_EXECUTED_QUERY"
               for e in runner.transcript.events)
    finish(runner)


def test_utility_coverage_novelty_cost_and_risk():
    option = SimpleNamespace(entity_ref="ent-service", independent_group="group-a",
                             query_fingerprint="query-a", query_cost=1)
    info = {"alignment": "matched", "symptom_only": False, "group_already_observed": False,
            "missing_risk": False, "entity_location_gain": False}
    task = SimpleNamespace(scheduling=info, status=SimpleNamespace(value="open"),
                           candidate_track_id="candidate", alternative_id="alternative-a",
                           candidate_prediction="rise", comparator_prediction="normal",
                           capability_options=[option], attempts=[])
    score, _ = task_score(task, option, [task])
    other = deepcopy(task)
    other.alternative_id = "alternative-b"
    assert task_score(task, option, [task, other])[0] > score
    expensive = deepcopy(option)
    expensive.query_cost = 3
    assert task_score(task, expensive, [task])[0] < score
    for field in ("symptom_only", "group_already_observed", "missing_risk"):
        penalized = deepcopy(task)
        penalized.scheduling[field] = True
        assert task_score(penalized, option, [penalized])[0] < score


@pytest.mark.parametrize("tamper", ["alignment", "pending", "execution"])
def test_rehashed_scheduler_tampering_is_rejected(cases_root, tamper):
    runner, bootstrap = setup_runner(cases_root)
    record, _ = round_record(bootstrap, [
        ("compare_metric_windows", "request_count"),
        ("compare_metric_windows", "request_latency_seconds"),
    ])
    runner.record_inquiry_round(record)
    runner.handle(runner.route_active_inquiry_action(offered_actions(record)[0]))
    runner.handle(runner.route_active_inquiry_action(offered_actions(record)[1]))
    finish(runner)
    rebuilt = Transcript(runner.episode_id)
    changed = False
    for event in runner.transcript.events:
        payload = deepcopy(event.payload)
        if not changed and tamper == "alignment" and event.event_type == EventType.DISCRIMINATION_TASK_CREATED:
            payload["scheduling"]["alignment"] = "mismatch"
            changed = True
        elif not changed and tamper == "pending" and event.event_type == EventType.DISCRIMINATION_ACTIVE_SELECTION_RECORDED:
            payload["pending_query_fingerprints"] = []
            changed = True
        elif not changed and tamper == "execution" and event.event_type == EventType.TOOL_REQUEST:
            payload["request"]["signal"] = "error_rate"
            payload["query_ref"] = query_fingerprint(payload["tool_name"], payload["request"])
            changed = True
        rebuilt.append(event.event_type, payload)
    assert changed
    with pytest.raises(ReplayValidationError):
        RecordedReplay.verify(rebuilt.events)


def test_scripted_inquiry_uses_two_queries_in_one_round_without_a_model(cases_root):
    class FixedInquiry:
        contract_version = 1
        calls = 0

        def plan(self, incident, options):
            self.calls += 1
            bootstrap = holder["bootstrap"]
            record, _ = round_record(bootstrap, [
                ("compare_metric_windows", "request_latency_seconds"),
                ("compare_metric_windows", "request_count"),
            ])
            draft = InquiryDraft.model_validate(record["draft"])
            return draft, tuple(options[c.query_ref]["action"] for c in draft.choices)

    inquiry = FixedInquiry()
    policy = InquiryInvestigationPolicy(inquiry, query_limit=2)
    runner = EpisodeRunner(Rca100Environment.from_case(cases_root, "t030"),
                           evidence_registry=rca100_evidence_registry,
                           investigation_workspace_mode="active", inquiry_contract_version=1)
    holder = {"bootstrap": Rca100Environment.from_case(cases_root, "t030").reset()}
    report = LangGraphInvestigation(
        runner, LangChainPolicyAdapter(policy), inquiry_rounds_provider=lambda: tuple(policy.inquiry_drafts),
    ).run()
    assert inquiry.calls == 1
    assert len(policy.executed_actions) == 2
    assert report.answer.verify_digest()
    assert RecordedReplay.verify(runner.transcript.events).valid
