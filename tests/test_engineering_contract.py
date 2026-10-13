"""Deterministic contract regressions; no provider or evaluator."""
import json
from copy import deepcopy

import pytest
from langchain_core.runnables import RunnableLambda
from test_query_strategy import finish, offered_actions, round_record
from test_query_strategy import setup_runner as legacy_setup_runner

from rca_agent.investigation.inquiry_contract import measurement_target, selected_bindings
from rca_agent.langchain_acquisition import LangChainInquiryAdapter, query_options
from rca_agent.models import CallToolAction
from rca_agent.runner import RecordedReplay, ReplayValidationError
from rca_agent.runner.transcript import EventType, Transcript


def setup_runner(cases_root):
    return legacy_setup_runner(cases_root, inquiry_contract_version=2)


def model_response(draft, payload):
    result = deepcopy(draft)
    menu = json.loads(payload["context"])["queries"]
    for choice in result["choices"]:
        target = menu[choice["query_ref"]]
        choice.update(mode="exploratory", purpose="Inspect the declared measured context",
                      measurement_target={k: target[k] for k in
                                          ("entity_ref", "signal", "measurement_kind", "windows")})
    return result


def upgrade(record, bootstrap, *, mode="direct"):
    record = deepcopy(record)
    record["draft"]["contract_version"] = 2
    from rca_agent.investigation.inquiry_contract import original_alternative_id
    for hypothesis in record["draft"]["hypotheses"]:
        hypothesis["alternative_id"] = original_alternative_id(hypothesis)
    for choice in record["draft"]["choices"]:
        selected = next(s for s in record["selected_actions"]
                        if s["need_id"] == choice["need_id"] and s["query_ref"] == choice["query_ref"])
        choice.update(mode=mode, purpose="Inspect declared measurements",
                      measurement_target=measurement_target(CallToolAction.model_validate(selected["action"])))
    from rca_agent.langchain_acquisition import InquiryDraft
    options = query_options({"bootstrap": bootstrap.model_dump(mode="json")},
                            [bootstrap.alert_entity.model_dump(mode="json")])
    draft = InquiryDraft.model_validate(record["draft"])
    actions = [CallToolAction.model_validate(s["action"]) for s in record["selected_actions"]]
    record["selected_actions"] = selected_bindings(draft, options, actions, record["round"])
    return record


def test_same_need_two_queries_survive_runner_and_replay(cases_root):
    runner, bootstrap = setup_runner(cases_root)
    record, _ = round_record(bootstrap, [
        ("get_metric_series", "request_count"),
        ("compare_metric_windows", "request_count"),
    ])
    record["draft"]["needs"] = record["draft"]["needs"][:1]
    record["draft"]["choices"][1]["need_id"] = "n0"
    record["selected_actions"][1]["need_id"] = "n0"
    record = upgrade(record, bootstrap)
    runner.record_inquiry_round(record)
    for action in offered_actions(record):
        runner.handle(runner.route_active_inquiry_action(action))
    tasks = runner.investigation_workspace_snapshot.tasks
    assert len(tasks) == 2
    assert all(t.attempts[0].completed_sequence for t in tasks)
    assert all(t.scheduling["measured_signals"] == ["request_count"] for t in tasks)
    finish(runner)


def test_shared_query_keeps_both_declared_bindings(cases_root):
    _, bootstrap = setup_runner(cases_root)
    record, _ = round_record(bootstrap, [("get_metric_series", "request_count")])
    draft = deepcopy(record["draft"])
    draft["needs"].append({**draft["needs"][0], "need_id": "n1"})
    draft["choices"] = [{"need_id": name, "query_ref": "q1"} for name in ("n0", "n1")]
    options = query_options({"bootstrap": bootstrap.model_dump(mode="json")},
                            [bootstrap.alert_entity.model_dump(mode="json")])
    parsed, actions = LangChainInquiryAdapter(RunnableLambda(lambda p: model_response(draft, p))).plan({}, options)
    assert len(actions) == 1
    assert len(parsed.choices) == 2


def test_menu_discloses_real_measurement(cases_root):
    _, bootstrap = setup_runner(cases_root)
    record, _ = round_record(bootstrap, [("get_metric_series", "request_count")])
    seen = []
    def respond(payload):
        seen.append(json.loads(payload["context"])["queries"])
        draft = deepcopy(record["draft"])
        draft["choices"][0]["query_ref"] = "q1"
        return model_response(draft, payload)
    options = query_options({"bootstrap": bootstrap.model_dump(mode="json")},
                            [bootstrap.alert_entity.model_dump(mode="json")])
    LangChainInquiryAdapter(RunnableLambda(respond)).plan({}, options)
    for item in seen[0].values():
        assert item["measurement_kind"] in {"series", "window_comparison", "entity_exploration"}
        assert "return_shape" in item and "unit" in item and "windows" in item


def test_inquiry_rejects_exact_self_alternative(cases_root):
    _, bootstrap = setup_runner(cases_root)
    record, _ = round_record(bootstrap, [("get_metric_series", "request_count")])
    draft = record["draft"]
    draft["hypotheses"][0]["alternative"] = "  WORKLOAD   change "
    draft["choices"][0]["query_ref"] = "q1"
    options = query_options({"bootstrap": bootstrap.model_dump(mode="json")},
                            [bootstrap.alert_entity.model_dump(mode="json")])
    with pytest.raises(ValueError, match="SELF_ALTERNATIVE"):
        LangChainInquiryAdapter(RunnableLambda(lambda _: draft)).plan({}, options)


def test_shared_physical_query_completes_two_distinct_tasks_once(cases_root):
    runner, bootstrap = setup_runner(cases_root)
    record, _ = round_record(bootstrap, [("get_metric_series", "request_count")])
    record["draft"]["needs"].append({**record["draft"]["needs"][0], "need_id": "n1"})
    record["draft"]["choices"].append({**record["draft"]["choices"][0], "need_id": "n1"})
    record["selected_actions"].append({**record["selected_actions"][0], "need_id": "n1"})
    record = upgrade(record, bootstrap)
    runner.record_inquiry_round(record)
    response = runner.handle(runner.route_active_inquiry_action(offered_actions(record)[0]))
    assert response.cost.units == 1
    assert len([e for e in runner.transcript.events if e.event_type == EventType.TOOL_REQUEST]) == 1
    tasks = runner.investigation_workspace_snapshot.tasks
    assert len(tasks) == 2 and len({t.source_binding_id for t in tasks}) == 2
    assert all(t.status.value in {"ambiguous", "unverifiable"} for t in tasks)
    assert all(t.attempts[0].observation_refs == tasks[0].attempts[0].observation_refs for t in tasks)
    finish(runner)


@pytest.mark.parametrize("change,code", [
    ("entity", "DIRECT_ENTITY_MISMATCH"), ("signal", "DIRECT_SIGNAL_MISMATCH"),
    ("measurement", "DIRECT_MEASUREMENT_MISMATCH"), ("windows", "DIRECT_WINDOW_MISMATCH"),
    ("target", "MEASUREMENT_TARGET_MISMATCH"), ("mode", "BINDING_MODE_REQUIRED"),
])
def test_adapter_rejects_explicit_mismatch_before_acceptance(cases_root, change, code):
    from rca_agent.inquiry_policy import InquiryInvestigationPolicy, InquiryUnavailableError
    runner, bootstrap = setup_runner(cases_root)
    record, _ = round_record(bootstrap, [("get_metric_series", "request_count")])
    record = upgrade(record, bootstrap)
    draft = record["draft"]
    need, choice = draft["needs"][0], draft["choices"][0]
    if change == "entity":
        need["entity_ref"] = "ent-another"
    elif change == "signal":
        need["observable_signals"] = ["sql_duration_ms"]
    elif change == "measurement":
        need["measurement_kind"] = "window_comparison"
    elif change == "windows":
        need["windows"] = {"series": choice["measurement_target"]["windows"]["series"].copy()}
        need["windows"]["series"]["start"] = need["windows"]["series"]["end"]
        # A valid narrower incident window cannot cover the full series.
        need["windows"]["series"] = bootstrap.observation_horizon.model_copy(update={
            "start": bootstrap.observation_horizon.start +
            (bootstrap.observation_horizon.end - bootstrap.observation_horizon.start) / 2,
        }).model_dump(mode="json")
    elif change == "target":
        choice["measurement_target"]["signal"] = "sql_duration_ms"
    else:
        choice.pop("mode")
    options = query_options({"bootstrap": bootstrap.model_dump(mode="json")},
                            [bootstrap.alert_entity.model_dump(mode="json")])
    alias = "q" + str(list(options).index(choice["query_ref"]) + 1)
    choice["query_ref"] = alias
    policy = InquiryInvestigationPolicy(LangChainInquiryAdapter(RunnableLambda(lambda _: draft)))
    policy.selection_calls = 1
    with pytest.raises(InquiryUnavailableError):
        policy._plan_inquiry_round({}, options)
    diagnostics = policy.drain_inquiry_diagnostics()
    assert len(diagnostics) == 2 and all(d["code"] == code for d in diagnostics)
    assert all(d["fields"] and d["stage"] == "inquiry_contract" for d in diagnostics)
    assert not policy.inquiry_drafts and not policy.selection_queue
    assert not runner.investigation_workspace_snapshot.tasks


@pytest.mark.parametrize("workspace_mode", ["shadow", "active"])
def test_exploratory_binding_is_accepted_without_direct_coverage(cases_root, workspace_mode):
    from rca_agent.core.runtime_registry import rca100_evidence_registry
    from rca_agent.environment import Rca100Environment
    from rca_agent.runner import EpisodeRunner
    runner = EpisodeRunner(Rca100Environment.from_case(cases_root, "t030"),
                           evidence_registry=rca100_evidence_registry,
                           investigation_workspace_mode=workspace_mode)
    bootstrap = runner.start()
    record, _ = round_record(bootstrap, [("get_metric_series", "request_count")],
                             needs_signals=["sql_duration_ms"])
    record = upgrade(record, bootstrap, mode="exploratory")
    runner.record_inquiry_round(record)
    action = offered_actions(record)[0]
    runner.handle(runner.route_active_inquiry_action(action))
    task = runner.investigation_workspace_snapshot.tasks[0]
    assert task.scheduling["alignment"] == "exploratory"
    assert task.status.value != "resolved"
    assert not any("COVERAGE" in reason for attempt in task.attempts for reason in attempt.selection_reason_codes)
    finish(runner)


@pytest.mark.parametrize("field", ["binding_id", "mode", "entity_ref", "signal", "query_ref", "target"])
def test_new_bindings_reject_rehashed_tampering(cases_root, field):
    runner, bootstrap = setup_runner(cases_root)
    record, _ = round_record(bootstrap, [("get_metric_series", "request_count")])
    runner.record_inquiry_round(upgrade(record, bootstrap))
    runner.handle(runner.route_active_inquiry_action(offered_actions(record)[0]))
    finish(runner)
    rebuilt = Transcript(runner.episode_id)
    for event in runner.transcript.events:
        payload = deepcopy(event.payload)
        if event.event_type == EventType.INQUIRY_ROUND_RECORDED:
            binding = payload["selected_actions"][0]
            if field == "target":
                binding["measurement_target"]["signal"] = "sql_duration_ms"
            else:
                binding[field] = "exploratory" if field == "mode" else "tampered"
        rebuilt.append(event.event_type, payload)
    with pytest.raises((ReplayValidationError, ValueError)):
        RecordedReplay.verify(rebuilt.events)


def test_direct_multisignal_need_records_only_measured_subset(cases_root):
    runner, bootstrap = setup_runner(cases_root)
    record, _ = round_record(bootstrap, [("get_metric_series", "request_count")])
    record["draft"]["needs"][0]["observable_signals"].append("error_rate")
    runner.record_inquiry_round(upgrade(record, bootstrap))
    runner.handle(runner.route_active_inquiry_action(offered_actions(record)[0]))
    task = runner.investigation_workspace_snapshot.tasks[0]
    assert task.scheduling["requested_signals"] == ["error_rate", "request_count"]
    assert task.scheduling["measured_signals"] == ["request_count"]
    assert task.status.value != "resolved"
    finish(runner)


def test_menu_projection_matches_typed_windows_units_and_order(cases_root):
    from rca_agent.investigation.inquiry_contract import public_query
    _, bootstrap = setup_runner(cases_root)
    options = query_options({"bootstrap": bootstrap.model_dump(mode="json")},
                            [bootstrap.alert_entity.model_dump(mode="json")])
    public = [public_query(item) for item in options.values()]
    descriptors = {(cap.tool_name, metric.signal): metric for cap in bootstrap.capabilities for metric in cap.metrics}
    for item, (_, option) in zip(public, options.items(), strict=True):
        action = option["action"]
        assert item["tool_name"] == action.tool_name.value
        assert item["entity_ref"] == action.request.entity_ref
        if item["measurement_kind"] == "entity_exploration":
            continue
        assert item["unit"] == (descriptors[(item["tool_name"], item["signal"])].unit or "unknown")
        for window in item["windows"].values():
            assert window["start"].endswith("Z") or "+" in window["start"]
        if item["measurement_kind"] == "window_comparison":
            from rca_agent.models import TimeWindow
            incident = TimeWindow.model_validate(item["windows"]["incident"])
            baseline = TimeWindow.model_validate(item["windows"]["baseline"])
            assert baseline.end == incident.start
            assert baseline.end - baseline.start == incident.end - incident.start
    counts = [i for i in public if i["signal"] == "request_count"]
    assert {i["return_shape"] for i in counts} == {"samples", "window statistics and ratio"}
    serialized = json.dumps(public)
    assert not any(token in serialized for token in ("/Users/", "case_id", "threshold", "ground_truth", "trace-"))


def test_duplicate_choices_deduplicate_without_refilling_or_losing_shared_need(cases_root):
    _, bootstrap = setup_runner(cases_root)
    record, _ = round_record(bootstrap, [("get_metric_series", "request_count")])
    options = query_options({"bootstrap": bootstrap.model_dump(mode="json")},
                            [bootstrap.alert_entity.model_dump(mode="json")])
    draft = record["draft"]
    draft["choices"] = [{"need_id": "n0", "query_ref": "q1"}] * 2 + [{"need_id": "n0", "query_ref": "invented"}]
    adapter = LangChainInquiryAdapter(RunnableLambda(lambda p: model_response(draft, p)))
    # The fixture adds target data only to the first two choices, just as live admission.
    def respond(payload):
        response = model_response({**draft, "choices": draft["choices"][:2]}, payload)
        response["choices"].append({**response["choices"][0], **draft["choices"][2]})
        return response
    adapter.runnable = RunnableLambda(respond)
    parsed, actions = adapter.plan({}, options)
    assert len(parsed.choices) == len(actions) == 1


def test_conflicting_duplicate_binding_is_rejected(cases_root):
    _, bootstrap = setup_runner(cases_root)
    record, _ = round_record(bootstrap, [("get_metric_series", "request_count")])
    options = query_options({"bootstrap": bootstrap.model_dump(mode="json")},
                            [bootstrap.alert_entity.model_dump(mode="json")])
    draft = record["draft"]
    draft["choices"] = [{"need_id": "n0", "query_ref": "q1"}] * 2
    def respond(payload):
        result = model_response(draft, payload)
        result["choices"] = [deepcopy(c) for c in result["choices"]]
        result["choices"][1]["purpose"] = "A conflicting declared purpose"
        return result
    with pytest.raises(ValueError, match="CONFLICTING_BINDING"):
        LangChainInquiryAdapter(RunnableLambda(respond)).plan({}, options)


@pytest.mark.parametrize("mode", ["shadow", "active"])
def test_runner_rejects_same_invalid_direct_in_both_modes(cases_root, mode):
    from rca_agent.environment import Rca100Environment
    from rca_agent.runner import EpisodeRunner
    runner = EpisodeRunner(Rca100Environment.from_case(cases_root, "t030"), investigation_workspace_mode=mode)
    bootstrap = runner.start()
    record, _ = round_record(bootstrap, [("get_metric_series", "request_count")], needs_signals=["sql_duration_ms"])
    with pytest.raises(ValueError, match="DIRECT_SIGNAL_MISMATCH"):
        runner.record_inquiry_round(upgrade(record, bootstrap))
    assert not any(e.event_type == EventType.INQUIRY_ROUND_RECORDED for e in runner.transcript.events)


def test_new_live_schema_requires_binding_declarations_and_legacy_omits_new_fields():
    from rca_agent.langchain_acquisition import InquiryDraft, InquiryDraftV2
    schema = InquiryDraftV2.model_json_schema()
    assert {"mode", "purpose", "measurement_target"} <= set(schema["$defs"]["InquiryChoiceV2"]["required"])
    legacy = {"hypotheses": [{"hypothesis_id": "h1", "mechanism": "cause", "alternative": "cause"}],
              "needs": [{"need_id": "n1", "hypothesis_id": "h1", "question": "q",
                         "expected_if_true": "yes", "expected_if_alternative": "no"}],
              "choices": [{"need_id": "n1", "query_ref": "q1"}]}
    assert InquiryDraft.model_validate(legacy).model_dump(mode="json") == legacy


def test_direct_wrong_issued_entity_rejected_but_exploration_allowed(cases_root):
    _, bootstrap = setup_runner(cases_root)
    options = query_options({"bootstrap": bootstrap.model_dump(mode="json")}, [
        bootstrap.alert_entity.model_dump(mode="json"),
        {**bootstrap.alert_entity.model_dump(mode="json"), "entity_ref": "ent-second"},
    ])
    alias = next("q" + str(i) for i, (_, option) in enumerate(options.items(), 1)
                 if option["action"].request.entity_ref == "ent-second")
    record, _ = round_record(bootstrap, [("get_metric_series", "request_count")])
    draft = record["draft"]
    draft["choices"][0]["query_ref"] = alias
    def direct(payload):
        result = model_response(draft, payload)
        result["choices"][0]["mode"] = "direct"
        result["needs"][0]["observable_signals"] = [result["choices"][0]["measurement_target"]["signal"]]
        return result
    with pytest.raises(ValueError, match="DIRECT_ENTITY_MISMATCH"):
        LangChainInquiryAdapter(RunnableLambda(direct)).plan({}, options)
    parsed, actions = LangChainInquiryAdapter(RunnableLambda(lambda p: model_response(draft, p))).plan({}, options)
    assert parsed.choices[0].mode == "exploratory" and actions[0].request.entity_ref == "ent-second"
