"""Adjacent admission, execution order and explicitly versioned compatibility boundaries."""
from copy import deepcopy

import pytest
from langchain_core.runnables import RunnableLambda
from test_engineering_independent_regressions import execute, output, policy_for, setup

from rca_agent.inquiry_policy import InquiryUnavailableError
from rca_agent.investigation.bound_formalization import bind_candidate, candidate_binding
from rca_agent.investigation.contract_enforcement import execution_binding_responses
from rca_agent.langchain_acquisition import LangChainInquiryAdapter
from rca_agent.models import RequestStopAction
from rca_agent.runner import RecordedReplay
from rca_agent.runner.transcript import EventType, Transcript
from rca_agent.tools.fingerprint import sha256_digest


def rebuild(events, mutate):
    result = Transcript(events[0].episode_id)
    for event in events:
        payload = deepcopy(event.payload)
        mutate(event, payload)
        if event.event_type == EventType.ANSWER_FROZEN:
            payload["transcript_digest"] = result.head_digest
            payload["artifact_digest"] = sha256_digest({
                key: value for key, value in payload.items() if key != "artifact_digest"})
        result.append(event.event_type, payload)
    return result.events


@pytest.mark.parametrize("host", [1, 2])
@pytest.mark.parametrize("version", [999, True, "2"])
def test_adapter_does_not_overwrite_unsupported_model_version(host, version):
    _, _, options = setup("off")

    def response(context):
        raw = output(context, (("n1", "q1"),))
        raw["contract_version"] = version
        return raw

    policy = policy_for(options, (("n1", "q1"),))
    policy.inquiry = LangChainInquiryAdapter(RunnableLambda(response), contract_version=host)
    with pytest.raises(InquiryUnavailableError):
        policy._plan_inquiry_round({}, options)
    assert policy.selection_errors[-1] == "UNSUPPORTED_INQUIRY_VERSION"
    assert policy.drain_inquiry_diagnostics()[-1]["fields"] == ["contract_version"]


@pytest.mark.parametrize("mode", ["off", "shadow", "active"])
def test_raw_runner_choice_cannot_silently_drop_extra_field(mode):
    env, runner, options = setup(mode)
    record = deepcopy(policy_for(options, (("n1", "q1"),)).inquiry_drafts[0])
    record["draft"]["choices"][0]["unexpected"] = "ignored"
    with pytest.raises(ValueError):
        runner.record_inquiry_round(record)
    assert env.calls == []
    assert not any(event.event_type == EventType.INQUIRY_ROUND_RECORDED
                   for event in runner.transcript.events)


def test_raw_fallback_choice_uses_the_same_strict_admission():
    _, _, options = setup("off")

    def response(context):
        raw = output(context, (("n1", "q1"),))
        raw["choices"][0]["unexpected"] = "ignored"
        return {"parsed": None, "raw": type("Raw", (), {
            "content": "", "tool_calls": [{"name": "InquiryDraftV2", "args": raw}]})()}

    with pytest.raises(ValueError):
        LangChainInquiryAdapter(RunnableLambda(response)).plan({}, options)


def test_replay_extra_field_rejected_even_after_rebinding_answer():
    _, runner, options = setup("off")
    execute(runner, policy_for(options, (("n1", "q1"),)))
    runner.handle(RequestStopAction(summary="strict replay fields"))

    def mutate(event, payload):
        if event.event_type == EventType.INQUIRY_ROUND_RECORDED:
            payload["draft"]["choices"][0]["unexpected"] = "ignored"

    with pytest.raises(ValueError):
        RecordedReplay.verify(rebuild(runner.transcript.events, mutate))


def test_declaration_between_request_and_response_is_not_an_execution_binding():
    _, runner, options = setup("off")
    policy = policy_for(options, (("n1", "q1"),))
    execute(runner, policy)
    round_event = next(e for e in runner.transcript.events
                       if e.event_type == EventType.INQUIRY_ROUND_RECORDED)
    request = next(e for e in runner.transcript.events if e.event_type == EventType.TOOL_REQUEST)
    response = next(e for e in runner.transcript.events if e.event_type == EventType.TOOL_RESPONSE)
    events = [request.model_copy(update={"sequence": 1}),
              round_event.model_copy(update={"sequence": 2}),
              response.model_copy(update={"sequence": 3})]
    assert execution_binding_responses(events) == {}


def test_explicit_history_offset_excludes_old_response():
    _, runner, options = setup("off")
    policy = policy_for(options, (("n1", "q1"),))
    history = execute(runner, policy)
    policy._binding_history_offsets[1] = len(history)
    assert policy.discovery_handoff(tuple(history))["query_links"] == []


@pytest.mark.parametrize("change", ["extra", "version"])
def test_old_record_admission_retains_explicit_historical_semantics(change):
    _, runner, options = setup("off")
    policy = policy_for(options, (("n1", "q1"),))
    history = execute(runner, policy)
    if change == "version":
        from test_dynamic_investigation import unknown_action
        runner.handle(unknown_action())
        runner.discover_candidates(inquiry_handoff=policy.discovery_handoff(tuple(history)))
    runner.handle(RequestStopAction(summary="historical revision compatibility"))

    def mutate(event, payload):
        if event.event_type == EventType.EPISODE_CREATED:
            payload.pop("engineering_contract_revision")
            payload.pop("inquiry_contract_version")
        if change == "version" and event.event_type == EventType.BOUND_CANDIDATE_FORMALIZATION_RECORDED:
            payload["binding"].pop("validation_revision")
        if change == "extra" and event.event_type == EventType.INQUIRY_ROUND_RECORDED:
            payload["draft"]["choices"][0]["historical_extension"] = "old reader ignores this"

    assert RecordedReplay.verify(rebuild(runner.transcript.events, mutate)).valid


def test_bound_unknown_versions_do_not_fall_back_to_legacy():
    candidate = {"hypothesis_id": "h1", "mechanism": "worker exhaustion",
                 "alternative": "connection pool exhaustion"}
    with pytest.raises(ValueError):
        candidate_binding(candidate, "h-source", version=999)
    with pytest.raises(ValueError):
        bind_candidate({"contract_version": 999}, {}, set())
    with pytest.raises(ValueError):
        bind_candidate({"contract_version": 7, "validation_revision": 999}, {}, set())


@pytest.mark.parametrize("version", [999, True, "2"])
def test_unsupported_host_version_is_rejected_before_model_binding(version):
    with pytest.raises(ValueError):
        LangChainInquiryAdapter(RunnableLambda(lambda _: {}), contract_version=version)
    with pytest.raises(ValueError):
        LangChainInquiryAdapter.from_chat_model(object(), contract_version=version)


def test_menu_limits_equal_typed_requests_and_public_capabilities():
    from rca_agent.investigation.inquiry_contract import public_query
    _, _, options = setup("off")
    for item in options.values():
        menu = public_query(item)
        request = item["action"].request
        for field in ("max_points", "max_results"):
            if hasattr(request, field):
                assert menu["result_limits"][field] == getattr(request, field)
        assert menu["result_limits"]["capability_max_results"] == item["capability_max_results"]
        assert "partial" in menu["truncation"]
