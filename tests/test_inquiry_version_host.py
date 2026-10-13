"""Host version authority across model admission, direct Runner records and recorded Replay."""
from copy import deepcopy

import pytest
from langchain_core.runnables import RunnableLambda
from test_dynamic_investigation import registry
from test_engineering_independent_regressions import PublicEnvironment, output, policy_for, setup

from rca_agent.inquiry_policy import InquiryInvestigationPolicy, InquiryUnavailableError
from rca_agent.investigation.workspace import InvestigationWorkspaceReducer
from rca_agent.langchain_acquisition import LangChainInquiryAdapter
from rca_agent.models import RequestStopAction
from rca_agent.runner import EpisodeRunner, RecordedReplay
from rca_agent.runner.transcript import GENESIS_DIGEST, EventType, TranscriptEvent
from rca_agent.tools.fingerprint import sha256_digest

VERSIONS = ("correct", "missing", "null", "unknown", "inconsistent")
MODES = ("off", "shadow", "active")


def rebuild(events, mutate):
    """Rehash the chain, its internal references and frozen Workspace together."""
    result, replacements = [], {}
    reducer = InvestigationWorkspaceReducer(events[0].episode_id)

    def replace(value):
        if isinstance(value, str):
            return replacements.get(value, value)
        if isinstance(value, list):
            return [replace(item) for item in value]
        if isinstance(value, dict):
            return {key: replace(item) for key, item in value.items()}
        return value

    for event in events:
        payload = replace(deepcopy(event.payload))
        mutate(event, payload)
        head = result[-1].digest if result else GENESIS_DIGEST
        if event.event_type == EventType.ANSWER_FROZEN:
            payload["transcript_digest"] = head
            if payload.get("investigation_workspace") is not None:
                payload["investigation_workspace"] = reducer.workspace.snapshot(
                    transcript_head_sequence=len(result), transcript_head_digest=head
                ).model_dump(mode="json")
            payload["artifact_digest"] = sha256_digest({
                key: value for key, value in payload.items() if key != "artifact_digest"})
        raw = event.model_dump(mode="json", exclude={"digest"})
        raw.update(payload=payload, previous_digest=head)
        rebuilt = TranscriptEvent(**raw, digest=sha256_digest(raw))
        reducer.apply(rebuilt)
        result.append(rebuilt)
        replacements[event.digest] = rebuilt.digest
    return tuple(result)


def set_version(raw, host, version):
    if version == "missing":
        raw.pop("contract_version", None)
    else:
        raw["contract_version"] = {"correct": host, "null": None, "unknown": 999,
                                   "inconsistent": 3 - host}[version]


def make_runner(host, mode):
    env = PublicEnvironment()
    runner = EpisodeRunner(env, evidence_registry=registry(), investigation_workspace_mode=mode,
                           inquiry_contract_version=host)
    runner.start()
    return env, runner


@pytest.mark.parametrize("entry", ("policy", "runner"))
@pytest.mark.parametrize("host", (1, 2))
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("version", VERSIONS)
@pytest.mark.parametrize("wrong", (False, True))
def test_host_live_version_matrix(entry, host, mode, version, wrong):
    env, runner = make_runner(host, mode)
    _, _, options = setup("off")

    def response(context):
        raw = output(context, (("n1", "q1"),))
        set_version(raw, host, version)
        if wrong:
            raw["needs"][0]["observable_signals"] = ["unavailable_sql_signal"]
        return raw

    policy = InquiryInvestigationPolicy(LangChainInquiryAdapter(RunnableLambda(response),
                                                               contract_version=host))
    policy.selection_calls = 1
    expected = version not in {"unknown", "inconsistent"} and (host == 1 or not wrong)
    initial_events = len(runner.transcript.events)
    initial_workspace = runner.investigation_workspace_snapshot
    if entry == "policy":
        if not expected:
            with pytest.raises(InquiryUnavailableError):
                policy._plan_inquiry_round({}, options)
            assert not policy.inquiry_drafts and not policy.selection_queue
            assert all(e["status"] != "accepted" for e in policy.drain_inquiry_diagnostics())
        else:
            policy._plan_inquiry_round({}, options)
            record = policy.inquiry_drafts[0]
    else:
        record = deepcopy(policy_for(options, (("n1", "q1"),)).inquiry_drafts[0])
        set_version(record["draft"], host, version)
        if wrong:
            record["draft"]["needs"][0]["observable_signals"] = ["unavailable_sql_signal"]
        if not expected:
            with pytest.raises(ValueError):
                runner.record_inquiry_round(record)
    if not expected:
        assert env.calls == [] and env.remaining == 20
        assert len(runner.transcript.events) == initial_events
        assert runner.investigation_workspace_snapshot == initial_workspace
        return
    runner.record_inquiry_round(record)
    declared = next(e.payload["draft"] for e in runner.transcript.events
                    if e.event_type == EventType.INQUIRY_ROUND_RECORDED)
    assert declared.get("contract_version") == (2 if host == 2 else None)
    action = options[record["selected_actions"][0]["query_ref"]]["action"]
    routed = runner.route_active_inquiry_action(action)
    assert routed is not None
    runner.handle(routed)
    assert env.calls == ["active_ratio"] and env.remaining == 19
    runner.handle(RequestStopAction(summary="host version matrix"))
    assert RecordedReplay.verify(runner.transcript.events).valid


@pytest.mark.parametrize("host", (1, 2))
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("version", VERSIONS)
@pytest.mark.parametrize("wrong", (False, True))
def test_recorded_host_version_matrix(host, mode, version, wrong):
    _, runner = make_runner(host, mode)
    _, _, options = setup("off")
    record = deepcopy(policy_for(options, (("n1", "q1"),)).inquiry_drafts[0])
    set_version(record["draft"], host, "correct")
    if host == 1 and wrong:
        record["draft"]["needs"][0]["observable_signals"] = ["unavailable_sql_signal"]
    runner.record_inquiry_round(record)
    # No tool need be issued to prove replay admission rejects a forged declaration.
    runner.handle(RequestStopAction(summary="recorded host version matrix"))
    assert RecordedReplay.verify(runner.transcript.events).valid

    def mutate(event, payload):
        if event.event_type == EventType.INQUIRY_ROUND_RECORDED:
            if version != "correct":
                set_version(payload["draft"], host, version)
            if host == 2 and wrong:
                payload["draft"]["needs"][0]["observable_signals"] = ["unavailable_sql_signal"]

    expected = version == "correct" and (host == 1 or not wrong)
    # Serialized legacy deliberately omits its marker; an explicit null has identical semantics.
    if host == 1 and version in {"missing", "null"}:
        expected = True
    if expected:
        assert RecordedReplay.verify(rebuild(runner.transcript.events, mutate)).valid
    else:
        with pytest.raises(ValueError):
            RecordedReplay.verify(rebuild(runner.transcript.events, mutate))


@pytest.mark.parametrize("field", ("mode", "purpose", "measurement_target"))
@pytest.mark.parametrize("version", ("missing", "null"))
@pytest.mark.parametrize("mode", MODES)
def test_missing_version_cannot_remove_new_required_fields(field, version, mode):
    env, runner, options = setup(mode)
    record = deepcopy(policy_for(options, (("n1", "q1"),)).inquiry_drafts[0])
    set_version(record["draft"], 2, version)
    record["draft"]["choices"][0].pop(field)
    with pytest.raises(ValueError):
        runner.record_inquiry_round(record)
    assert env.calls == [] and env.remaining == 20


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("version", ("missing", "null"))
def test_independent_missing_version_direct_reproduction(mode, version):
    env, runner, options = setup(mode)
    record = deepcopy(policy_for(options, (("n1", "q1"),)).inquiry_drafts[0])
    set_version(record["draft"], 2, version)
    record["draft"]["needs"][0]["observable_signals"] = ["unavailable_sql_signal"]
    assert runner.transcript.events[0].payload["engineering_contract_revision"] == 2
    with pytest.raises(ValueError):
        runner.record_inquiry_round(record)
    assert env.calls == []


@pytest.mark.parametrize("version", ("missing", "null"))
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("defect", ("extra", "self", "target", "binding", "mode"))
def test_missing_marker_does_not_disable_other_contract_checks(version, mode, defect):
    env, runner, options = setup(mode)
    record = deepcopy(policy_for(options, (("n1", "q1"),)).inquiry_drafts[0])
    set_version(record["draft"], 2, version)
    choice = record["draft"]["choices"][0]
    if defect == "extra":
        choice["unpublished_extension"] = True
    elif defect == "self":
        hypothesis = record["draft"]["hypotheses"][0]
        hypothesis["alternative"] = hypothesis["mechanism"]
    elif defect == "target":
        choice["measurement_target"]["signal"] = "invented"
    elif defect == "binding":
        record["selected_actions"][0]["binding_id"] = "forged"
    else:
        choice["mode"] = "invented"
    before = (len(runner.transcript.events), runner.investigation_workspace_snapshot)
    with pytest.raises(ValueError):
        runner.record_inquiry_round(record)
    assert (len(runner.transcript.events), runner.investigation_workspace_snapshot) == before
    assert env.calls == [] and env.remaining == 20


@pytest.mark.parametrize("host", (1, 2))
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("historical", (False, True))
def test_replay_locks_first_round_and_rejects_later_version_change(host, mode, historical):
    _, runner = make_runner(host, mode)
    _, _, options = setup("off")
    policy = policy_for(options, (("n1", "q1"),))
    for number in (1, 2):
        policy.selection_calls = number
        policy._plan_inquiry_round({}, options)
        record = deepcopy(policy.inquiry_drafts[-1])
        set_version(record["draft"], host, "correct")
        runner.record_inquiry_round(record)
    runner.handle(RequestStopAction(summary="mixed versions"))

    def old_format(event, payload):
        if historical and event.event_type == EventType.EPISODE_CREATED:
            payload.pop("engineering_contract_revision")
            payload.pop("inquiry_contract_version")

    baseline = rebuild(runner.transcript.events, old_format)
    assert RecordedReplay.verify(baseline).valid

    def mutate(event, payload):
        if event.event_type == EventType.INQUIRY_ROUND_RECORDED and payload["round"] == 2:
            set_version(payload["draft"], host, "inconsistent")

    with pytest.raises(ValueError, match="Inquiry contract"):
        RecordedReplay.verify(rebuild(baseline, mutate))


@pytest.mark.parametrize("version", (None, True, "2", 999))
def test_replay_rejects_invalid_episode_host_declaration(version):
    _, runner = make_runner(2, "off")
    runner.handle(RequestStopAction(summary="invalid host metadata"))

    def mutate(event, payload):
        if event.event_type == EventType.EPISODE_CREATED:
            payload["inquiry_contract_version"] = version

    with pytest.raises(ValueError):
        RecordedReplay.verify(rebuild(runner.transcript.events, mutate))


@pytest.mark.parametrize("version", ("missing", "null"))
def test_custom_policy_adapter_and_default_parser_cannot_infer_legacy(version):
    from rca_agent.langchain_acquisition import InquiryDraft, parse_inquiry_draft
    _, _, options = setup("off")
    record = deepcopy(policy_for(options, (("n1", "q1"),)).inquiry_drafts[0])
    set_version(record["draft"], 2, version)
    record["draft"]["choices"][0].pop("mode")
    with pytest.raises(ValueError, match="BINDING_MODE_REQUIRED"):
        parse_inquiry_draft(record["draft"])

    class Custom:
        contract_version = 2

        def plan(self, context, options):
            return InquiryDraft.model_validate(record["draft"]), (options["q1"]["action"],)

    policy = InquiryInvestigationPolicy(Custom())
    policy.selection_calls = 1
    with pytest.raises(InquiryUnavailableError):
        policy._plan_inquiry_round({}, options)
    assert not policy.selection_queue and not policy.inquiry_drafts
    assert all(e["status"] != "accepted" for e in policy.drain_inquiry_diagnostics())


@pytest.mark.parametrize("version", ("missing", "null", "unknown"))
@pytest.mark.parametrize("form", ("parsed", "tool_call", "json"))
@pytest.mark.parametrize("wrong", (False, True))
def test_all_model_response_forms_use_host_version(version, form, wrong):
    import json
    _, _, options = setup("off")

    def response(context):
        raw = output(context, (("n1", "q1"),))
        set_version(raw, 2, version)
        if wrong:
            raw["needs"][0]["observable_signals"] = ["unavailable_sql_signal"]
        if form == "parsed":
            return {"parsed": raw, "raw": None}
        message = type("Raw", (), {"content": json.dumps(raw) if form == "json" else "",
            "tool_calls": [] if form == "json" else [{"name": "InquiryDraftV2", "args": raw}]})()
        return {"parsed": None, "raw": message}

    adapter = LangChainInquiryAdapter(RunnableLambda(response))
    if version == "unknown" or wrong:
        with pytest.raises(ValueError):
            adapter.plan({}, options)
    else:
        draft, actions = adapter.plan({}, options)
        assert draft.contract_version == 2 and len(actions) == 1


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("selector", ("inquiry", "revision", "bound"))
def test_explicit_host_rollback_retains_true_legacy_schema(mode, selector):
    config = {"inquiry": {"inquiry_contract_version": 1},
              "revision": {"engineering_contract_revision": 1},
              "bound": {"bound_handoff_version": 6}}[selector]
    env = PublicEnvironment()
    runner = EpisodeRunner(env, evidence_registry=registry(),
                           investigation_workspace_mode=mode, **config)
    runner.start()
    _, _, options = setup("off")
    record = deepcopy(policy_for(options, (("n1", "q1"),)).inquiry_drafts[0])
    record["draft"].pop("contract_version")
    for field in ("mode", "purpose", "measurement_target"):
        record["draft"]["choices"][0].pop(field)
    record["draft"]["needs"][0]["observable_signals"] = ["unavailable_sql_signal"]
    runner.record_inquiry_round(record)
    action = options[record["selected_actions"][0]["query_ref"]]["action"]
    runner.handle(runner.route_active_inquiry_action(action))
    assert env.calls == ["active_ratio"] and env.remaining == 19
    runner.handle(RequestStopAction(summary="explicit rollback"))
    assert RecordedReplay.verify(runner.transcript.events).valid


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("historical", (False, True))
@pytest.mark.parametrize("version", ("missing", "null"))
def test_recorded_v2_cannot_omit_marker_after_contract_lock(mode, historical, version):
    _, runner = make_runner(2, mode)
    _, _, options = setup("off")
    policy = policy_for(options, (("n1", "q1"),))
    for number in (1, 2):
        policy.selection_calls = number
        policy._plan_inquiry_round({}, options)
        runner.record_inquiry_round(policy.inquiry_drafts[-1])
    runner.handle(RequestStopAction(summary="missing marker in later round"))

    def format_metadata(event, payload):
        if event.event_type == EventType.EPISODE_CREATED:
            payload.pop("inquiry_contract_version")
            if historical:
                payload.pop("engineering_contract_revision")

    baseline = rebuild(runner.transcript.events, format_metadata)
    assert RecordedReplay.verify(baseline).valid

    def mutate(event, payload):
        if event.event_type == EventType.INQUIRY_ROUND_RECORDED and payload["round"] == 2:
            set_version(payload["draft"], 2, version)

    with pytest.raises(ValueError, match="Inquiry contract"):
        RecordedReplay.verify(rebuild(baseline, mutate))
