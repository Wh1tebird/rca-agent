"""Synthetic state-machine controls, without provider, evaluator or GT."""
from copy import deepcopy

import pytest
from test_engineering_independent_regressions import output, setup

from rca_agent.investigation.inquiry_contract import public_query
from rca_agent.investigation.inquiry_loop import (
    LoopState,
    accept_loop_step,
    apply_loop_response,
    normalize_loop_step,
)
from rca_agent.models import RequestStopAction
from rca_agent.runner import EpisodeRunner, RecordedReplay


def menu_and_draft(options):
    menu = {ref: {**item, "action": item["action"].model_dump(mode="json")}
            for ref, item in options.items()}
    public = {f"q{i}": public_query(item) for i, item in enumerate(options.values(), 1)}
    raw = output({"context": __import__("json").dumps({"queries": public})}, (("n1", "q1"),))
    return menu, raw


def planning_step(state, draft, selection="q1"):
    return {"based_on_revision": state.revision, "based_on_digest": state.digest,
            "batch_assessments": [], "candidate_updates": [], "need_updates": [],
            "pending_decisions": [], "draft": draft,
            "next_batch": selection if isinstance(selection, list) else [selection],
            "batch_reason": "Inspect the candidate and comparison", "decision": "continue"}


def test_batched_initial_step_is_atomic_and_binds_real_menu():
    _, runner, options = setup("shadow")
    state = LoopState.initial(runner.episode_id)
    assert state.verify_digest()
    menu, draft = menu_and_draft(options)
    raw = planning_step(state, draft)
    parsed = normalize_loop_step(raw, menu)
    next_state, round_record = accept_loop_step(state, parsed, menu=menu,
                                                batch_limit=2, query_limit=10)
    assert state.revision == 0 and state.pending == ()
    assert next_state.verify_digest() and next_state.phase == "awaiting_results"
    assert len(next_state.pending) == 1 and next_state.pending[0].status == "scheduled"
    assert len(round_record["selected_actions"]) == 1
    assert round_record["draft"]["contract_version"] == 2
    assert round_record["draft"]["hypotheses"][0]["alternative_id"]


@pytest.mark.parametrize("change", ("stale", "extra", "direct", "unknown_query", "empty_batch"))
def test_batched_bad_step_cannot_partially_change_state(change):
    _, runner, options = setup("shadow")
    state = LoopState.initial(runner.episode_id)
    menu, draft = menu_and_draft(options)
    raw = planning_step(state, deepcopy(draft))
    if change == "stale":
        raw["based_on_revision"] = 1
    elif change == "extra":
        raw["draft"]["choices"][0]["extra"] = "forbidden"
    elif change == "direct":
        raw["draft"]["needs"][0]["observable_signals"] = ["unavailable"]
    elif change == "unknown_query":
        raw["draft"]["choices"][0]["query_ref"] = "q999"
    else:
        raw["next_batch"] = []
    with pytest.raises(ValueError):
        parsed = normalize_loop_step(raw, menu)
        accept_loop_step(state, parsed, menu=menu, batch_limit=2, query_limit=10)
    assert state.revision == 0 and state.pending == () and state.verify_digest()


def test_two_member_batch_is_rejected_before_execution_when_budget_fits_only_one():
    from test_dynamic_investigation import registry
    from test_engineering_independent_regressions import PublicEnvironment

    env = PublicEnvironment(budget=1)
    runner = EpisodeRunner(env, evidence_registry=registry(),
        investigation_workspace_mode="shadow", inquiry_loop_mode="batched")
    runner.start()
    view = runner.inquiry_loop_view()
    draft = output({"context": __import__("json").dumps({"queries": view["view"]["menu"]})},
                   (("n1", "q1"), ("n1", "q2")))
    state = LoopState.model_validate(runner.investigation_workspace_snapshot.inquiry_loop)
    raw = planning_step(state, draft, ["q1", "q2"])
    initial_events = len(runner.transcript.events)
    with pytest.raises(ValueError, match="insufficient budget"):
        runner.record_inquiry_loop_step(raw, input_view=view)
    assert len(runner.transcript.events) == initial_events
    assert runner.investigation_workspace_snapshot.inquiry_loop["revision"] == 0
    assert env.remaining == 1 and env.calls == []


@pytest.mark.parametrize("workspace_mode,contract", (("off", 2), ("shadow", 1), ("active", 1)))
def test_invalid_batched_host_configuration_fails_before_reset(workspace_mode, contract):
    from test_dynamic_investigation import registry
    from test_engineering_independent_regressions import PublicEnvironment

    env = PublicEnvironment()
    with pytest.raises(ValueError):
        EpisodeRunner(env, evidence_registry=registry(),
            investigation_workspace_mode=workspace_mode, inquiry_loop_mode="batched",
            inquiry_contract_version=contract)
    assert env.calls == [] and env.remaining == 20


def test_view_omits_oversized_observation_whole_and_records_reference():
    from types import SimpleNamespace

    from test_engineering_independent_regressions import PublicEnvironment

    from rca_agent.investigation.inquiry_loop import make_loop_view
    from rca_agent.langchain_acquisition import query_options

    env = PublicEnvironment()
    bootstrap = env.reset()
    action = next(iter(query_options({"bootstrap": bootstrap.model_dump(mode="json")},
        [bootstrap.alert_entity.model_dump(mode="json")]).values()))["action"]
    response = env.get_metric_series(action.request)
    oversized = response.observations[0].model_copy(update={"value": "X" * (129 * 1024)})
    response = response.model_copy(update={"observations": (oversized,)})
    events = (SimpleNamespace(event_type="tool_response", payload=response.model_dump(mode="json")),)
    state = LoopState.initial("ep-" + "a" * 32)
    view, _ = make_loop_view(state, events, bootstrap, budget_remaining=19,
        query_limit=10, batch_limit=2, workspace_mode="shadow")
    assert view["observations"] == []
    assert view["omitted_refs"] == [oversized.observation_id]
    assert view["omitted_count"] == 1 and view["truncation_reason"] == "projection_limit"


def test_pending_retain_defer_cancel_controls_later_execution():
    from test_batched_inquiry_graph import BatchedEnvironment
    from test_dynamic_investigation import registry

    from rca_agent.models import CallToolAction

    env = BatchedEnvironment(values={"active_ratio": 0.2, "queue_ratio": 0.8})
    runner = EpisodeRunner(env, evidence_registry=registry(),
        investigation_workspace_mode="shadow", inquiry_loop_mode="batched")
    runner.start()
    view = runner.inquiry_loop_view()
    draft = output({"context": __import__("json").dumps({"queries": view["view"]["menu"]})},
                   (("n1", "q1"), ("n1", "q2"), ("n1", "q3")))
    state = LoopState.model_validate(runner.investigation_workspace_snapshot.inquiry_loop)
    first = runner.record_inquiry_loop_step(planning_step(state, draft), input_view=view)
    response = runner.handle(CallToolAction.model_validate(first["selected_actions"][0]["action"]))
    state = LoopState.model_validate(runner.investigation_workspace_snapshot.inquiry_loop)
    old = {item.query_ref: item for item in state.pending if item.status == "pending"}
    assert len(old) == 2

    def assessment(record, actual):
        binding = record["selected_actions"][0]
        return {"binding_id": binding["binding_id"], "query_ref": actual.query_ref,
            "observation_refs": [o.observation_id for o in actual.observations],
            "status": "answered", "candidate_effect": "unclear",
            "alternative_effect": "unclear", "remaining_gap": "Other results needed",
            "reason": "The returned result was inspected"}

    middle = {"based_on_revision": state.revision, "based_on_digest": state.digest,
        "batch_assessments": [assessment(first, response)], "candidate_updates": [],
        "need_updates": [], "pending_decisions": [
            {"pending_id": old[next(iter(old))].pending_id, "disposition": "defer",
             "reason": "Wait for complementary metric"},
            {"pending_id": old[list(old)[1]].pending_id, "disposition": "retain",
             "reason": "Use this now"}],
        "draft": None, "next_batch": [old[list(old)[1]].pending_id],
        "batch_reason": "Inspect complementary result", "decision": "continue"}
    second = runner.record_inquiry_loop_step(middle, input_view=runner.inquiry_loop_view())
    second_response = runner.handle(CallToolAction.model_validate(
        second["selected_actions"][0]["action"]))
    state = LoopState.model_validate(runner.investigation_workspace_snapshot.inquiry_loop)
    deferred = next(p for p in state.pending if p.status == "deferred")
    final = {"based_on_revision": state.revision, "based_on_digest": state.digest,
        "batch_assessments": [assessment(second, second_response)], "candidate_updates": [],
        "need_updates": [], "pending_decisions": [{"pending_id": deferred.pending_id,
            "disposition": "cancel", "reason": "The new result made it unnecessary",
            "basis_observation_refs": [o.observation_id for o in second_response.observations]}],
        "draft": None, "next_batch": [], "decision": "handoff", "stop_reason": "Reviewed"}
    runner.record_inquiry_loop_step(final, input_view=runner.inquiry_loop_view())
    assert len(env.calls) == 2
    assert {p.status for p in LoopState.model_validate(
        runner.investigation_workspace_snapshot.inquiry_loop).pending} == {"executed", "cancelled"}
    runner.handle(RequestStopAction(summary="reviewed"))
    assert RecordedReplay.verify(runner.transcript.events).valid


@pytest.mark.parametrize("value,expected", ((0, True), (None, False), (0.95, True)))
def test_direct_answer_requires_visible_matched_result(value, expected):
    from test_engineering_independent_regressions import PublicEnvironment

    env = PublicEnvironment(values={"active_ratio": value})
    _, runner, options = setup("shadow")
    state = LoopState.initial(runner.episode_id)
    menu, draft = menu_and_draft(options)
    planned, round_record = accept_loop_step(state,
        normalize_loop_step(planning_step(state, draft), menu),
        menu=menu, batch_limit=2, query_limit=10)
    action = options[round_record["selected_actions"][0]["query_ref"]]["action"]
    response = env.get_metric_series(action.request)
    observed = apply_loop_response(planned, response, sequence=10)
    assert observed.phase == "awaiting_assessment" and observed.verify_digest()
    binding = round_record["selected_actions"][0]
    assessment = {"binding_id": binding["binding_id"], "query_ref": response.query_ref,
                  "observation_refs": [o.observation_id for o in response.observations],
                  "status": "answered", "candidate_effect": "unclear",
                  "alternative_effect": "unclear", "remaining_gap": "none",
                  "reason": "The direct result is visible"}
    review = {"based_on_revision": observed.revision, "based_on_digest": observed.digest,
              "batch_assessments": [assessment], "candidate_updates": [], "need_updates": [],
              "pending_decisions": [], "draft": None, "next_batch": [],
              "decision": "handoff", "stop_reason": "Review completed"}
    parsed = normalize_loop_step(review, menu)
    if expected:
        complete, next_round = accept_loop_step(observed, parsed, menu=menu,
            batch_limit=2, query_limit=10,
            visible_refs={o.observation_id for o in response.observations})
        assert next_round is None and complete.phase == "handoff"
    else:
        with pytest.raises(ValueError, match="directly matched value"):
            accept_loop_step(observed, parsed, menu=menu, batch_limit=2,
                query_limit=10, visible_refs={o.observation_id for o in response.observations})
        assert observed.phase == "awaiting_assessment"


@pytest.mark.parametrize("case", ("missing", "unknown", "failed", "truncated", "exploratory"))
def test_noncomplete_result_cannot_be_promoted_to_answered(case):
    from test_engineering_independent_regressions import PublicEnvironment

    from rca_agent.models import DataStatus, ToolStatus

    env = PublicEnvironment(values={"active_ratio": 0.4})
    _, runner, options = setup("shadow")
    state = LoopState.initial(runner.episode_id)
    menu, draft = menu_and_draft(options)
    if case == "exploratory":
        draft["choices"][0]["mode"] = "exploratory"
    planned, round_record = accept_loop_step(state,
        normalize_loop_step(planning_step(state, draft), menu),
        menu=menu, batch_limit=2, query_limit=10)
    action = options[round_record["selected_actions"][0]["query_ref"]]["action"]
    response = env.get_metric_series(action.request)
    if case in {"missing", "unknown"}:
        observation = response.observations[0].model_copy(update={
            "data_status": DataStatus.MISSING if case == "missing" else DataStatus.UNKNOWN,
            "value": None})
        response = response.model_copy(update={"observations": (observation,)})
    elif case == "failed":
        response = response.model_copy(update={"status": ToolStatus.QUERY_FAILED,
            "observations": ()})
    elif case == "truncated":
        response = response.model_copy(update={"truncated": True})
    observed = apply_loop_response(planned, response, sequence=10)
    binding = round_record["selected_actions"][0]
    refs = [o.observation_id for o in response.observations]
    review = {"based_on_revision": observed.revision, "based_on_digest": observed.digest,
        "batch_assessments": [{"binding_id": binding["binding_id"],
            "query_ref": response.query_ref, "observation_refs": refs,
            "status": "answered", "candidate_effect": "unclear",
            "alternative_effect": "unclear", "remaining_gap": "none", "reason": "Claimed answer"}],
        "candidate_updates": [], "need_updates": [], "pending_decisions": [],
        "draft": None, "next_batch": [], "decision": "handoff", "stop_reason": "Reviewed"}
    with pytest.raises(ValueError):
        accept_loop_step(observed, normalize_loop_step(review, menu), menu=menu,
            batch_limit=2, query_limit=10, visible_refs=set(refs))
    review["batch_assessments"][0]["status"] = "unavailable" if case == "failed" else "partial"
    final, _ = accept_loop_step(observed, normalize_loop_step(review, menu), menu=menu,
        batch_limit=2, query_limit=10, visible_refs=set(refs))
    assert final.phase == "handoff" and observed.phase == "awaiting_assessment"


def test_fifth_planning_step_still_allows_one_review_only_step():
    from test_engineering_independent_regressions import PublicEnvironment

    from rca_agent.investigation.inquiry_loop import _state

    env = PublicEnvironment()
    _, runner, options = setup("shadow")
    state = LoopState.initial(runner.episode_id)
    menu, draft = menu_and_draft(options)
    planned, round_record = accept_loop_step(state,
        normalize_loop_step(planning_step(state, draft), menu),
        menu=menu, batch_limit=2, query_limit=10)
    action = options[round_record["selected_actions"][0]["query_ref"]]["action"]
    response = env.get_metric_series(action.request)
    observed = apply_loop_response(planned, response, sequence=10)
    observed = _state(observed, planning_steps=5)
    binding = round_record["selected_actions"][0]
    base = {"based_on_revision": observed.revision, "based_on_digest": observed.digest,
        "batch_assessments": [{"binding_id": binding["binding_id"],
            "query_ref": response.query_ref,
            "observation_refs": [o.observation_id for o in response.observations],
            "status": "ambiguous", "candidate_effect": "unclear",
            "alternative_effect": "unclear", "remaining_gap": "Causal distinction",
            "reason": "Result is insufficient alone"}],
        "candidate_updates": [], "need_updates": [], "pending_decisions": [],
        "draft": None, "next_batch": [], "decision": "handoff", "stop_reason": "Final review"}
    finished, _ = accept_loop_step(observed, normalize_loop_step(base, menu),
        menu=menu, batch_limit=2, query_limit=10,
        visible_refs={o.observation_id for o in response.observations})
    assert finished.review_steps == 1 and finished.planning_steps == 5
    continuing = {**base, "decision": "continue", "next_batch": ["q2"],
                  "batch_reason": "Too late", "stop_reason": None}
    with pytest.raises(ValueError):
        accept_loop_step(observed, normalize_loop_step(continuing, menu),
            menu=menu, batch_limit=2, query_limit=10,
            visible_refs={o.observation_id for o in response.observations})


def test_retained_pending_becomes_unavailable_if_exact_menu_action_disappears():
    from test_engineering_independent_regressions import PublicEnvironment

    env = PublicEnvironment()
    _, runner, options = setup("shadow")
    state = LoopState.initial(runner.episode_id)
    menu, draft = menu_and_draft(options)
    public = {f"q{i}": public_query(option)
              for i, option in enumerate(options.values(), 1)}
    draft = output({"context": __import__("json").dumps({"queries": public})},
                   (("n1", "q1"), ("n1", "q2")))
    planned, round_record = accept_loop_step(state,
        normalize_loop_step(planning_step(state, draft), menu),
        menu=menu, batch_limit=2, query_limit=10)
    action = options[round_record["selected_actions"][0]["query_ref"]]["action"]
    response = env.get_metric_series(action.request)
    observed = apply_loop_response(planned, response, sequence=10)
    binding = round_record["selected_actions"][0]
    old = next(p for p in observed.pending if p.status == "pending")
    smaller_menu = {ref: item for ref, item in menu.items() if ref != old.query_ref}
    review = {"based_on_revision": observed.revision, "based_on_digest": observed.digest,
        "batch_assessments": [{"binding_id": binding["binding_id"],
            "query_ref": response.query_ref,
            "observation_refs": [o.observation_id for o in response.observations],
            "status": "ambiguous", "candidate_effect": "unclear",
            "alternative_effect": "unclear", "remaining_gap": "Later query not available",
            "reason": "Metric is not decisive"}],
        "candidate_updates": [], "need_updates": [],
        "pending_decisions": [{"pending_id": old.pending_id,
            "disposition": "retain", "reason": "Still desired"}],
        "draft": None, "next_batch": [], "decision": "handoff", "stop_reason": "Reviewed"}
    finished, _ = accept_loop_step(observed, normalize_loop_step(review, smaller_menu),
        menu=smaller_menu, batch_limit=2, query_limit=10,
        visible_refs={o.observation_id for o in response.observations})
    assert next(p for p in finished.pending if p.pending_id == old.pending_id).status == "unavailable"


def test_runner_records_batch_result_assessment_and_replays():
    from test_dynamic_investigation import registry
    from test_engineering_independent_regressions import PublicEnvironment

    env = PublicEnvironment()
    runner = EpisodeRunner(env, evidence_registry=registry(),
                           investigation_workspace_mode="shadow", inquiry_loop_mode="batched")
    runner.start()
    view = runner.inquiry_loop_view()
    draft = output({"context": __import__("json").dumps({"queries": view["view"]["menu"]})},
                   (("n1", "q1"),))
    current = LoopState.model_validate(runner.investigation_workspace_snapshot.inquiry_loop)
    round_record = runner.record_inquiry_loop_step(planning_step(current, draft), input_view=view)
    assert round_record is not None
    action = view["menu"][round_record["selected_actions"][0]["query_ref"]]["action"]
    from rca_agent.models import CallToolAction

    response = runner.handle(CallToolAction.model_validate(action))
    current = LoopState.model_validate(runner.investigation_workspace_snapshot.inquiry_loop)
    assert current.phase == "awaiting_assessment" and env.calls == ["active_ratio"]
    next_view = runner.inquiry_loop_view()
    binding = round_record["selected_actions"][0]
    review = {"based_on_revision": current.revision, "based_on_digest": current.digest,
        "batch_assessments": [{"binding_id": binding["binding_id"],
            "query_ref": response.query_ref,
            "observation_refs": [o.observation_id for o in response.observations],
            "status": "answered", "candidate_effect": "unclear",
            "alternative_effect": "unclear", "remaining_gap": "none", "reason": "Measured"}],
        "candidate_updates": [], "need_updates": [], "pending_decisions": [],
        "draft": None, "next_batch": [], "decision": "handoff", "stop_reason": "Reviewed"}
    assert runner.record_inquiry_loop_step(review, input_view=next_view) is None
    runner.handle(RequestStopAction(summary="synthetic review completed"))
    assert runner.answer.verify_digest() and RecordedReplay.verify(runner.transcript.events).valid


def test_review_priority_reorders_real_handoff_without_changing_original_identity():
    from test_dynamic_investigation import registry
    from test_engineering_independent_regressions import PublicEnvironment

    runner = EpisodeRunner(PublicEnvironment(), evidence_registry=registry(),
        investigation_workspace_mode="shadow", inquiry_loop_mode="batched")
    runner.start()
    view = runner.inquiry_loop_view()
    draft = output({"context": __import__("json").dumps({"queries": view["view"]["menu"]})},
                   (("n1", "q1"),))
    draft["hypotheses"].append({"hypothesis_id": "h2", "mechanism": "queue starvation",
        "alternative": "upstream throttling"})
    draft["needs"].append({**draft["needs"][0], "need_id": "n2", "hypothesis_id": "h2"})
    state = LoopState.model_validate(runner.investigation_workspace_snapshot.inquiry_loop)
    initial = runner.record_inquiry_loop_step(planning_step(state, draft), input_view=view)
    action = initial["selected_actions"][0]["action"]
    from rca_agent.models import CallToolAction

    response = runner.handle(CallToolAction.model_validate(action))
    state = LoopState.model_validate(runner.investigation_workspace_snapshot.inquiry_loop)
    binding = initial["selected_actions"][0]
    review = {"based_on_revision": state.revision, "based_on_digest": state.digest,
        "batch_assessments": [{"binding_id": binding["binding_id"],
            "query_ref": response.query_ref,
            "observation_refs": [o.observation_id for o in response.observations],
            "status": "ambiguous", "candidate_effect": "unclear",
            "alternative_effect": "unclear", "remaining_gap": "More differentiation",
            "reason": "One metric is not decisive"}],
        "candidate_updates": [{"hypothesis_id": "h2", "priority": 3,
            "disposition": "active", "reason": "Investigate first"}],
        "need_updates": [], "pending_decisions": [], "draft": None,
        "next_batch": [], "decision": "handoff", "stop_reason": "Reviewed"}
    runner.record_inquiry_loop_step(review, input_view=runner.inquiry_loop_view())
    handoff = runner.inquiry_loop_handoff()
    assert [h["hypothesis_id"] for h in handoff["hypotheses"]] == ["h2", "h1"]
    assert handoff["candidate_history"][1]["hypothesis"]["alternative"] == "connection pool exhaustion"
    assert handoff["need_assessments"][0]["status"] == "ambiguous"
    runner.handle(RequestStopAction(summary="reviewed"))
    assert RecordedReplay.verify(runner.transcript.events).valid

    from rca_agent.runner import ReplayValidationError
    from rca_agent.runner.transcript import EventType, Transcript
    from rca_agent.tools.fingerprint import sha256_digest

    for change in ("missing_version", "wrong_state", "future_observation"):
        rebuilt = Transcript(runner.episode_id)
        for event in runner.transcript.events:
            payload = deepcopy(event.payload)
            if event.event_type == EventType.INQUIRY_LOOP_STEP_ACCEPTED:
                if change == "missing_version":
                    payload["step"]["contract_version"] = None
                elif change == "wrong_state":
                    payload["step"]["based_on_revision"] += 1
                elif payload["step"]["batch_assessments"]:
                    payload["step"]["batch_assessments"][0]["observation_refs"] = ["obs-future"]
            if event.event_type == EventType.ANSWER_FROZEN:
                payload["transcript_digest"] = rebuilt.head_digest
                payload["artifact_digest"] = sha256_digest({
                    key: value for key, value in payload.items() if key != "artifact_digest"})
            rebuilt.append(event.event_type, payload)
        with pytest.raises(ReplayValidationError, match="workspace event is invalid"):
            RecordedReplay.verify(rebuilt.events)
