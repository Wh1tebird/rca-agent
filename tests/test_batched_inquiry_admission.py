"""Execution eligibility at the public Runner boundary, using synthetic tools."""
import json
from datetime import timedelta

import pytest
from test_batched_inquiry_graph import BatchedEnvironment
from test_batched_inquiry_state import planning_step
from test_dynamic_investigation import registry
from test_engineering_independent_regressions import output

from rca_agent.investigation.inquiry_loop import LoopState
from rca_agent.models import CallToolAction, RequestStopAction
from rca_agent.runner import EpisodeRunner, RecordedReplay, ReplayValidationError
from rca_agent.runner.episode import ActionRejectedError
from rca_agent.runner.transcript import GENESIS_DIGEST, EventType, TranscriptEvent
from rca_agent.tools.fingerprint import sha256_digest


def _runner(mode="shadow"):
    env = BatchedEnvironment(values={"active_ratio": 0.3, "queue_ratio": 0.6})
    runner = EpisodeRunner(env, evidence_registry=registry(),
                           investigation_workspace_mode=mode, inquiry_loop_mode="batched")
    runner.start()
    return env, runner


def _menu_action(view, signal):
    alias = next(ref for ref, item in view["view"]["menu"].items()
                 if item.get("signal") == signal and item.get("measurement_kind") == "series")
    return CallToolAction.model_validate(list(view["menu"].values())[int(alias[1:]) - 1]["action"])


def _reject_without_effect(env, runner, action):
    before = (tuple(env.calls), env.remaining, len(runner.transcript.events),
              runner.investigation_workspace_snapshot.inquiry_loop)
    with pytest.raises(ActionRejectedError):
        runner.handle(action)
    assert (tuple(env.calls), env.remaining, len(runner.transcript.events),
            runner.investigation_workspace_snapshot.inquiry_loop) == before


@pytest.mark.parametrize("mode", ["shadow", "active"])
def test_direct_runner_requires_accepted_current_unspent_batch(mode):
    env, runner = _runner(mode)
    view = runner.inquiry_loop_view()
    selected = _menu_action(view, "active_ratio")
    other = _menu_action(view, "queue_ratio")
    _reject_without_effect(env, runner, selected)
    first_alias = next(ref for ref, item in view["view"]["menu"].items()
                       if item.get("signal") == "active_ratio"
                       and item.get("measurement_kind") == "series")
    second_alias = next(ref for ref, item in view["view"]["menu"].items()
                        if item.get("signal") == "queue_ratio"
                        and item.get("measurement_kind") == "series")
    draft = output({"context": json.dumps({"queries": view["view"]["menu"]})},
                   (("n1", first_alias), ("n1", second_alias)))
    state = LoopState.model_validate(runner.investigation_workspace_snapshot.inquiry_loop)
    runner.record_inquiry_loop_step(planning_step(state, draft, first_alias), input_view=view)
    _reject_without_effect(env, runner, other)  # Pending, but outside this batch.
    altered = selected.model_copy(update={"request": selected.request.model_copy(
        update={"entity_ref": "ent-other-service"})})
    _reject_without_effect(env, runner, altered)
    altered_signal = selected.model_copy(update={"request": selected.request.model_copy(
        update={"signal": "queue_ratio"})})
    _reject_without_effect(env, runner, altered_signal)
    altered_window = selected.model_copy(update={"request": selected.request.model_copy(
        update={"window": selected.request.window.model_copy(update={
            "start": selected.request.window.start + timedelta(seconds=1)})})})
    _reject_without_effect(env, runner, altered_window)
    runner.handle(selected)
    assert env.calls == ["active_ratio"] and env.remaining == 19
    _reject_without_effect(env, runner, selected)  # Consumed grant, same Action.
    answer = runner.handle(RequestStopAction(summary="bounded admission test"))
    assert answer.verify_digest() and RecordedReplay.verify(runner.transcript.events).valid
    assert sum(e.event_type == EventType.TOOL_REQUEST for e in runner.transcript.events) == 1


def test_legacy_runner_retains_unbatched_tool_execution():
    env = BatchedEnvironment(values={"active_ratio": 0.3})
    runner = EpisodeRunner(env, evidence_registry=registry(),
                           investigation_workspace_mode="shadow", inquiry_loop_mode="legacy")
    runner.start()
    action = CallToolAction.model_validate({"tool_name": "get_metric_series", "request": {
        "entity_ref": runner.transcript.events[1].payload["bootstrap"]["alert_entity"]["entity_ref"],
        "signal": "active_ratio", "window": runner.transcript.events[1].payload["bootstrap"]["observation_horizon"],
        "max_points": 50}})
    runner.handle(action)
    assert env.calls == ["active_ratio"] and env.remaining == 19


def test_host_startup_grant_is_exact_and_public_runner_cannot_invent_it():
    from rca_agent.models import (
        GetServiceTraceSummaryRequest,
        QueryCost,
        ToolName,
        ToolResponse,
        ToolStatus,
    )
    from rca_agent.tools.fingerprint import query_fingerprint

    class StartupEnvironment(BatchedEnvironment):
        def get_service_trace_summary(self, request):
            before = self.remaining
            self.remaining -= 1
            self.calls.append("startup_trace")
            return ToolResponse(tool_name=ToolName.GET_SERVICE_TRACE_SUMMARY,
                query_ref=query_fingerprint(ToolName.GET_SERVICE_TRACE_SUMMARY, request),
                status=ToolStatus.EMPTY, observations=(), truncated=False,
                message="No trace samples in synthetic fixture",
                cost=QueryCost(units=1, budget_before=before, budget_after=self.remaining))

    env = StartupEnvironment(values={"active_ratio": 0.3})
    runner = EpisodeRunner(env, evidence_registry=registry(),
                           investigation_workspace_mode="shadow", inquiry_loop_mode="batched")
    runner.start()
    bootstrap = runner.transcript.events[1].payload["bootstrap"]
    action = CallToolAction(tool_name=ToolName.GET_SERVICE_TRACE_SUMMARY,
        request=GetServiceTraceSummaryRequest(
            entity_ref=bootstrap["alert_entity"]["entity_ref"],
            window=bootstrap["observation_horizon"], min_duration_ms=100, max_traces=4))
    _reject_without_effect(env, runner, action)
    assert runner.select_startup_query() == action
    runner.handle_startup_query(action)
    assert env.remaining == 19
    _reject_without_effect(env, runner, action)
    runner.handle(RequestStopAction(summary="startup grant test"))
    assert RecordedReplay.verify(runner.transcript.events).valid


@pytest.mark.parametrize("disposition", ["defer", "cancel"])
def test_deferred_or_cancelled_member_has_no_direct_execution_grant(disposition):
    env, runner = _runner()
    first_view = runner.inquiry_loop_view()
    menu = first_view["view"]["menu"]
    aliases = {signal: next(ref for ref, item in menu.items()
                if item.get("signal") == signal and item.get("measurement_kind") == "series")
               for signal in ("active_ratio", "queue_ratio")}
    draft = output({"context": json.dumps({"queries": menu})},
                   (("n1", aliases["active_ratio"]), ("n1", aliases["queue_ratio"])))
    state = LoopState.model_validate(runner.investigation_workspace_snapshot.inquiry_loop)
    runner.record_inquiry_loop_step(planning_step(state, draft, aliases["active_ratio"]),
                                    input_view=first_view)
    runner.handle(_menu_action(first_view, "active_ratio"))
    second = runner.inquiry_loop_view()
    result = second["view"]["last_batch"]["results"][0]
    pending = next(p for p in second["view"]["pending"] if p["status"] == "pending")
    runner.record_inquiry_loop_step({
        "contract_version": 1, "based_on_revision": second["view"]["based_on_revision"],
        "based_on_digest": second["view"]["based_on_digest"],
        "batch_assessments": [{"binding_id": binding["binding_id"],
            "query_ref": result["query_fingerprint"],
            "observation_refs": result["observation_refs"], "status": "answered",
            "candidate_effect": "unchanged", "alternative_effect": "unchanged",
            "remaining_gap": "Another measurement deferred", "reason": "Synthetic result assessed"}
            for binding in result["bindings"]],
        "candidate_updates": [], "need_updates": [],
        "pending_decisions": [{"pending_id": pending["pending_id"],
            "disposition": disposition, "reason": "No further execution in this batch",
            "basis_observation_refs": result["observation_refs"] if disposition == "cancel" else []}],
        "draft": None, "next_batch": [], "batch_reason": None,
        "decision": "handoff", "stop_reason": "Result reviewed"}, input_view=second)
    _reject_without_effect(env, runner, CallToolAction.model_validate(pending["action"]))
    runner.handle(RequestStopAction(summary="stale query rejected"))
    assert RecordedReplay.verify(runner.transcript.events).valid


def _rechain_with_forged_source(events, source):
    result = []
    for event in events:
        payload = dict(event.payload)
        if event.event_type == EventType.TOOL_REQUEST:
            payload["execution_source"] = source
        head = result[-1].digest if result else GENESIS_DIGEST
        if event.event_type == EventType.ANSWER_FROZEN:
            payload["transcript_digest"] = head
            payload["artifact_digest"] = sha256_digest({
                key: value for key, value in payload.items() if key != "artifact_digest"})
        raw = event.model_dump(mode="json", exclude={"digest"})
        raw.update(payload=payload, previous_digest=head)
        result.append(TranscriptEvent(**raw, digest=sha256_digest(raw)))
    assert all(event.verify_digest() for event in result)
    return tuple(result)


@pytest.mark.parametrize("source", ["startup", "verification", "unknown"])
def test_replay_rejects_forged_execution_source_after_chain_and_answer_rehash(source):
    _, runner = _runner()
    view = runner.inquiry_loop_view()
    alias = next(ref for ref, item in view["view"]["menu"].items()
                 if item.get("signal") == "active_ratio" and item.get("measurement_kind") == "series")
    draft = output({"context": json.dumps({"queries": view["view"]["menu"]})},
                   (("n1", alias),))
    state = LoopState.model_validate(runner.investigation_workspace_snapshot.inquiry_loop)
    runner.record_inquiry_loop_step(planning_step(state, draft, alias), input_view=view)
    runner.handle(_menu_action(view, "active_ratio"))
    runner.handle(RequestStopAction(summary="source integrity test"))
    assert RecordedReplay.verify(runner.transcript.events).valid
    with pytest.raises(ReplayValidationError):
        RecordedReplay.verify(_rechain_with_forged_source(runner.transcript.events, source))


def test_batched_handoff_allows_host_planner_verification():
    from langchain_core.runnables import RunnableLambda
    from test_batched_inquiry_graph import DirectBatchedPolicy
    from test_dynamic_investigation import TwoStagePlanner
    from test_engineering_alternatives import DiscoveryV7

    from rca_agent.batched_inquiry_policy import LangChainBatchedInquiryAdapter
    from rca_agent.langchain_policy import LangChainPolicyAdapter
    from rca_agent.langgraph_orchestrator import LangGraphInvestigation

    env = BatchedEnvironment(values={"active_ratio": 0.95, "queue_ratio": 0.9,
        "downstream_wait_ratio": 0, "gc_pause_ratio": 0, "traffic_growth_ratio": 0})
    runner = EpisodeRunner(env, evidence_registry=registry(), discovery=DiscoveryV7(),
        verification_planner=TwoStagePlanner(), investigation_workspace_mode="shadow",
        inquiry_loop_mode="batched")

    def model(payload):
        view = payload["state"]
        base = {"contract_version": 1, "based_on_revision": view["based_on_revision"],
            "based_on_digest": view["based_on_digest"], "candidate_updates": [],
            "need_updates": [], "pending_decisions": []}
        if view["last_batch"] is None:
            alias = next(ref for ref, item in view["menu"].items()
                if item.get("signal") == "active_ratio"
                and item.get("measurement_kind") == "series")
            return {**base, "batch_assessments": [],
                "draft": output({"context": json.dumps({"queries": view["menu"]})},
                                (("n1", alias),)), "next_batch": [alias],
                "batch_reason": "Inspect occupancy", "decision": "continue"}
        result = view["last_batch"]["results"][0]
        return {**base, "batch_assessments": [{
            "binding_id": binding["binding_id"],
            "query_ref": result["query_fingerprint"],
            "observation_refs": result["observation_refs"],
            "status": "answered", "candidate_effect": "unclear",
            "alternative_effect": "unchanged", "remaining_gap": "Needs verification",
            "reason": "Synthetic value inspected"}
            for binding in result["bindings"]], "draft": None,
            "next_batch": [], "batch_reason": None, "decision": "handoff",
            "stop_reason": "Inquiry result reviewed"}

    policy = DirectBatchedPolicy(LangChainBatchedInquiryAdapter(RunnableLambda(model)))
    report = LangGraphInvestigation(runner, LangChainPolicyAdapter(policy)).run()
    assert report.answer.verify_digest()
    assert RecordedReplay.verify(runner.transcript.events).valid
    requests = [e for e in runner.transcript.events if e.event_type == EventType.TOOL_REQUEST]
    assert requests[0].payload["execution_source"] == "inquiry_batch"
    assert any(e.payload["execution_source"] == "verification" for e in requests[1:])
