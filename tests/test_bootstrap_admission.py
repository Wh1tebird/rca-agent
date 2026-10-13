"""Host-selected startup grants, exercised through actual public Runner entries."""
from copy import deepcopy
from datetime import timedelta

import pytest
from test_batched_inquiry_graph import BatchedEnvironment
from test_dynamic_investigation import registry

from rca_agent.models import (
    CallToolAction,
    DataStatus,
    ListRelatedEntitiesRequest,
    Modality,
    Observation,
    QueryCost,
    RequestStopAction,
    ToolName,
    ToolResponse,
    ToolStatus,
)
from rca_agent.runner import EpisodeRunner, RecordedReplay, ReplayValidationError
from rca_agent.runner.episode import ActionRejectedError
from rca_agent.runner.transcript import GENESIS_DIGEST, EventType, TranscriptEvent
from rca_agent.tools.fingerprint import query_fingerprint, sha256_digest


class StartupFacts(BatchedEnvironment):
    def __init__(self, *, trace=True, participant=False):
        super().__init__(values={"active_ratio": .42})
        self.trace = trace
        self.participant = participant

    def _response(self, tool, request, observations=()):
        before = self.remaining
        self.remaining -= 1
        self.calls.append(tool.value)
        return ToolResponse(tool_name=tool, query_ref=query_fingerprint(tool, request),
            status=ToolStatus.SUCCESS if observations else ToolStatus.EMPTY,
            observations=observations, message="Synthetic startup facts",
            cost=QueryCost(units=1, budget_before=before, budget_after=self.remaining))

    def get_service_trace_summary(self, request):
        observations = (Observation(observation_id="obs-" + "a" * 16,
            modality=Modality.TRACE, entity_ref=request.entity_ref,
            signal="operation_trace_duration_ms", value=480,
            data_status=DataStatus.OBSERVED, details={"trace_ref": "trace-" + "a" * 16}),
        ) if self.trace else (Observation(observation_id="obs-" + "c" * 16,
            modality=Modality.TRACE, entity_ref=request.entity_ref, signal="related_entity",
            value={"entity_ref": "ent-operation", "display_name": "operation",
                   "entity_type": "apm.operation"}, data_status=DataStatus.OBSERVED),)
        return self._response(ToolName.GET_SERVICE_TRACE_SUMMARY, request, observations)

    def get_trace_stage_breakdown(self, request):
        observations = (Observation(observation_id="obs-" + "b" * 16,
            modality=Modality.TRACE, entity_ref="ent-service", signal="trace_participant",
            value={"entity_ref": "ent-downstream-service", "display_name": "downstream",
                   "entity_type": "apm.service"}, data_status=DataStatus.DERIVED,
            details={"trace_ref": request.trace_ref}),) if self.participant else ()
        return self._response(ToolName.GET_TRACE_STAGE_BREAKDOWN, request, observations)

    def list_related_entities(self, request):
        return self._response(ToolName.LIST_RELATED_ENTITIES, request)


def startup_runner(mode="shadow", **facts):
    env = StartupFacts(**facts)
    runner = EpisodeRunner(env, evidence_registry=registry(),
                           investigation_workspace_mode=mode, inquiry_loop_mode="batched")
    runner.start()
    return env, runner


def unchanged_rejection(env, runner, invoke):
    before = (tuple(env.calls), env.remaining, runner.transcript.events,
              runner.investigation_workspace_snapshot.model_dump(mode="json"))
    with pytest.raises(ActionRejectedError):
        invoke()
    assert before == (tuple(env.calls), env.remaining, runner.transcript.events,
                      runner.investigation_workspace_snapshot.model_dump(mode="json"))


@pytest.mark.parametrize("mode", ["shadow", "active"])
@pytest.mark.parametrize("entry", ["startup", "handle", "private", "grant"])
def test_no_selection_is_not_authorization(mode, entry):
    env, runner = startup_runner(mode)
    from rca_agent.investigation.inquiry_loop import LoopState
    from rca_agent.investigation.startup_authorization import selected_startup_action

    action = selected_startup_action(
        LoopState.model_validate(runner.investigation_workspace_snapshot.inquiry_loop),
        runner.bootstrap, runner.transcript.events)
    invoke = {"startup": lambda: runner.handle_startup_query(action),
              "handle": lambda: runner.handle(action),
              "private": lambda: runner._call_tool(action),
              "grant": lambda: runner._handle_granted_query(action, "startup")}[entry]
    unchanged_rejection(env, runner, invoke)


@pytest.mark.parametrize("mode", ["shadow", "active"])
@pytest.mark.parametrize("changed", ["entity", "window", "limit", "duration", "other", "planner"])
def test_selected_action_is_bound_in_full_without_partial_consumption(mode, changed):
    env, runner = startup_runner(mode)
    selected = runner.select_startup_query()
    assert runner.select_startup_query() == selected  # Does not create another grant.
    updates = {"entity": {"entity_ref": "ent-other"}, "limit": {"max_traces": 3},
               "duration": {"min_duration_ms": 101}, "window": {"window":
                   selected.request.window.model_copy(update={"start":
                       selected.request.window.start + timedelta(seconds=1)})}}
    action = selected.model_copy(update={"request": selected.request.model_copy(
        update=updates[changed])}) if changed in updates else CallToolAction(
            tool_name=ToolName.LIST_RELATED_ENTITIES, request=ListRelatedEntitiesRequest(
                entity_ref=runner.bootstrap.alert_entity.entity_ref, max_results=50))
    invoke = (lambda: runner.verify_feedback_query("h1", "active_ratio", action)) if (
        changed == "planner") else lambda: runner.handle_startup_query(action)
    unchanged_rejection(env, runner, invoke)
    runner.handle_startup_query(selected)
    assert env.remaining == 19 and len(env.calls) == 1
    unchanged_rejection(env, runner, lambda: runner.handle_startup_query(selected))
    runner.handle(RequestStopAction(summary="startup complete"))
    assert RecordedReplay.verify(runner.transcript.events).valid


@pytest.mark.parametrize("mode", ["shadow", "active"])
@pytest.mark.parametrize("trace,participant,next_tool", [
    (False, False, ToolName.LIST_RELATED_ENTITIES),
    (True, False, ToolName.GET_TRACE_STAGE_BREAKDOWN),
    (True, True, ToolName.GET_TRACE_STAGE_BREAKDOWN),
])
def test_observations_determine_followup_and_stop_startup(mode, trace, participant, next_tool):
    env, runner = startup_runner(mode, trace=trace, participant=participant)
    runner.handle_startup_query(runner.select_startup_query())
    next_action = runner.select_startup_query()
    assert next_action.tool_name == next_tool
    other = CallToolAction(tool_name=ToolName.LIST_RELATED_ENTITIES,
        request=ListRelatedEntitiesRequest(entity_ref="ent-downstream-service", max_results=50))
    unchanged_rejection(env, runner, lambda: runner.handle_startup_query(other))
    runner.handle_startup_query(next_action)
    if participant:
        assert runner.select_startup_query() is None
        unchanged_rejection(env, runner, lambda: runner.handle_startup_query(other))
    runner.handle(RequestStopAction(summary="startup observations reviewed"))
    assert RecordedReplay.verify(runner.transcript.events).valid


def test_old_state_selection_expires_before_execution():
    env, runner = startup_runner()
    action = runner.select_startup_query()
    runner.reject_raw_action({"source": "synthetic"})
    unchanged_rejection(env, runner, lambda: runner.handle_startup_query(action))
    assert runner.select_startup_query() == action
    runner.handle_startup_query(action)
    runner.handle(RequestStopAction(summary="old selection invalidated"))
    assert RecordedReplay.verify(runner.transcript.events).valid


@pytest.mark.parametrize("stale", [False, True])
def test_caller_supplied_authorization_event_cannot_create_live_host_selection(stale):
    env, runner = startup_runner()
    from rca_agent.investigation.inquiry_loop import LoopState
    from rca_agent.investigation.startup_authorization import selected_startup_action

    action = selected_startup_action(
        LoopState.model_validate(runner.investigation_workspace_snapshot.inquiry_loop),
        runner.bootstrap, runner.transcript.events)
    if stale:
        assert runner.select_startup_query() == action
        runner.reject_raw_action({"source": "synthetic stale selection"})
    runner.transcript.append(EventType.STARTUP_ACTION_SELECTED, {
        "based_on_digest": runner.transcript.events[-1].digest,
        "action": action.model_dump(mode="json")})
    unchanged_rejection(env, runner, lambda: runner.handle_startup_query(action))
    unchanged_rejection(env, runner, runner.select_startup_query)


@pytest.mark.parametrize("mode", ["shadow", "active"])
def test_real_graph_startup_then_reviewed_inquiry(mode):
    import json

    from langchain_core.runnables import RunnableLambda
    from test_batched_inquiry_graph import _step
    from test_engineering_independent_regressions import output

    from rca_agent.batched_inquiry_policy import (
        BatchedInquiryInvestigationPolicy,
        LangChainBatchedInquiryAdapter,
    )
    from rca_agent.langchain_policy import LangChainPolicyAdapter
    from rca_agent.langgraph_orchestrator import LangGraphInvestigation

    env = StartupFacts(participant=True)
    runner = EpisodeRunner(env, evidence_registry=registry(),
                           investigation_workspace_mode=mode, inquiry_loop_mode="batched")

    def model(payload):
        view = payload["state"]
        if view["last_batch"] is None:
            alias = next(ref for ref, item in view["menu"].items()
                if item.get("signal") == "active_ratio" and item.get("measurement_kind") == "series")
            return _step(view, draft=output({"context": json.dumps({"queries": view["menu"]})},
                                           (("n1", alias),)), batch=[alias])
        result = view["last_batch"]["results"][0]
        return _step(view, assessments=[{"binding_id": b["binding_id"],
            "query_ref": result["query_fingerprint"], "observation_refs": result["observation_refs"],
            "status": "answered", "candidate_effect": "unclear", "alternative_effect": "unclear",
            "remaining_gap": "No further synthetic query", "reason": "Actual result reviewed"}
            for b in result["bindings"]])

    policy = BatchedInquiryInvestigationPolicy(LangChainBatchedInquiryAdapter(RunnableLambda(model)))
    report = LangGraphInvestigation(runner, LangChainPolicyAdapter(policy)).run()
    assert report.answer.verify_digest() and RecordedReplay.verify(runner.transcript.events).valid
    assert env.calls == ["get_service_trace_summary", "get_trace_stage_breakdown", "active_ratio"]
    assert runner.investigation_workspace_snapshot.inquiry_loop["phase"] == "handoff"


def rechain(events):
    rewritten, replacement = [], {}

    def swap(value):
        if isinstance(value, str):
            return replacement.get(value, value)
        if isinstance(value, list):
            return [swap(v) for v in value]
        if isinstance(value, dict):
            return {k: swap(v) for k, v in value.items()}
        return value

    for event in events:
        fields = event.model_dump(mode="json", exclude={"digest"})
        payload = swap(deepcopy(fields["payload"]))
        head = rewritten[-1].digest if rewritten else GENESIS_DIGEST
        if event.event_type == EventType.ANSWER_FROZEN:
            payload["transcript_digest"] = head
            if payload.get("investigation_workspace"):
                payload["investigation_workspace"]["transcript_head_digest"] = head
            payload["artifact_digest"] = sha256_digest(
                {k: v for k, v in payload.items() if k != "artifact_digest"})
        fields.update(payload=payload, previous_digest=head, sequence=len(rewritten) + 1)
        rebuilt = TranscriptEvent(**fields, digest=sha256_digest(fields))
        rewritten.append(rebuilt)
        replacement[event.digest] = rebuilt.digest
    return rewritten


@pytest.mark.parametrize("attack", ["missing", "late", "duplicate", "action", "basis",
                                  "source", "version_null", "version_unknown", "consumed",
                                  "request", "version_missing", "mid_episode_version", "after_freeze"])
def test_replay_rebuilds_host_selection_after_rehashing(attack):
    _, runner = startup_runner()
    action = runner.select_startup_query()
    runner.handle_startup_query(action)
    runner.handle(RequestStopAction(summary="startup replay consistency"))
    events = list(runner.transcript.events)
    index = next(i for i, e in enumerate(events) if e.event_type == EventType.STARTUP_ACTION_SELECTED)
    if attack == "missing":
        events.pop(index)
    elif attack == "late":
        selected = events.pop(index)
        request = next(i for i, e in enumerate(events) if e.event_type == EventType.TOOL_REQUEST)
        events.insert(request + 1, selected)
    elif attack == "duplicate":
        events.insert(index + 1, events[index])
    elif attack == "consumed":
        begin = next(i for i, e in enumerate(events) if e.event_type == EventType.POLICY_ACTION)
        end = next(i for i, e in enumerate(events) if e.event_type == EventType.QUERY_COST)
        events[end + 1:end + 1] = events[begin:end + 1]
    elif attack == "mid_episode_version":
        events.insert(index + 1, events[0])
    elif attack == "after_freeze":
        events.append(events[index])
    else:
        target = index
        payload = deepcopy(events[target].payload)
        if attack == "action":
            payload["action"]["request"]["max_traces"] = 3
        elif attack == "basis":
            payload["based_on_digest"] = GENESIS_DIGEST
        elif attack == "source":
            target = next(i for i, e in enumerate(events) if e.event_type == EventType.TOOL_REQUEST)
            payload = {**events[target].payload, "execution_source": "verification"}
        elif attack == "request":
            target = next(i for i, e in enumerate(events) if e.event_type == EventType.TOOL_REQUEST)
            payload = deepcopy(events[target].payload)
            payload["request"]["max_traces"] = 3
            request = action.request.model_copy(update={"max_traces": 3})
            payload["query_ref"] = query_fingerprint(action.tool_name, request)
        else:
            target = 0
            payload = {**events[0].payload, "inquiry_loop_version":
                       None if attack == "version_null" else 99}
            if attack == "version_missing":
                payload.pop("inquiry_loop_version")
        events[target] = events[target].model_copy(update={"payload": payload})
    rewritten = rechain(events)
    assert all(e.verify_digest() for e in rewritten)
    from rca_agent.models import AnswerArtifact
    assert AnswerArtifact.model_validate(next(e.payload for e in rewritten
        if e.event_type == EventType.ANSWER_FROZEN)).verify_digest()
    with pytest.raises(ReplayValidationError):
        RecordedReplay.verify(rewritten)
