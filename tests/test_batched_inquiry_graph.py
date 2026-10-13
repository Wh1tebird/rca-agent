"""Synthetic full graph evidence for result-driven Inquiry batches."""
import json
import os
from pathlib import Path

import pytest
from langchain_core.runnables import RunnableLambda
from test_dynamic_investigation import registry, unknown_action
from test_engineering_independent_regressions import (
    DiscoveryV7,
    EmptyPlanner,
    PublicEnvironment,
    output,
)

from rca_agent.batched_inquiry_policy import (
    BatchedInquiryInvestigationPolicy,
    LangChainBatchedInquiryAdapter,
)
from rca_agent.langchain_policy import LangChainPolicyAdapter, PolicyEnvelope
from rca_agent.langgraph_orchestrator import LangGraphInvestigation
from rca_agent.models import (
    CapabilityDescriptor,
    DataStatus,
    Modality,
    Observation,
    QueryCost,
    RequestStopAction,
    ToolResponse,
    ToolStatus,
)
from rca_agent.runner import EpisodeRunner, RecordedReplay
from rca_agent.tools.fingerprint import query_fingerprint, sha256_digest


def _emit_evidence(name, runner):
    target = os.environ.get("RCA_BATCH_EVIDENCE_DIR")
    if target is None:
        return
    directory = Path(target) / name
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "transcript.jsonl").write_text("".join(
        event.model_dump_json() + "\n" for event in runner.transcript.events))
    (directory / "answer.json").write_text(runner.answer.model_dump_json(indent=2))


def _step(state, **fields):
    return {"contract_version": 1, "based_on_revision": state["based_on_revision"],
            "based_on_digest": state["based_on_digest"],
            "batch_assessments": fields.get("assessments", []),
            "candidate_updates": fields.get("candidates", []),
            "need_updates": [], "pending_decisions": fields.get("pending", []),
            "draft": fields.get("draft"), "next_batch": fields.get("batch", []),
            "batch_reason": "Use the actual returned value" if fields.get("batch") else None,
            "decision": "continue" if fields.get("batch") else "handoff",
            "stop_reason": None if fields.get("batch") else "All selected facts reviewed"}


class DirectBatchedPolicy(BatchedInquiryInvestigationPolicy):
    """Synthetic same-role driver; Graph still owns routing and Runner admission."""

    def invoke(self, input, config=None, **kwargs):
        del config, kwargs
        state = input["context"]
        action = self._inquiry_screen(state)
        if action is not None:
            return PolicyEnvelope(action=action)
        if not self._inquiry_finished:
            raise AssertionError("Inquiry did not reach reviewed handoff")
        if not hasattr(self, "_synthetic_submitted"):
            self._synthetic_submitted = True
            return PolicyEnvelope(action=unknown_action())
        return PolicyEnvelope(action=RequestStopAction(summary="synthetic inquiry completed"))


class BatchedEnvironment(PublicEnvironment):
    fail_signal = None
    fail_status = ToolStatus.EMPTY

    def reset(self):
        base = super().reset()
        comparison = CapabilityDescriptor(tool_name="compare_metric_windows",
            modality=Modality.METRIC, available=True, query_cost=1,
            max_window_seconds=300, max_results=100,
            metrics=base.capabilities[0].metrics)
        return base.model_copy(update={"capabilities": (*base.capabilities, comparison)})

    def get_metric_series(self, request):
        response = super().get_metric_series(request)
        if request.signal == self.fail_signal:
            return response.model_copy(update={"status": self.fail_status,
                                               "observations": ()})
        return response

    def compare_metric_windows(self, request):
        before = self.remaining
        self.remaining -= 1
        self.calls.append(request.signal)
        return ToolResponse(tool_name="compare_metric_windows",
            query_ref=query_fingerprint("compare_metric_windows", request),
            status=ToolStatus.SUCCESS,
            observations=(Observation(observation_id="obs-" + sha256_digest({
                "comparison": request.signal})[:16], modality=Modality.METRIC,
                entity_ref=request.entity_ref, signal=request.signal,
                value=self.values[request.signal], data_status=DataStatus.DERIVED),),
            cost=QueryCost(units=1, budget_before=before, budget_after=self.remaining),
            message="synthetic window comparison")


@pytest.mark.parametrize("first_value,second_signal", [(0.95, "queue_ratio"), (0.15, "active_ratio")])
def test_graph_replans_from_result_and_cancels_old_pending(first_value, second_signal):
    env = BatchedEnvironment(values={"active_ratio": first_value, "queue_ratio": 0.3})
    runner = EpisodeRunner(env, evidence_registry=registry(), discovery=DiscoveryV7(),
        verification_planner=EmptyPlanner(), investigation_workspace_mode="shadow",
        inquiry_loop_mode="batched")
    calls = []

    def model(payload):
        view = payload["state"]
        calls.append(view)
        menu = view["menu"]
        if len(calls) == 1:
            aliases = [ref for ref, item in menu.items()
                       if item.get("signal") in {"active_ratio", "queue_ratio"}
                       and item.get("measurement_kind") == "series"]
            assert len(aliases) == 2
            draft = output({"context": json.dumps({"queries": menu})},
                           (("n1", aliases[0]), ("n1", aliases[1])))
            return _step(view, draft=draft, batch=[aliases[0]])
        prior = view["last_batch"]["results"]
        assert len(prior) == 1 and len(view["observations"]) >= 1
        binding = prior[0]["bindings"][0]
        value = next(o["value"] for o in view["observations"]
                     if o["observation_id"] in prior[0]["observation_refs"])
        assessment = {"binding_id": binding["binding_id"],
            "query_ref": prior[0]["query_fingerprint"],
            "observation_refs": prior[0]["observation_refs"], "status": "answered",
            "candidate_effect": "unclear", "alternative_effect": "unclear",
            "remaining_gap": "Other measurement", "reason": "The actual value was read"}
        if len(calls) == 2:
            old = next(p for p in view["pending"] if p["status"] == "pending")
            if value > 0.5:
                return _step(view, assessments=[assessment], pending=[{
                    "pending_id": old["pending_id"], "disposition": "retain",
                    "reason": "High occupancy needs queue comparison"}], batch=[old["pending_id"]])
            next_ref = next(ref for ref, item in menu.items()
                            if item.get("signal") == "active_ratio"
                            and item.get("measurement_kind") == "window_comparison")
            draft = output({"context": json.dumps({"queries": menu})},
                           (("n2", next_ref),))
            return _step(view, assessments=[assessment], pending=[{
                "pending_id": old["pending_id"], "disposition": "cancel",
                "reason": "Low occupancy makes the old queue check lower priority",
                "basis_observation_refs": prior[0]["observation_refs"]}],
                draft=draft, batch=[next_ref])
        return _step(view, assessments=[assessment])

    policy = DirectBatchedPolicy(LangChainBatchedInquiryAdapter(RunnableLambda(model)))
    report = LangGraphInvestigation(runner, LangChainPolicyAdapter(policy)).run()
    assert report.answer.verify_digest()
    assert RecordedReplay.verify(runner.transcript.events).valid
    assert len(calls) == 3
    assert env.calls == ["active_ratio", second_signal]
    state = runner.investigation_workspace_snapshot.inquiry_loop
    assert state["phase"] == "handoff" and len(state["assessments"]) == 2
    if first_value < 0.5:
        assert any(p["status"] == "cancelled" for p in state["pending"])
    handoff = runner.inquiry_loop_handoff()
    assert handoff["loop_state_digest"] == state["digest"]
    _emit_evidence("branch-high" if first_value > 0.5 else "branch-low", runner)


@pytest.mark.parametrize("shared,failed", [(False, False), (False, True), (True, False)])
def test_graph_waits_for_complete_batch_and_keeps_shared_bindings(shared, failed):
    env = BatchedEnvironment(values={"active_ratio": 0, "queue_ratio": 0.9})
    env.fail_signal = "queue_ratio" if failed else None
    runner = EpisodeRunner(env, evidence_registry=registry(), discovery=DiscoveryV7(),
        verification_planner=EmptyPlanner(), investigation_workspace_mode="active",
        inquiry_loop_mode="batched")
    calls = []

    def model(payload):
        view = payload["state"]
        calls.append(view)
        if len(calls) == 1:
            series = [ref for ref, item in view["menu"].items()
                      if item.get("measurement_kind") == "series"]
            pairs = (("n1", series[0]), ("n2", series[0] if shared else series[1]))
            draft = output({"context": json.dumps({"queries": view["menu"]})}, pairs)
            return _step(view, draft=draft,
                         batch=[series[0]] if shared else list(series))
        results = view["last_batch"]["results"]
        assert len(results) == (1 if shared else 2)
        assessments = [{"binding_id": binding["binding_id"],
            "query_ref": result["query_fingerprint"],
            "observation_refs": result["observation_refs"],
            "status": "unavailable" if result["status"] == "empty" else "answered",
            "candidate_effect": "unclear", "alternative_effect": "unclear",
            "remaining_gap": "Comparison is still tentative", "reason": "Observed zero is valid"}
            for result in results for binding in result["bindings"]]
        assert len(assessments) == 2
        return _step(view, assessments=assessments)

    policy = DirectBatchedPolicy(LangChainBatchedInquiryAdapter(RunnableLambda(model)))
    report = LangGraphInvestigation(runner, LangChainPolicyAdapter(policy)).run()
    assert report.answer.verify_digest()
    assert RecordedReplay.verify(runner.transcript.events).valid
    assert len(calls) == 2
    assert (env.calls == ["active_ratio"] if shared
            else sorted(env.calls) == ["active_ratio", "queue_ratio"])
    state = runner.investigation_workspace_snapshot.inquiry_loop
    assert len(state["assessments"]) == 2
    if failed:
        assert [result["status"] for result in state["batches"][0]["results"]].count(
            "empty") == 1
    assert len(state["batches"][0]["original_order"]) == (1 if shared else 2)
    assert len(state["batches"][0]["actual_order"]) == (1 if shared else 2)
    handoff = runner.inquiry_loop_handoff()
    assert len(handoff["query_links"]) == 2
    assert len({item["query_ref"] for item in handoff["query_links"]}) == (1 if shared else 2)
    _emit_evidence("shared" if shared else "double-empty" if failed else "double", runner)


def test_fatal_partial_batch_freezes_without_model_review():
    env = BatchedEnvironment(values={"active_ratio": 0.8, "queue_ratio": 0.1})
    env.fail_signal = "queue_ratio"
    env.fail_status = ToolStatus.QUERY_FAILED
    runner = EpisodeRunner(env, evidence_registry=registry(),
        investigation_workspace_mode="shadow", inquiry_loop_mode="batched")
    calls = []

    def model(payload):
        calls.append(payload["state"])
        assert len(calls) == 1
        view = payload["state"]
        series = [ref for ref, item in view["menu"].items()
                  if item.get("measurement_kind") == "series"]
        draft = output({"context": json.dumps({"queries": view["menu"]})},
                       (("n1", series[0]), ("n2", series[1])))
        return _step(view, draft=draft, batch=series)

    policy = DirectBatchedPolicy(LangChainBatchedInquiryAdapter(RunnableLambda(model)))
    report = LangGraphInvestigation(runner, LangChainPolicyAdapter(policy)).run()
    assert report.answer.verify_digest() and RecordedReplay.verify(runner.transcript.events).valid
    assert len(calls) == 1 and env.calls == ["active_ratio", "queue_ratio"]
    state = runner.investigation_workspace_snapshot.inquiry_loop
    assert not state["batches"][0]["assessed"]
    assert state["batches"][0]["unassessed_reason"] is not None
    _emit_evidence("fatal-partial", runner)


def test_graph_rejects_policy_mode_or_limit_mismatch_before_reset():
    from rca_agent.langchain_policy import ScriptedPolicyRunnable

    env = BatchedEnvironment()
    runner = EpisodeRunner(env, evidence_registry=registry(),
        investigation_workspace_mode="shadow", inquiry_loop_mode="batched")
    with pytest.raises(ValueError, match="loop mode"):
        LangGraphInvestigation(runner, LangChainPolicyAdapter(ScriptedPolicyRunnable(())))
    with pytest.raises(ValueError, match="query limit"):
        LangGraphInvestigation(runner, LangChainPolicyAdapter(DirectBatchedPolicy(
            LangChainBatchedInquiryAdapter(RunnableLambda(lambda _: {})), query_limit=9)))
    assert env.calls == [] and env.remaining == 20


def test_invalid_step_retries_once_without_partial_admission_or_extra_charge():
    from rca_agent.runner.transcript import EventType

    env = BatchedEnvironment()
    runner = EpisodeRunner(env, evidence_registry=registry(),
        investigation_workspace_mode="shadow", inquiry_loop_mode="batched")
    calls = []

    def model(payload):
        view = payload["state"]
        calls.append(view)
        if len(calls) <= 2:
            series = next(ref for ref, item in view["menu"].items()
                          if item.get("measurement_kind") == "series")
            draft = output({"context": json.dumps({"queries": view["menu"]})},
                           (("n1", series),))
            step = _step(view, draft=draft, batch=[series])
            if len(calls) == 1:
                step["draft"]["choices"][0]["unexpected"] = "do not persist this value"
            return step
        result = view["last_batch"]["results"][0]
        binding = result["bindings"][0]
        return _step(view, assessments=[{"binding_id": binding["binding_id"],
            "query_ref": result["query_fingerprint"],
            "observation_refs": result["observation_refs"], "status": "answered",
            "candidate_effect": "unclear", "alternative_effect": "unclear",
            "remaining_gap": "No other fact", "reason": "Observed"}])

    policy = DirectBatchedPolicy(LangChainBatchedInquiryAdapter(RunnableLambda(model)))
    report = LangGraphInvestigation(runner, LangChainPolicyAdapter(policy)).run()
    assert report.answer.verify_digest() and RecordedReplay.verify(runner.transcript.events).valid
    assert len(calls) == 3 and env.calls == ["active_ratio"] and env.remaining == 19
    attempts = [event.payload for event in runner.transcript.events
                if event.event_type == EventType.INQUIRY_LOOP_ATTEMPT_RECORDED]
    assert [attempt["status"] for attempt in attempts] == ["failed", "parsed", "parsed"]
    assert "extra" in str(attempts[0]["fields"])
    assert "do not persist this value" not in str(attempts)
    assert len([event for event in runner.transcript.events
                if event.event_type == EventType.INQUIRY_LOOP_STEP_ACCEPTED]) == 2
