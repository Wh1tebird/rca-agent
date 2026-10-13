"""v4 pending projection and repair-input contract tests."""
import json

import pytest
from langchain_core.runnables import RunnableLambda
from test_batched_inquiry_state import planning_step
from test_dynamic_investigation import registry
from test_engineering_alternatives import DiscoveryV7
from test_engineering_independent_regressions import EmptyPlanner, PublicEnvironment, output

from rca_agent.batched_inquiry_policy import (
    BatchedInquiryInvestigationPolicy,
    LangChainBatchedInquiryAdapter,
)
from rca_agent.investigation.inquiry_loop import (
    AssessmentEvidenceError,
    AssessmentReferenceError,
    LoopIdentityError,
    LoopState,
    PendingDecisionError,
    accept_loop_step,
    normalize_loop_step,
    provider_loop_schema,
)
from rca_agent.models import CallToolAction, RequestStopAction
from rca_agent.runner import EpisodeRunner, RecordedReplay


def _batched():
    env = PublicEnvironment()
    runner = EpisodeRunner(env, evidence_registry=registry(),
                           discovery=DiscoveryV7(),
                           verification_planner=EmptyPlanner(),
                           investigation_workspace_mode="shadow", inquiry_loop_mode="batched")
    runner.start()
    menu = runner.inquiry_loop_view()["menu"]
    return runner, menu


def test_v4_view_separates_actionable_pending_from_executed_history():
    runner, _ = _batched()
    assert runner.transcript.events[0].payload["inquiry_loop_version"] == 4
    view = runner.inquiry_loop_view()
    draft = output({"context": json.dumps({"queries": view["view"]["menu"]})}, (("n1", "q1"),))
    state = LoopState.model_validate(runner.investigation_workspace_snapshot.inquiry_loop)
    first = runner.record_inquiry_loop_step(
        planning_step(state, draft, "q1"), input_view=view)
    runner.handle(CallToolAction.model_validate(first["selected_actions"][0]["action"]))
    next_view = runner.inquiry_loop_view()["view"]
    executed = {item["pending_id"] for item in next_view["pending_history"]
                if item["status"] == "executed"}
    assert executed
    assert not executed & set(next_view["expected_actionable_ids"])
    assert not executed & {item["pending_id"] for item in next_view["actionable_pending"]}
    assert {item["pending_id"] for item in next_view["existing_queries"]} == {
        item["pending_id"] for item in runner.investigation_workspace_snapshot.inquiry_loop["pending"]
    }
    assert all({"pending_id", "query_fingerprint", "status"} <= set(item)
               for item in next_view["existing_queries"])
    assert runner.investigation_workspace_snapshot.inquiry_loop["pending"]
    runner.handle(RequestStopAction(summary="projection test"))


def test_executed_disposition_reports_unexpected_and_state_without_effect():
    runner, _ = _batched()
    view = runner.inquiry_loop_view()
    draft = output({"context": json.dumps({"queries": view["view"]["menu"]})}, (("n1", "q1"),))
    state = LoopState.model_validate(runner.investigation_workspace_snapshot.inquiry_loop)
    first = runner.record_inquiry_loop_step(
        planning_step(state, draft, "q1"), input_view=view)
    runner.handle(CallToolAction.model_validate(first["selected_actions"][0]["action"]))
    current = runner.inquiry_loop_view()
    executed = next(item for item in current["view"]["pending_history"]
                    if item["status"] == "executed")
    result = current["view"]["last_batch"]["results"][0]
    binding = result["bindings"][0]
    step = {
        "contract_version": 1,
        "based_on_revision": current["view"]["based_on_revision"],
        "based_on_digest": current["view"]["based_on_digest"],
        "batch_assessments": [{"binding_id": binding["binding_id"],
                                "query_ref": result["query_fingerprint"],
                                "observation_refs": result["observation_refs"],
                                "status": "ambiguous", "candidate_effect": "unclear",
                                "alternative_effect": "unclear", "remaining_gap": "test",
                                "reason": "test"}],
        "candidate_updates": [], "need_updates": [],
        "pending_decisions": [{"pending_id": executed["pending_id"],
                                "disposition": "cancel", "reason": "already done"}],
        "draft": None, "next_batch": [], "batch_reason": None,
        "decision": "handoff", "stop_reason": "test"}
    before = runner.investigation_workspace_snapshot.inquiry_loop
    with pytest.raises(PendingDecisionError) as caught:
        runner.record_inquiry_loop_step(step, input_view=current)
    assert caught.value.unexpected == (executed["pending_id"],)
    assert caught.value.invalid_states == {executed["pending_id"]: "executed"}
    assert runner.investigation_workspace_snapshot.inquiry_loop == before
    assert runner.investigation_workspace_snapshot.inquiry_loop["revision"] == before["revision"]


def test_provider_schema_scopes_outer_and_nested_versions():
    schema = provider_loop_schema()
    assert schema["properties"]["contract_version"] == {"const": 1}
    nested = schema["$defs"]["LoopDraftV2"]
    assert nested["properties"]["contract_version"] == {"const": 2}
    assert "contract_version" in schema["required"]
    assert "contract_version" in nested["required"]
    assert "Result fingerprint" in schema["$defs"]["LoopAssessment"]["properties"]["query_ref"]["description"]
    assert "Menu query ref" in schema["$defs"]["InquiryChoiceV2"]["properties"]["query_ref"]["description"]


@pytest.mark.parametrize(("code", "field"), (("DUPLICATE_PHYSICAL_QUERY", "draft.choices[].query_ref"),
                                               ("NEED_IDENTITY_CHANGED", "draft.needs[].need_id")))
def test_identity_errors_are_structured_and_actionable(code, field):
    error = LoopIdentityError(code=code, field=field, identifier="pending-x",
                              status="pending", fingerprint="qry-x",
                              instruction="use the existing identity or supersede it")
    assert error.code == code
    assert error.fields == (field,)
    assert error.conflict_id == "pending-x"
    assert error.instruction.startswith("use")


def _completed_batch():
    from test_dynamic_investigation import registry
    from test_engineering_alternatives import DiscoveryV7
    from test_engineering_independent_regressions import EmptyPlanner, PublicEnvironment
    env = PublicEnvironment()
    runner = EpisodeRunner(env, evidence_registry=registry(),
                           discovery=DiscoveryV7(), verification_planner=EmptyPlanner(),
                           investigation_workspace_mode="shadow", inquiry_loop_mode="batched")
    runner.start()
    view = runner.inquiry_loop_view()
    state = LoopState.model_validate(runner.investigation_workspace_snapshot.inquiry_loop)
    draft = output({"context": json.dumps({"queries": view["view"]["menu"]})}, (("n1", "q1"),))
    planned = runner.record_inquiry_loop_step(planning_step(state, draft, "q1"), input_view=view)
    runner.handle(CallToolAction.model_validate(planned["selected_actions"][0]["action"]))
    current = runner.inquiry_loop_view()
    result = current["view"]["last_batch"]["results"][0]
    binding = result["bindings"][0]
    return env, runner, current, result, binding


def _assessment_step(current, result, binding, *, query_ref, observation_refs):
    raw = {
        "contract_version": 1,
        "based_on_revision": current["view"]["based_on_revision"],
        "based_on_digest": current["view"]["based_on_digest"],
        "batch_assessments": [{"binding_id": binding["binding_id"], "query_ref": query_ref,
                                "observation_refs": observation_refs, "status": "ambiguous",
                                "candidate_effect": "unclear", "alternative_effect": "unclear",
                                "remaining_gap": "test", "reason": "test"}],
        "candidate_updates": [], "need_updates": [], "pending_decisions": [], "draft": None,
        "next_batch": [], "batch_reason": None, "decision": "handoff", "stop_reason": "test",
    }
    menu = {key: {**item, "action": CallToolAction.model_validate(item["action"])}
            for key, item in current["menu"].items()}
    return normalize_loop_step(raw, menu), menu


def test_assessment_menu_ref_is_rejected_with_available_result_ids_and_no_new_call():
    env, runner, current, result, binding = _completed_batch()
    step, menu = _assessment_step(current, result, binding, query_ref="aq-menu-ref",
                                  observation_refs=list(result["observation_refs"]))
    before = runner.investigation_workspace_snapshot.inquiry_loop
    with pytest.raises(AssessmentReferenceError) as caught:
        accept_loop_step(LoopState.model_validate(before), step, menu=menu,
                         batch_limit=1, query_limit=3,
                         visible_refs={o["observation_id"] for o in current["view"]["observations"]})
    assert caught.value.available_result_ids == (result["query_fingerprint"],)
    assert caught.value.expected_format.startswith("qry-")
    assert env.calls and len(env.calls) == 1
    assert runner.investigation_workspace_snapshot.inquiry_loop == before
    runner.handle(RequestStopAction(summary="query ref namespace test"))
    assert RecordedReplay.verify(runner.transcript.events).valid


def test_assessment_extra_observation_is_rejected_without_partial_update():
    env, runner, current, result, binding = _completed_batch()
    step, menu = _assessment_step(current, result, binding, query_ref=result["query_fingerprint"],
                                  observation_refs=[*result["observation_refs"], "obs-extra"])
    before = runner.investigation_workspace_snapshot.inquiry_loop
    with pytest.raises(AssessmentEvidenceError, match="subset"):
        accept_loop_step(LoopState.model_validate(before), step, menu=menu,
                         batch_limit=1, query_limit=3,
                         visible_refs={o["observation_id"] for o in current["view"]["observations"]})
    assert len(env.calls) == 1
    assert runner.investigation_workspace_snapshot.inquiry_loop == before
    runner.handle(RequestStopAction(summary="observation ref test"))
    assert RecordedReplay.verify(runner.transcript.events).valid


def test_retry_feedback_names_assessment_field_format_and_available_results():
    env, runner, current, result, binding = _completed_batch()
    calls = []

    def invoke(payload):
        calls.append(payload)
        if len(calls) == 1:
            query_ref = "aq-menu-ref"
        else:
            query_ref = result["query_fingerprint"]
        return {"contract_version": 1,
                "based_on_revision": payload["state"]["based_on_revision"],
                "based_on_digest": payload["state"]["based_on_digest"],
                "batch_assessments": [{"binding_id": binding["binding_id"],
                    "query_ref": query_ref, "observation_refs": result["observation_refs"],
                    "status": "ambiguous", "candidate_effect": "unclear",
                    "alternative_effect": "unclear", "remaining_gap": "test", "reason": "test"}],
                "candidate_updates": [], "need_updates": [], "pending_decisions": [],
                "draft": None, "next_batch": [], "batch_reason": None,
                "decision": "handoff", "stop_reason": "done"}

    policy = BatchedInquiryInvestigationPolicy(
        LangChainBatchedInquiryAdapter(RunnableLambda(invoke)), query_limit=3
    )
    context = {"inquiry_state_view": current,
               "investigation_workspace": runner.investigation_workspace_snapshot.model_dump(mode="json")}
    assert policy._inquiry_screen(context) is None
    assert len(calls) == 2
    feedback = calls[1]["repair_feedback"]
    print("DEBUG_IDENTITY_FEEDBACK", feedback)
    assert feedback["fields"] == ["batch_assessments[].query_ref"]
    assert feedback["expected_format"].startswith("qry-")
    assert feedback["available_result_ids"] == [result["query_fingerprint"]]
    assert env.calls and len(env.calls) == 1


def test_retry_feedback_explains_invalid_answered_status():
    env, runner, current, result, binding = _completed_batch()
    calls = []

    def invoke(payload):
        calls.append(payload)
        status = "answered" if len(calls) == 1 else "ambiguous"
        return {"contract_version": 1,
                "based_on_revision": payload["state"]["based_on_revision"],
                "based_on_digest": payload["state"]["based_on_digest"],
                "batch_assessments": [{"binding_id": binding["binding_id"],
                    "query_ref": result["query_fingerprint"],
                    "observation_refs": (),
                    "status": status, "candidate_effect": "unclear",
                    "alternative_effect": "unclear", "remaining_gap": "test",
                    "reason": "test"}],
                "candidate_updates": [], "need_updates": [], "pending_decisions": [],
                "draft": None, "next_batch": [], "batch_reason": None,
                "decision": "handoff", "stop_reason": "done"}

    policy = BatchedInquiryInvestigationPolicy(
        LangChainBatchedInquiryAdapter(RunnableLambda(invoke)), query_limit=3
    )
    context = {"inquiry_state_view": current,
               "investigation_workspace": runner.investigation_workspace_snapshot.model_dump(mode="json")}
    assert policy._inquiry_screen(context) is None
    assert len(calls) == 2
    feedback = calls[1]["repair_feedback"]
    assert feedback["code"] == "ASSESSMENT_EVIDENCE_INVALID"
    assert feedback["fields"] == ["batch_assessments[].status"]
    assert feedback["evidence_error"] == "incomplete_or_exploratory"
    assert "partial" in feedback["instruction"]
    assert len(env.calls) == 1


def test_repair_feedback_reaches_adapter_input():
    captured = {}

    def invoke(payload):
        captured.update(payload)
        return {
            "contract_version": 1,
            "based_on_revision": payload["state"]["based_on_revision"],
            "based_on_digest": payload["state"]["based_on_digest"],
            "pending_decisions": [], "batch_assessments": [],
            "candidate_updates": [], "need_updates": [], "draft": None,
            "next_batch": [], "batch_reason": None, "decision": "handoff",
            "stop_reason": "test",
        }

    runner, _ = _batched()
    view = runner.inquiry_loop_view()
    feedback = {"code": "PENDING_DECISION_SET_MISMATCH", "unexpected": ["p-x"]}
    adapter = LangChainBatchedInquiryAdapter(RunnableLambda(invoke))
    adapter.step({}, view, repair_feedback=feedback)
    assert json.loads(captured["repair_feedback_json"]) == feedback
    assert feedback == captured["repair_feedback"]
