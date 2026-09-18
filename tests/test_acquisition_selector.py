import json

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from rca_agent.environment.rca100 import Rca100Environment
from rca_agent.langchain_acquisition import (
    InquiryOutputError,
    LangChainAcquisitionSelector,
    LangChainInquiryAdapter,
    inquiry_candidate_records,
    query_options,
)


def test_ai_can_select_new_bounded_metric_and_runner_executes(cases_root):
    from rca_agent.models import RequestStopAction
    from rca_agent.runner import EpisodeRunner, RecordedReplay

    runner = EpisodeRunner(Rca100Environment.from_case(cases_root, "t007"))
    bootstrap = runner.start()
    context = {"bootstrap": bootstrap.model_dump(mode="json")}
    options = query_options(context, [context["bootstrap"]["alert_entity"]])
    chosen = next(ref for ref, item in options.items()
                  if item["action"].tool_name == "compare_metric_windows"
                  and item["action"].request.signal == "request_count")
    seen = []

    def select(payload):
        seen.append(json.loads(payload["context"]))
        return {"query_refs": [chosen]}

    selector = LangChainAcquisitionSelector(RunnableLambda(select))
    actions = selector.select({"alert": bootstrap.alert_summary}, options)
    response = runner.handle(actions[0])
    assert response.cost.units == 1
    assert response.cost.budget_after == bootstrap.query_budget.initial - 1
    assert seen[0]["queries"][chosen]["action"]["request"]["entity_ref"] == (
        bootstrap.alert_entity.entity_ref
    )
    runner.handle(RequestStopAction(summary="test complete"))
    assert RecordedReplay.verify(runner.transcript.events).valid


@pytest.mark.parametrize("refs", [["unknown"], ["one", "one"], [str(i) for i in range(6)]])
def test_ai_cannot_invent_duplicate_or_unbound_queries(refs):
    selector = LangChainAcquisitionSelector(RunnableLambda(lambda _: {"query_refs": refs}))
    with pytest.raises(ValueError):
        selector.select({}, {})


def test_ai_screen_is_bounded_and_does_not_execute_tools(cases_root):
    from evaluation.unknown_benchmark import HoldoutAcquisitionPolicy

    bootstrap = Rca100Environment.from_case(cases_root, "t007").reset()
    context = {"bootstrap": bootstrap.model_dump(mode="json"), "history": []}

    def select(payload):
        options = json.loads(payload["context"])["queries"]
        return {"query_refs": list(options)[:5]}

    policy = HoldoutAcquisitionPolicy(LangChainAcquisitionSelector(RunnableLambda(select)))
    actions = [policy._ai_screen(context) for _ in range(12)]
    assert sum(a is not None for a in actions) == 10
    assert policy.selection_calls == 2
    assert len({a.model_dump_json() for a in actions if a is not None}) == 10
    assert bootstrap.query_budget.remaining == 20


def test_inquiry_binds_query_to_need_and_replans_after_round(cases_root):
    from evaluation.unknown_benchmark import HoldoutAcquisitionPolicy

    bootstrap = Rca100Environment.from_case(cases_root, "t007").reset()
    context = {"bootstrap": bootstrap.model_dump(mode="json"), "history": []}
    seen = []

    def plan(payload):
        data = json.loads(payload["context"])
        seen.append(data["incident"])
        refs = [ref for ref, item in data["queries"].items()
                if item["signal"] == "request_count"][:1]
        if not refs:
            refs = [ref for ref, item in data["queries"].items()
                    if item["signal"] != "related_entity"][:1]
        assert refs
        return {
            "hypotheses": [{"hypothesis_id": "h1", "mechanism": "traffic surge",
                            "alternative": "slow downstream service"}],
            "needs": [{"need_id": "n1", "hypothesis_id": "h1",
                       "question": "Did traffic rise?", "expected_if_true": "yes",
                       "expected_if_alternative": "not necessarily"}],
            "choices": [{"need_id": "n1", "query_ref": refs[0]}],
        }

    policy = HoldoutAcquisitionPolicy(inquiry=LangChainInquiryAdapter(RunnableLambda(plan)))
    first = policy._inquiry_screen(context)
    assert first.request.signal == "request_count"
    assert policy.inquiry_drafts[0]["draft"]["needs"][0]["question"] == "Did traffic rise?"
    assert policy.inquiry_drafts[0]["selected_actions"][0]["signal"] == "request_count"
    context["history"] = [{"result": {"value": {"observations": [
        {"signal": "request_count", "entity_ref": bootstrap.alert_entity.entity_ref,
         "data_status": "observed", "value": 9},
    ]}}}]
    second = policy._inquiry_screen(context)
    assert seen[1]["recent_observations"][0]["value"] == "9"
    assert "choices" not in seen[1]["previous_inquiry"]
    assert policy.selection_calls == 2
    assert second is not None
    assert first.model_dump_json() != second.model_dump_json()
    assert not policy.selection_errors


def test_inquiry_query_limit_reserves_capacity_for_verification(cases_root):
    from evaluation.unknown_benchmark import HoldoutAcquisitionPolicy

    bootstrap = Rca100Environment.from_case(cases_root, "t007").reset()
    context = {"bootstrap": bootstrap.model_dump(mode="json"), "history": []}

    def plan(payload):
        refs = list(json.loads(payload["context"])["queries"])[:2]
        return {
            "hypotheses": [{"hypothesis_id": "h1", "mechanism": "pool saturation",
                            "alternative": "database slowness"}],
            "needs": [{"need_id": "n1", "hypothesis_id": "h1", "question": "Which cause?",
                       "expected_if_true": "pool busy", "expected_if_alternative": "DB slow"}],
            "choices": [{"need_id": "n1", "query_ref": ref} for ref in refs],
        }

    policy = HoldoutAcquisitionPolicy(
        inquiry=LangChainInquiryAdapter(RunnableLambda(plan)), inquiry_query_limit=1,
    )
    assert policy._inquiry_screen(context) is not None
    assert policy._inquiry_screen(context) is None
    assert len(policy.selected_actions) == 1
    assert len(policy.inquiry_drafts[0]["selected_actions"]) == 1


def test_inquiry_rejects_unoffered_query(cases_root):
    bootstrap = Rca100Environment.from_case(cases_root, "t007").reset()
    context = {"bootstrap": bootstrap.model_dump(mode="json")}
    options = query_options(context, [context["bootstrap"]["alert_entity"]])
    invalid = {
        "hypotheses": [{"hypothesis_id": "h1", "mechanism": "traffic surge",
                        "alternative": "slow dependency"}],
        "needs": [{"need_id": "n1", "hypothesis_id": "h1", "question": "why?",
                   "expected_if_true": "up", "expected_if_alternative": "flat"}],
        "choices": [{"need_id": "n1", "query_ref": "invented"}],
    }
    adapter = LangChainInquiryAdapter(RunnableLambda(lambda _: invalid))
    with pytest.raises(ValueError, match="not offered"):
        adapter.plan({}, options)


def test_inquiry_retries_one_invalid_binding_without_executing_it(cases_root):
    from evaluation.unknown_benchmark import HoldoutAcquisitionPolicy

    bootstrap = Rca100Environment.from_case(cases_root, "t007").reset()
    context = {"bootstrap": bootstrap.model_dump(mode="json"), "history": []}
    calls = 0

    def plan(payload):
        nonlocal calls
        calls += 1
        data = json.loads(payload["context"])
        ref = next(iter(data["queries"]))
        return {
            "hypotheses": [{"hypothesis_id": "h1", "mechanism": "traffic surge",
                            "alternative": "slow dependency"}],
            "needs": [{"need_id": "n1", "hypothesis_id": "h1", "question": "why?",
                       "expected_if_true": "up", "expected_if_alternative": "flat"}],
            "choices": [{"need_id": "n1", "query_ref": "wrong" if calls == 1 else ref}],
        }

    policy = HoldoutAcquisitionPolicy(inquiry=LangChainInquiryAdapter(RunnableLambda(plan)))
    action = policy._inquiry_screen(context)
    assert action is not None
    assert calls == 2
    assert policy.selection_errors == ["inquiry choice is not offered"]
    assert len(policy.selected_actions) == 1


def test_inquiry_reports_schema_failure_without_provider_payload_leak(cases_root):
    bootstrap = Rca100Environment.from_case(cases_root, "t007").reset()
    context = {"bootstrap": bootstrap.model_dump(mode="json")}
    options = query_options(context, [context["bootstrap"]["alert_entity"]])
    adapter = LangChainInquiryAdapter(RunnableLambda(lambda _: {
        "raw": AIMessage(content=json.dumps({"hypotheses": [], "needs": [], "choices": []})),
        "parsed": None,
        "parsing_error": ValueError("provider secret"),
    }))
    with pytest.raises(InquiryOutputError) as raised:
        adapter.plan({}, options)
    assert raised.value.code == "schema_mismatch"
    assert any("hypotheses" in field for field in raised.value.fields)
    assert "provider secret" not in str(raised.value.diagnostic())


def test_inquiry_uses_first_two_queries_when_model_offers_more(cases_root):
    bootstrap = Rca100Environment.from_case(cases_root, "t007").reset()
    context = {"bootstrap": bootstrap.model_dump(mode="json")}
    options = query_options(context, [context["bootstrap"]["alert_entity"]])

    def plan(payload):
        refs = list(json.loads(payload["context"])["queries"])[:3]
        return {
            "hypotheses": [{"hypothesis_id": "h1", "mechanism": "traffic surge",
                            "alternative": "slow dependency"}],
            "needs": [{"need_id": "n1", "hypothesis_id": "h1", "question": "which metric?",
                       "expected_if_true": "high", "expected_if_alternative": "normal"}],
            "choices": [{"need_id": "n1", "query_ref": ref} for ref in refs],
        }

    draft, actions = LangChainInquiryAdapter(RunnableLambda(plan)).plan({}, options)
    assert len(draft.choices) == len(actions) == 2
    assert all(choice.query_ref in options for choice in draft.choices)


def test_inquiry_handoff_reaches_discovery_and_replays(cases_root):
    from evaluation.unknown_benchmark import HoldoutAcquisitionPolicy
    from rca_agent.core.dynamic import InquiryDisposition, MechanismDiscoveryPayload
    from rca_agent.core.runtime_registry import rca100_evidence_registry
    from rca_agent.langchain_policy import LangChainPolicyAdapter
    from rca_agent.langgraph_orchestrator import LangGraphInvestigation
    from rca_agent.runner import CoreDecisionProvider, EpisodeRunner, RecordedReplay

    class CaptureDiscovery:
        def discover(self, context):
            self.context = context
            return MechanismDiscoveryPayload(inquiry_dispositions=(InquiryDisposition(
                hypothesis_id="h1", disposition="discarded",
                reason="The issued observations do not establish the proposed queue mechanism.",
            ),))

    def plan(payload):
        queries = json.loads(payload["context"])["queries"]
        ref = next(ref for ref, item in queries.items()
                   if item["signal"] != "related_entity")
        return {
            "hypotheses": [{"hypothesis_id": "h1", "mechanism": "thread pool backlog",
                            "alternative": "slow downstream service"}],
            "needs": [{"need_id": "n1", "hypothesis_id": "h1",
                       "question": "Is the pool queue growing?",
                       "expected_if_true": "queue rises",
                       "expected_if_alternative": "queue may remain flat"}],
            "choices": [{"need_id": "n1", "query_ref": ref}],
        }

    discovery = CaptureDiscovery()
    acquisition = HoldoutAcquisitionPolicy(inquiry=LangChainInquiryAdapter(RunnableLambda(plan)))
    runner = EpisodeRunner(
        Rca100Environment.from_case(cases_root, "t007"), max_rounds=30,
        decision_provider=CoreDecisionProvider(frozenset({"threadExhaustion"})),
        discovery=discovery, verification_planner=object(),
        evidence_registry=rca100_evidence_registry,
    )
    report = LangGraphInvestigation(
        runner, LangChainPolicyAdapter(acquisition),
        discovery_notes_provider=acquisition.discovery_handoff,
        inquiry_rounds_provider=lambda: tuple(acquisition.inquiry_drafts),
    ).run(recursion_limit=100)
    handoff = discovery.context["inquiry_handoff"]
    assert handoff["hypotheses"][0]["mechanism"] == "thread pool backlog"
    assert handoff["query_links"]
    assert handoff["query_links"][0]["question"] == "Is the pool queue growing?"
    assert handoff["query_links"][0]["hypothesis_id"] == "h1"
    completed = next(event.payload for event in runner.transcript.events
                     if event.event_type.value == "discovery_completed")
    assert completed["inquiry_dispositions"][0]["disposition"] == "discarded"
    assert report.answer is not None
    assert report.answer.inquiry_candidates[0].mechanism == "thread pool backlog"
    assert report.answer.inquiry_candidates[0].discovery_disposition == "discarded"
    assert report.answer.inquiry_candidates[0].observation_refs
    assert sum(event.event_type.value == "inquiry_round_recorded"
               for event in runner.transcript.events) == len(acquisition.inquiry_drafts)
    assert RecordedReplay.verify(runner.transcript.events).valid


def test_inquiry_preserves_early_idea_when_later_round_reuses_model_id():
    drafts = [
        {"round": 1, "draft": {"hypotheses": [
            {"hypothesis_id": "h1", "mechanism": "thread pool exhaustion",
             "alternative": "slow database"},
        ]}},
        {"round": 2, "draft": {"hypotheses": [
            {"hypothesis_id": "h1", "mechanism": "database latency",
             "alternative": "thread pool exhaustion"},
        ]}},
        {"round": 3, "draft": {"hypotheses": [
            {"hypothesis_id": "other", "mechanism": "thread pool exhaustion",
             "alternative": "slow database"},
        ]}},
    ]
    records = inquiry_candidate_records(drafts)
    assert len(records) == 2
    assert records[0].hypothesis.mechanism == "thread pool exhaustion"
    assert (records[0].first_round, records[0].last_round, records[0].mention_count) == (1, 3, 2)
    assert records[1].hypothesis.hypothesis_id != records[0].hypothesis.hypothesis_id


def test_recorded_inquiry_survives_when_discovery_never_runs(cases_root):
    from rca_agent.models import RequestStopAction
    from rca_agent.runner import EpisodeRunner, RecordedReplay

    runner = EpisodeRunner(Rca100Environment.from_case(cases_root, "t007"))
    runner.start()
    runner.record_inquiry_round({
        "round": 2,  # The first model call may have failed before producing a valid draft.
        "draft": {
            "hypotheses": [{"hypothesis_id": "h1", "mechanism": "pool saturation",
                            "alternative": "downstream latency"}],
            "needs": [{"need_id": "n1", "hypothesis_id": "h1", "question": "Is queue high?",
                       "expected_if_true": "yes", "expected_if_alternative": "maybe"}],
            "choices": [{"need_id": "n1", "query_ref": "q1"}],
        },
    })
    answer = runner.handle(RequestStopAction(summary="No discovery available"))
    assert answer.inquiry_candidates[0].mechanism == "pool saturation"
    assert answer.inquiry_candidates[0].first_round == 2
    assert answer.inquiry_candidates[0].discovery_disposition == "unreported"
    assert RecordedReplay.verify(runner.transcript.events).valid
