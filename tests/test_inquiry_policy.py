from rca_agent.inquiry_policy import _history_observations


def test_failed_inquiry_cannot_silently_enter_discovery():
    import pytest

    from rca_agent.inquiry_policy import InquiryInvestigationPolicy, InquiryUnavailableError
    from rca_agent.langchain_acquisition import InquiryOutputError

    class Broken:
        def plan(self, incident, options):
            if "validation_error" in incident:
                assert "choices.query_ref" in incident["validation_error"]
            raise InquiryOutputError("inquiry choice is not offered", fields=("choices.query_ref",))

    policy = InquiryInvestigationPolicy(Broken())
    policy.selection_calls = 1
    with pytest.raises(InquiryUnavailableError):
        policy._plan_inquiry_round({}, {})
    records = policy.drain_inquiry_diagnostics()
    assert [record["attempt"] for record in records] == [1, 2]
    assert all(record["status"] == "failed" for record in records)
    assert not policy.inquiry_drafts
    assert not policy.drain_inquiry_diagnostics()


def test_inquiry_attempts_are_safe_and_replay_valid(cases_root):
    import pytest

    from rca_agent.environment import Rca100Environment
    from rca_agent.models import RequestStopAction
    from rca_agent.runner import EpisodeRunner, RecordedReplay

    runner = EpisodeRunner(Rca100Environment.from_case(cases_root, "t007"))
    runner.start()
    runner.record_inquiry_attempt({"round": 1, "attempt": 1, "status": "failed",
                                   "code": "schema_mismatch", "fields": ["choices:too_short"]})
    with pytest.raises(ValueError):
        runner.record_inquiry_attempt({"round": 1, "attempt": 1, "status": "failed",
                                       "code": "sk-secret", "fields": []})
    runner.handle(RequestStopAction(summary="local test"))
    assert RecordedReplay.verify(runner.transcript.events).valid


def test_inquiry_failed_attempt_diagnostics_survive_graph_termination(cases_root):
    from rca_agent.environment import Rca100Environment
    from rca_agent.inquiry_policy import InquiryInvestigationPolicy, InquiryUnavailableError
    from rca_agent.langchain_policy import LangChainPolicyAdapter
    from rca_agent.langgraph_orchestrator import LangGraphInvestigation
    from rca_agent.runner import EpisodeRunner, EventType, RecordedReplay

    class BrokenPolicy(InquiryInvestigationPolicy):
        def invoke(self, input, config=None, **kwargs):
            self._diagnostic_events.append({"round": 1, "attempt": 1, "status": "failed",
                                            "code": "empty_response", "fields": []})
            raise InquiryUnavailableError("no safe draft")

    policy = BrokenPolicy(None)
    runner = EpisodeRunner(Rca100Environment.from_case(cases_root, "t007"))
    report = LangGraphInvestigation(runner, LangChainPolicyAdapter(policy)).run()
    assert report.answer.verify_digest()
    assert any(e.event_type == EventType.INQUIRY_ATTEMPT_RECORDED
               for e in runner.transcript.events)
    assert not any(e.event_type == EventType.DISCOVERY_STARTED for e in runner.transcript.events)
    assert RecordedReplay.verify(runner.transcript.events).valid


def test_history_observations_skips_non_tool_results_with_null_values():
    history = (
        {"result": {"kind": "candidate_evidence_updated", "value": None}},
        {
            "result": {
                "kind": "tool_response",
                "value": {"observations": ({"signal": "sql_shape"},)},
            }
        },
    )

    assert _history_observations(history) == [{"signal": "sql_shape"}]


def test_suppressed_inquiry_query_does_not_consume_executed_query_count(cases_root):
    from rca_agent.environment.rca100 import Rca100Environment
    from rca_agent.inquiry_policy import InquiryInvestigationPolicy
    from rca_agent.langchain_acquisition import query_options
    from rca_agent.models import CallToolAction

    bootstrap = Rca100Environment.from_case(cases_root, "t007").reset()
    entity = bootstrap.alert_entity.model_dump(mode="json")
    options = query_options({"bootstrap": bootstrap.model_dump(mode="json")}, [entity])
    action = CallToolAction.model_validate(next(
        item["action"] for item in options.values()
        if getattr(item["action"].request, "signal", None) == "request_count"
        and item["action"].tool_name.value == "compare_metric_windows"
    ))
    policy = InquiryInvestigationPolicy(None, query_limit=1)  # type: ignore[arg-type]
    policy.selected_actions.append(action)
    policy._pending_selected_action = action
    policy.record_routed_action(action, None)
    assert policy._query_count() == 0
    policy._pending_selected_action = action
    policy.record_routed_action(action, action)
    assert policy._query_count() == 1
