from __future__ import annotations

import json
from pathlib import Path

import pytest

from rca_agent.environment.rca100 import Rca100Environment
from rca_agent.models import EpisodeStatus, RequestStopAction, StopReason
from rca_agent.runner import (
    EpisodeRunner,
    EventType,
    RecordedReplay,
    ScriptedPolicyProcessHost,
    ScriptedPolicySpec,
)


def _runner(cases_root: Path, transcript_path: Path | None = None) -> EpisodeRunner:
    return EpisodeRunner(
        Rca100Environment.from_case(cases_root, "t025", query_budget=3),
        transcript_path=transcript_path,
    )


def test_policy_runs_in_sanitized_process_with_only_serialized_context(
    cases_root: Path,
    tmp_path: Path,
):
    runner = _runner(cases_root, tmp_path / "policy.jsonl")
    runner.start()
    policy = ScriptedPolicySpec(
        actions=(
            RequestStopAction(summary="scripted process lifecycle check").model_dump(mode="json"),
        )
    )

    report = ScriptedPolicyProcessHost().run(runner, policy)

    assert runner.status == EpisodeStatus.ANSWER_FROZEN
    assert report.exit_kind == "completed"
    assert set(report.visible_context) == {
        "bootstrap",
        "environment_keys",
        "working_directory",
    }
    assert set(report.visible_context["bootstrap"]) == {
        "event_time",
        "alert_summary",
        "alert_entity",
        "observation_horizon",
        "capabilities",
        "query_budget",
    }
    assert not any(
        key.startswith("RCA100") or "EVALUATOR" in key
        for key in report.visible_context["environment_keys"]
    )
    payload = json.dumps(report.visible_context, ensure_ascii=False)
    assert str(cases_root) not in payload
    assert "t025" not in payload
    assert RecordedReplay.from_path(tmp_path / "policy.jsonl").valid


def test_invalid_policy_action_is_rejected_and_never_executed(cases_root: Path):
    runner = _runner(cases_root)
    policy = ScriptedPolicySpec(
        actions=(
            {
                "action_type": "call_tool",
                "tool_name": "execute_sql",
                "request": {"sql": "select secret"},
            },
        )
    )

    report = ScriptedPolicyProcessHost().run(runner, policy)

    assert report.exit_kind == "invalid_action"
    assert report.answer.termination_reason == StopReason.POLICY_INVALID_ACTION
    event_types = [item.event_type for item in runner.transcript.events]
    assert EventType.ACTION_REJECTED in event_types
    assert EventType.TOOL_REQUEST not in event_types
    assert "select secret" not in json.dumps(
        [item.model_dump(mode="json") for item in runner.transcript.events]
    )


@pytest.mark.parametrize(
    ("failure_field", "expected_reason", "expected_event"),
    [
        ("crash_after_actions", StopReason.POLICY_CRASH, EventType.POLICY_EXCEPTION),
        ("hang_after_actions", StopReason.POLICY_TIMEOUT, EventType.POLICY_TIMEOUT),
    ],
)
def test_policy_crash_or_timeout_preserves_transcript_and_freezes_failure(
    cases_root: Path,
    tmp_path: Path,
    failure_field: str,
    expected_reason: StopReason,
    expected_event: EventType,
):
    transcript_path = tmp_path / f"{failure_field}.jsonl"
    runner = _runner(cases_root, transcript_path)
    spec = {
        "actions": (RequestStopAction(summary="action is not reached").model_dump(mode="json"),),
        failure_field: 0,
    }

    report = ScriptedPolicyProcessHost().run(
        runner,
        ScriptedPolicySpec(**spec),
        timeout_seconds=0.3,
    )

    assert report.answer.termination_reason == expected_reason
    assert expected_event in {item.event_type for item in runner.transcript.events}
    assert transcript_path.read_text()
    assert RecordedReplay.from_path(transcript_path).valid
