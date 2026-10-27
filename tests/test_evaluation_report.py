from __future__ import annotations

from pathlib import Path

from evaluation.models import EvaluationOutcome, FailureCategory
from evaluation.report import build_evaluation_report, failure_taxonomy
from rca_agent.environment.rca100 import Rca100Environment
from rca_agent.models import EpisodeStatus, RequestStopAction, StopReason
from rca_agent.runner import EpisodeRunner, RecordedReplay, ReplayReport
from rca_agent.runner.transcript import EventType, Transcript


def test_evaluation_report_keeps_dimensions_independent(cases_root: Path) -> None:
    runner = EpisodeRunner(Rca100Environment.from_case(cases_root, "t025"))
    runner.start()
    answer = runner.handle(RequestStopAction(summary="evaluation projection"))
    replay = RecordedReplay.verify(runner.transcript.events)
    report = build_evaluation_report(
        runner,
        replay,
        evaluator_version="test-evaluator",
        evaluator_record_digest="a" * 64,
    )

    assert answer.artifact_digest == report.answer_artifact_digest
    assert report.outcome.result is EvaluationOutcome.NOT_EVALUABLE
    assert report.evidence.result == "not_evaluable"
    assert report.efficiency.query_count == 0
    assert report.contract.compliant
    assert report.replay.valid
    assert report.verify_digest()
    assert report.provenance.phase == "post_freeze"


def test_failure_taxonomy_is_diagnostic_and_does_not_change_answer(cases_root: Path) -> None:
    runner = EpisodeRunner(Rca100Environment.from_case(cases_root, "t025"))
    runner.start()
    answer = runner.terminate_policy_failure(
        reason=StopReason.POLICY_TIMEOUT,
        error_type="deterministic_timeout",
    )
    report = build_evaluation_report(
        runner,
        RecordedReplay.verify(runner.transcript.events),
        evaluator_version="test-evaluator",
        evaluator_record_digest="b" * 64,
    )

    assert answer.termination_reason.value == "policy_timeout"
    assert any(item.failure_kind == "policy_timeout" for item in report.failure_taxonomy)
    assert report.outcome.result is EvaluationOutcome.NOT_EVALUABLE


def test_failure_taxonomy_covers_layered_categories_without_ground_truth() -> None:
    transcript = Transcript("ep-" + "c" * 32)
    transcript.append(EventType.TOOL_EXCEPTION, {"error_type": "tool_failure"})
    transcript.append(EventType.TOOL_EXCEPTION, {"error_type": "worker_crash"})
    transcript.append(EventType.DISCOVERY_PAYLOAD_REJECTED, {"error_type": "ValidationError"})
    transcript.append(EventType.ACTION_REJECTED, {"error_type": "invalid_action"})
    transcript.append(EventType.CANDIDATE_EVIDENCE_UPDATED, {
        "failure_category": "grounding",
        "failure_kind": "cross_candidate_evidence_leak",
        "phase": "admission",
    })
    transcript.append(EventType.STOP_REQUESTED, {"reason": "budget_exhausted"})
    transcript.append(EventType.ACTION_REJECTED, {
        "failure_category": "state",
        "failure_kind": "state_divergence",
        "phase": "projection",
    })
    replay = ReplayReport(
        valid=False,
        episode_id=transcript.episode_id,
        event_count=len(transcript.events),
        final_status=EpisodeStatus.ACTIVE,
        head_digest=transcript.head_digest,
        tool_calls=0,
    )
    categories = {item.category for item in failure_taxonomy(transcript.events, replay)}
    assert categories >= {
        FailureCategory.EXECUTION,
        FailureCategory.PROTOCOL,
        FailureCategory.STATE,
        FailureCategory.INVESTIGATION,
        FailureCategory.GROUNDING,
        FailureCategory.CONTRACT,
        FailureCategory.REPLAY,
        FailureCategory.ADAPTER_ORCHESTRATION,
    }
