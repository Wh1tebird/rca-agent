from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter

from evaluation.ground_truth import FlatGroundTruthRepository
from evaluation.models import EvaluationArtifact, EvaluationOutcome
from evaluation.report import build_evaluation_report
from rca_agent.models import EpisodeStatus
from rca_agent.runner import EpisodeRunner, EpisodeStateError, RecordedReplay
from rca_agent.tools.fingerprint import sha256_digest


class EvaluatorHost:
    """Host-only post-freeze loader. Evaluation data never enters the inference package."""

    def __init__(
        self, evaluator_data_root: Path, evaluator_version: str = "phase2-loader-v1"
    ) -> None:
        self.__evaluator_data_root = evaluator_data_root.resolve()
        self.__evaluator_version = evaluator_version

    def evaluate(
        self,
        runner: EpisodeRunner,
        case_selector: str,
        *,
        artifact_path: Path | None = None,
    ) -> EvaluationArtifact:
        if runner.status != EpisodeStatus.ANSWER_FROZEN or runner.answer is None:
            raise EpisodeStateError("evaluation data cannot be loaded before answer freeze")
        if not runner.answer.verify_digest():
            raise EpisodeStateError("frozen answer artifact failed digest validation")
        record = FlatGroundTruthRepository(self.__evaluator_data_root).load(case_selector)
        replay = RecordedReplay.verify(runner.transcript.events)
        outcome = EvaluationOutcome.INCONCLUSIVE
        outcome_detail = "answer_not_verified"
        if runner.answer.authoritative_decision.status.value in {
            "supported_hypothesis", "discovered_verified", "confirmed_root_cause",
        }:
            matched = any(
                proposal.candidate_id == record.expected_fault_id
                for proposal in runner.answer.policy_proposals
            )
            outcome = EvaluationOutcome.CORRECT if matched else EvaluationOutcome.INCORRECT
            outcome_detail = "candidate_id_match" if matched else "candidate_id_mismatch"
        report = build_evaluation_report(
            runner,
            replay,
            evaluator_version=self.__evaluator_version,
            evaluator_record_digest=sha256_digest(record.model_dump(mode="json")),
            outcome=outcome,
            outcome_detail=outcome_detail,
        )
        runner.begin_evaluation()
        evaluated_at = datetime.now(UTC)
        fields = {
            "episode_id": runner.episode_id,
            "answer_artifact_digest": runner.answer.artifact_digest,
            "evaluator_version": self.__evaluator_version,
            "evaluated_at": evaluated_at,
            "evaluation_status": "post_freeze_record_loaded",
            "evaluator_record_digest": sha256_digest(record.model_dump(mode="json")),
            "report": report,
        }
        artifact = EvaluationArtifact(
            **fields,
            artifact_digest=sha256_digest(_jsonable(fields)),
        )
        if artifact_path is not None:
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            with artifact_path.open("x", encoding="utf-8") as stream:
                stream.write(artifact.model_dump_json(indent=2))
                stream.write("\n")
        runner.complete_evaluation(artifact.artifact_digest)
        return artifact


def _jsonable(value: dict[str, Any]) -> dict[str, Any]:
    return TypeAdapter(dict[str, Any]).dump_python(value, mode="json")
