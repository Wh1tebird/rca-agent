from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation import EvaluatorHost
from rca_agent.environment.rca100 import Rca100Environment
from rca_agent.models import EpisodeStatus, RequestStopAction
from rca_agent.runner import EpisodeRunner, EpisodeStateError, RecordedReplay


@pytest.fixture
def synthetic_evaluator_root(tmp_path: Path) -> Path:
    root = tmp_path / "evaluator-private"
    root.mkdir()
    raw = {
        "outcome": {"expected_fault_id": "synthetic_fault"},
        "reasoning": {
            "steps": [
                {
                    "step": 1,
                    "step_type": "cause",
                    "target": "synthetic-service",
                    "fault_id": "synthetic_fault",
                    "observability": [],
                }
            ]
        },
    }
    payload = {
        "root_cause_entities": ["synthetic-service"],
        "root_cause_types": ["synthetic_fault"],
        "raw_ground_truth": json.dumps(raw),
    }
    (root / "t999.gt.json").write_text(json.dumps(payload))
    return root


def test_evaluator_cannot_load_before_freeze(cases_root: Path, synthetic_evaluator_root: Path):
    runner = EpisodeRunner(Rca100Environment.from_case(cases_root, "t025"))
    runner.start()
    evaluator = EvaluatorHost(synthetic_evaluator_root)

    with pytest.raises(EpisodeStateError, match="before answer freeze"):
        evaluator.evaluate(runner, "t999")

    assert runner.status == EpisodeStatus.ACTIVE
    assert not any(
        path.name.endswith("evaluation.json") for path in synthetic_evaluator_root.iterdir()
    )


def test_legal_lifecycle_reaches_evaluated_with_separate_artifact(
    cases_root: Path,
    synthetic_evaluator_root: Path,
    tmp_path: Path,
):
    transcript_path = tmp_path / "episode.jsonl"
    evaluation_path = tmp_path / "evaluation.json"
    runner = EpisodeRunner(
        Rca100Environment.from_case(cases_root, "t025"),
        transcript_path=transcript_path,
    )
    runner.start()
    answer = runner.handle(RequestStopAction(summary="freeze before evaluator access"))
    answer_before = answer.model_dump_json()

    artifact = EvaluatorHost(synthetic_evaluator_root).evaluate(
        runner,
        "t999",
        artifact_path=evaluation_path,
    )

    assert runner.status == EpisodeStatus.EVALUATED
    assert artifact.verify_digest()
    assert artifact.answer_artifact_digest == answer.artifact_digest
    assert runner.answer.model_dump_json() == answer_before
    assert evaluation_path.is_file()
    assert RecordedReplay.from_path(transcript_path).final_status == EpisodeStatus.EVALUATED
    transcript = transcript_path.read_text()
    answer_payload = answer.model_dump_json()
    for private_value in (
        str(synthetic_evaluator_root),
        "t999",
        "synthetic_fault",
        "synthetic-service",
    ):
        assert private_value not in transcript
        assert private_value not in answer_payload

    with pytest.raises(EpisodeStateError):
        EvaluatorHost(synthetic_evaluator_root).evaluate(runner, "t999")
