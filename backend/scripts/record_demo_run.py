"""Record a small GT-free RCA100 investigation for local console demonstration.

This is not a benchmark and does not invoke an LLM. It exercises the real
Environment, EpisodeRunner, Transcript, AnswerArtifact and Recorded Replay.
"""
from __future__ import annotations

import argparse
import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

from rca_agent.environment import Rca100Environment
from rca_agent.models import (
    CallToolAction,
    GetServiceTraceSummaryRequest,
    RequestStopAction,
    ToolName,
)
from rca_agent.runner import EpisodeRunner, RecordedReplay


def record(case_id: str, cases_root: Path, runs_root: Path) -> str:
    run_id = f"run-demo-{uuid.uuid4().hex[:12]}"
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    transcript_path = run_dir / "transcript.jsonl"
    started_at = datetime.now(UTC)
    runner = EpisodeRunner(
        Rca100Environment.from_case(cases_root, case_id),
        transcript_path=transcript_path,
    )
    bootstrap = runner.start()
    runner.handle(CallToolAction(
        tool_name=ToolName.GET_SERVICE_TRACE_SUMMARY,
        request=GetServiceTraceSummaryRequest(
            entity_ref=bootstrap.alert_entity.entity_ref,
            window=bootstrap.observation_horizon,
            min_duration_ms=100,
            max_traces=4,
        ),
    ))
    answer = runner.handle(RequestStopAction(
        summary="Recorded a bounded trace summary for the investigation console demonstration.",
    ))
    assert answer is not None
    replay = RecordedReplay.from_path(transcript_path)
    if not replay.valid:
        raise RuntimeError("recorded demo run did not pass replay")
    finished_at = datetime.now(UTC)
    (run_dir / "answer.json").write_text(answer.model_dump_json(indent=2) + "\n", encoding="utf-8")
    (run_dir / "manifest.json").write_text(json.dumps({
        "schema_version": 1,
        "run_id": run_id,
        "case_id": case_id,
        "status": "succeeded",
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "profile_id": "scripted-trace-demo",
        "model_label": "No model — bounded runner fixture",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return run_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", default="t021")
    parser.add_argument(
        "--cases-root",
        type=Path,
        default=Path.home() / "study" / "AgenticOpsEval" / "RCA100" / "cases",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "var" / "runs",
    )
    args = parser.parse_args()
    print(record(args.case_id, args.cases_root, args.runs_root))


if __name__ == "__main__":
    main()
