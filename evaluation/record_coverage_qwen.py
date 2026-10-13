"""Record small GT-free Qwen Coverage shadow runs without starting the web UI.

Load provider settings from the process environment. Each Case is run to a frozen
Answer and independently replayed before its answer file is saved. Failures are
recorded by safe error type and do not stop the remaining Cases. Run the separate
mechanism_coverage_shadow evaluator only after this command finishes.
"""

from __future__ import annotations

import argparse
import json
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path

from backend.app.config import Settings
from backend.app.schemas import RunStatus
from backend.app.services.run_executor import (
    _run_qwen_investigation_profile,
    _write_manifest,
)
from backend.app.services.run_store import RunManifest
from rca_agent.runner import RecordedReplay


def record_cases(case_ids: tuple[str, ...], runs_root: Path) -> list[dict[str, object]]:
    settings = Settings.from_environment()
    if not all(re.fullmatch(r"t\d{3}", item) for item in case_ids):
        raise ValueError("Case IDs must match tNNN")
    runs_root.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, object]] = []
    for case_id in case_ids:
        run_id = f"run-coverage-{case_id}-{uuid.uuid4().hex[:12]}"
        run_dir = runs_root / run_id
        run_dir.mkdir()
        manifest = RunManifest(
            schema_version=1,
            run_id=run_id,
            case_id=case_id,
            status=RunStatus.RUNNING,
            started_at=datetime.now(UTC).isoformat(),
            profile_id="qwen-coverage-shadow",
            model_label=settings.qwen_model,
        )
        _write_manifest(run_dir, manifest)
        row: dict[str, object] = {"case_id": case_id, "run_id": run_id}
        try:
            answer = _run_qwen_investigation_profile(
                settings,
                case_id,
                run_dir / "transcript.jsonl",
                mechanism_coverage_mode="shadow",
            )
            replay = RecordedReplay.from_path(run_dir / "transcript.jsonl")
            if not replay.valid or not answer.verify_digest():
                raise RuntimeError("frozen Answer or Recorded Replay is invalid")
            (run_dir / "answer.json").write_text(
                answer.model_dump_json(indent=2) + "\n", encoding="utf-8"
            )
            manifest = manifest.model_copy(update={
                "status": RunStatus.SUCCEEDED,
                "finished_at": datetime.now(UTC).isoformat(),
            })
            row.update({
                "status": "succeeded",
                "replay_valid": True,
                "final_status": answer.authoritative_decision.status.value,
            })
        except Exception as exc:  # noqa: BLE001 - record failure and continue the batch
            manifest = manifest.model_copy(update={
                "status": RunStatus.FAILED,
                "finished_at": datetime.now(UTC).isoformat(),
                "error_code": type(exc).__name__,
            })
            row.update({"status": "failed", "error_code": type(exc).__name__})
        _write_manifest(run_dir, manifest)
        results.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_ids", nargs="+", help="Small, preselected Case set, e.g. t030 t007")
    parser.add_argument(
        "--runs-root", type=Path,
        default=Path("analysis/mechanism_coverage_qwen_20260924/runs"),
    )
    args = parser.parse_args()
    record_cases(tuple(args.case_ids), args.runs_root)


if __name__ == "__main__":
    main()
