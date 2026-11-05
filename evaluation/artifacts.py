"""Persistent benchmark artifacts and offline post-freeze evaluation.

This module is evaluator-side only. It never constructs an Environment, calls a
provider, or imports the inference package's data loaders.
"""
from __future__ import annotations

import json
import shutil
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from evaluation.models import (
    ContractEvaluation,
    EfficiencyEvaluation,
    EvaluationOutcome,
    EvaluationProvenance,
    EvaluationReport,
    EvidenceEvaluation,
    OutcomeEvaluation,
    ReplayEvaluation,
)
from rca_agent.runner import RecordedReplay
from rca_agent.runner.transcript import load_jsonl_events
from rca_agent.tools.fingerprint import sha256_digest


class UsageCall(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: str
    model: str
    call_id: str
    phase: str
    started_at: datetime | None = None
    finished_at: datetime | None = None
    duration_seconds: float | str = "unavailable"
    input_tokens: int | str = "unavailable"
    output_tokens: int | str = "unavailable"
    total_tokens: int | str = "unavailable"
    usage_source: str
    unavailable_reason: str | None = None


class FrozenAnswerArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "benchmark-frozen-answer-v1"
    episode_id: str
    answer_digest: str | None = None
    answer_status: str
    answer_payload: dict[str, Any]
    final_status: str
    winner: dict[str, Any] | None = None
    stop_reason: str | None = None
    candidate_frontier_digest: str | None = None
    workspace_digest: str | None = None
    transcript_head_digest: str | None = None
    answer_digest_status: str


class TranscriptArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "benchmark-transcript-v1"
    episode_id: str | None = None
    transcript_path: str | None = None
    event_count: int = Field(ge=0)
    transcript_head_digest: str | None = None
    replay_status: str
    complete: bool
    availability_reason: str | None = None


class EvaluatorInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "benchmark-evaluator-input-v1"
    episode_id: str | None = None
    frozen_answer: FrozenAnswerArtifact
    observation_refs: tuple[str, ...] = ()
    candidate_admission: tuple[dict[str, Any], ...] = ()
    replay_metadata: dict[str, Any]
    usage_calls: tuple[UsageCall, ...] = ()
    ground_truth_available: bool = False


def _safe_answer_projection(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": result.get("status", result.get("run_status")),
        "supported_hypothesis_ids": result.get("supported_hypothesis_ids", []),
        "policy_proposals": result.get("policy_proposals", []),
        "winner": result.get("winner"),
        "observation_refs": result.get("observation_refs", []),
        "missing_evidence": result.get("missing_evidence", []),
    }


def _answer_record(result: dict[str, Any], answer_payload: dict[str, Any] | None) -> FrozenAnswerArtifact:
    payload = answer_payload or _safe_answer_projection(result)
    expected = result.get("answer_digest")
    digest_payload = dict(payload)
    digest_payload.pop("artifact_digest", None)
    status = "verified" if answer_payload is not None and expected and sha256_digest(digest_payload) == expected else (
        "not_evaluable" if expected else "unavailable"
    )
    return FrozenAnswerArtifact(
        episode_id=result.get("episode_id"),
        answer_digest=expected,
        answer_status="frozen" if result.get("answer_digest") else "unavailable",
        answer_payload=payload,
        final_status=str(result.get("status", result.get("run_status", "unavailable"))),
        winner=result.get("winner"),
        stop_reason=result.get("stop_reason"),
        candidate_frontier_digest=result.get("candidate_frontier_digest"),
        workspace_digest=result.get("workspace_digest"),
        transcript_head_digest=result.get("transcript_head_digest"),
        answer_digest_status=status,
    )


def persist_benchmark_result(
    result: dict[str, Any], output_dir: Path, *,
    transcript_path: Path | None = None,
    answer_payload: dict[str, Any] | None = None,
    usage_calls: Iterable[UsageCall] = (),
) -> Path:
    """Persist a result without changing episode or Runtime state."""
    if not result.get("answer_digest") and answer_payload is None:
        raise ValueError("artifact persistence requires an Answer Freeze")
    output_dir.mkdir(parents=True, exist_ok=True)
    answer = _answer_record(result, answer_payload)
    answer_path = output_dir / "frozen_answer.json"
    answer_path.write_text(answer.model_dump_json(indent=2) + "\n", encoding="utf-8")
    calls = tuple(usage_calls)
    usage_path = output_dir / "usage.json"
    usage_path.write_text(json.dumps({"schema_version": "benchmark-usage-v1", "calls": [
        call.model_dump(mode="json") for call in calls
    ]}, indent=2) + "\n", encoding="utf-8")

    transcript_meta = TranscriptArtifact(
        episode_id=result.get("episode_id"), event_count=0,
        transcript_head_digest=result.get("transcript_head_digest"),
        replay_status=result.get("replay_status", "unavailable"), complete=False,
        availability_reason="complete transcript was not included in source result",
    )
    if transcript_path is not None:
        target = output_dir / "transcript.jsonl"
        shutil.copyfile(transcript_path, target)
        events = load_jsonl_events(target)
        replay = RecordedReplay.verify(events)
        transcript_meta = TranscriptArtifact(
            episode_id=events[0].episode_id, transcript_path="transcript.jsonl",
            event_count=len(events), transcript_head_digest=replay.head_digest,
            replay_status="valid" if replay.valid else "invalid", complete=True,
        )
    (output_dir / "transcript_metadata.json").write_text(
        transcript_meta.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    replay_metadata = {
        "status": transcript_meta.replay_status,
        "head_digest": transcript_meta.transcript_head_digest,
        "answer_digest": answer.answer_digest,
    }
    evaluator_input = EvaluatorInput(
        episode_id=answer.episode_id,
        frozen_answer=answer,
        observation_refs=tuple(answer.answer_payload.get("observation_refs", ())),
        candidate_admission=tuple(result.get("registered_candidates", ())),
        replay_metadata=replay_metadata,
        usage_calls=calls,
    )
    (output_dir / "evaluator_input.json").write_text(
        evaluator_input.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": "benchmark-artifact-manifest-v1",
        "case_id": result.get("case_id"),
        "profile": result.get("acquisition_mode", result.get("profile")),
        "episode_id": answer.episode_id,
        "files": ["frozen_answer.json", "transcript_metadata.json", "usage.json", "evaluator_input.json"],
        "answer_digest": answer.answer_digest,
        "transcript_head_digest": transcript_meta.transcript_head_digest,
        "created_at": datetime.now(UTC).isoformat(),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return output_dir


class OfflineArtifactEvaluator:
    """Evaluate persisted facts without a live Runner or provider."""

    def load_report(self, artifact_dir: Path) -> EvaluationReport:
        evaluator_input = EvaluatorInput.model_validate_json(
            (artifact_dir / "evaluator_input.json").read_text(encoding="utf-8")
        )
        answer = evaluator_input.frozen_answer
        if answer.answer_digest_status != "verified":
            outcome = EvaluationOutcome.NOT_EVALUABLE
            detail = "answer_digest_not_verifiable_from_persisted_projection"
        elif evaluator_input.replay_metadata.get("status") != "valid":
            outcome = EvaluationOutcome.NOT_EVALUABLE
            detail = "replay_invalid_or_unavailable"
        else:
            outcome = EvaluationOutcome.NOT_EVALUABLE
            detail = "ground_truth_not_loaded"
        replay = ReplayEvaluation(
            valid=evaluator_input.replay_metadata.get("status") == "valid",
            head_digest=evaluator_input.replay_metadata.get("head_digest") or "0" * 64,
            answer_digest=answer.answer_digest,
            candidate_frontier_digest=answer.candidate_frontier_digest,
            workspace_digest=answer.workspace_digest,
            stop_reason=answer.stop_reason,
        )
        fields = {
            "episode_id": answer.episode_id,
            "answer_artifact_digest": answer.answer_digest or "0" * 64,
            "outcome": OutcomeEvaluation(result=outcome, detail_code=detail),
            "evidence": EvidenceEvaluation(
                result="supported" if evaluator_input.observation_refs else "not_evaluable",
                valid_observation_count=len(evaluator_input.observation_refs),
                unissued_observation_count=0, explicit_admission_count=len(evaluator_input.observation_refs),
                cross_candidate_leak_count=0, ineligible_positive_count=0,
                same_episode=True,
            ),
            "efficiency": EfficiencyEvaluation(
                query_count=0, budget_consumed=0, invalid_query_count=0,
                rejected_or_failed_query_count=0, duplicate_query_count=0, stop_quality="unknown",
            ),
            "contract": ContractEvaluation(compliant=replay.valid, violation_count=0 if replay.valid else 1),
            "replay": replay,
            "failure_taxonomy": (),
            "provenance": EvaluationProvenance(
                evaluator_version="offline-artifact-v1", record_digest=sha256_digest(evaluator_input.model_dump(mode="json")),
            ),
        }
        digest_fields = {
            key: (value.model_dump(mode="json") if isinstance(value, BaseModel) else value)
            for key, value in fields.items()
        }
        return EvaluationReport(**fields, report_digest=sha256_digest(digest_fields))
