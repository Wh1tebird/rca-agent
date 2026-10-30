"""Read recorded investigation runs without reading evaluation or Ground Truth data."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from backend.app.schemas import RecoveryStatus, ReplayStatus, RunStatus, RunSummary
from rca_agent.models import AnswerArtifact
from rca_agent.runner import (
    AuditRecord,
    EpisodeRecovery,
    RecordedReplay,
    RecoverySnapshot,
    RecoveryState,
    ReplayReport,
    ReplayValidationError,
    audit_transcript,
)
from rca_agent.runner.transcript import TranscriptEvent, load_jsonl_events

RUN_ID_RE = re.compile(r"^run-[a-z0-9][a-z0-9-]{2,63}$")


class RunManifest(BaseModel):
    """Host metadata only; the run record has no evaluator or answer-key field."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = Field(ge=1)
    run_id: str = Field(pattern=r"^run-[a-z0-9][a-z0-9-]{2,63}$")
    case_id: str = Field(pattern=r"^t\d{3}$")
    status: RunStatus
    started_at: str
    finished_at: str | None = None
    profile_id: str = Field(min_length=1, max_length=100)
    model_label: str | None = Field(default=None, max_length=200)
    error_code: str | None = Field(default=None, max_length=100)
    recovery_status: RecoveryStatus = RecoveryStatus.NOT_AVAILABLE
    recovery_failure: str | None = Field(default=None, max_length=120)


@dataclass(frozen=True, slots=True)
class StoredRun:
    manifest: RunManifest
    events: tuple[TranscriptEvent, ...]
    answer: AnswerArtifact | None
    replay_status: ReplayStatus
    replay_error: str | None
    replay_report: ReplayReport | None
    recovery: RecoverySnapshot
    audit: tuple[AuditRecord, ...]


class RunNotFound(KeyError):
    pass


class RunStore:
    def __init__(self, runs_root: Path) -> None:
        self._runs_root = runs_root

    def list_runs(self, case_id: str | None = None) -> tuple[RunSummary, ...]:
        if not self._runs_root.is_dir():
            return ()
        items = []
        for directory in self._runs_root.iterdir():
            if not directory.is_dir() or not RUN_ID_RE.fullmatch(directory.name):
                continue
            try:
                stored = self._load_directory(directory)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            if case_id is None or stored.manifest.case_id == case_id:
                items.append(self._summary(stored))
        return tuple(sorted(items, key=lambda item: item.started_at, reverse=True))

    def get_run(self, run_id: str) -> StoredRun:
        if not RUN_ID_RE.fullmatch(run_id):
            raise RunNotFound(run_id)
        directory = self._runs_root / run_id
        if not directory.is_dir():
            raise RunNotFound(run_id)
        try:
            return self._load_directory(directory)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise RunNotFound(run_id) from exc

    def get_summary(self, run_id: str) -> RunSummary:
        return self._summary(self.get_run(run_id))

    def _load_directory(self, directory: Path) -> StoredRun:
        manifest = RunManifest.model_validate_json((directory / "manifest.json").read_text("utf-8"))
        if manifest.run_id != directory.name:
            raise ValueError("manifest run ID mismatch")
        transcript_path = directory / "transcript.jsonl"
        answer_path = directory / "answer.json"
        events: tuple[TranscriptEvent, ...] = ()
        replay_status = ReplayStatus.NOT_AVAILABLE
        replay_error = None
        replay_report: ReplayReport | None = None
        transcript_error = False
        if transcript_path.is_file():
            try:
                events = load_jsonl_events(transcript_path)
            except (OSError, ValueError, json.JSONDecodeError):
                transcript_error = True
                replay_status = ReplayStatus.INVALID
                replay_error = "TranscriptInvalid"
            if not transcript_error:
                try:
                    replay_report = RecordedReplay.verify(events)
                    replay_status = ReplayStatus.VALID
                except ReplayValidationError as exc:
                    replay_status = ReplayStatus.INVALID
                    replay_error = type(exc).__name__
        answer = None
        if answer_path.is_file():
            try:
                answer = AnswerArtifact.model_validate_json(answer_path.read_text("utf-8"))
            except (OSError, ValueError, json.JSONDecodeError):
                replay_status = ReplayStatus.INVALID
                replay_error = "AnswerArtifactInvalid"
        if answer is not None and not answer.verify_digest():
            replay_status = ReplayStatus.INVALID
            replay_error = "AnswerArtifactDigestInvalid"
        recovery = (
            _corrupted_without_transcript((replay_error or "transcript_invalid").casefold())
            if transcript_error
            else (EpisodeRecovery.inspect(events) if events else _recovery_without_transcript(manifest))
        )
        if replay_status == ReplayStatus.INVALID and replay_error in {
            "AnswerArtifactInvalid", "AnswerArtifactDigestInvalid",
        }:
            recovery = recovery.model_copy(update={
                "state": RecoveryState.CORRUPTED,
                "failure_category": "REPLAY",
                "failure_kind": replay_error.casefold(),
            })
        if manifest.recovery_status != RecoveryStatus.NOT_AVAILABLE and not events:
            recovery = recovery.model_copy(update={
                "state": RecoveryState(manifest.recovery_status.value),
                "failure_kind": manifest.recovery_failure or recovery.failure_kind,
            })
        audit = audit_transcript(
            events,
            replay_result=replay_status.value if events else None,
        )
        return StoredRun(
            manifest=manifest,
            events=events,
            answer=answer,
            replay_status=replay_status,
            replay_error=replay_error,
            replay_report=replay_report,
            recovery=recovery,
            audit=audit,
        )

    @staticmethod
    def _summary(stored: StoredRun) -> RunSummary:
        tool_calls = sum(1 for event in stored.events if event.event_type.value == "tool_response")
        answer = stored.answer
        authoritative = (
            answer is not None
            and stored.replay_status == ReplayStatus.VALID
            and stored.recovery.state == RecoveryState.FROZEN
        )
        return RunSummary(
            schema_version=1,
            run_id=stored.manifest.run_id,
            case_id=stored.manifest.case_id,
            episode_id=(answer.episode_id if answer else stored.recovery.episode_id),
            status=stored.manifest.status,
            replay_status=stored.replay_status,
            recovery_status=_recovery_status(stored.recovery.state),
            started_at=stored.manifest.started_at,
            finished_at=stored.manifest.finished_at,
            duration_ms=_duration_ms(stored.manifest.started_at, stored.manifest.finished_at),
            query_count=tool_calls,
            final_status=(
                _public_decision_status(answer.authoritative_decision.status.value)
                if authoritative and answer is not None else None
            ),
            answer_digest=(answer.artifact_digest if answer else stored.recovery.answer_digest),
            stop_reason=(
                answer.termination_reason.value
                if authoritative and answer is not None
                else (stored.recovery.stop_reason.value if stored.recovery.stop_reason else None)
            ),
            profile_id=stored.manifest.profile_id,
            model_label=stored.manifest.model_label,
            error_code=stored.manifest.error_code,
        )


def _recovery_without_transcript(manifest: RunManifest) -> RecoverySnapshot:
    state = (
        RecoveryState.RETRYABLE
        if manifest.status in {RunStatus.QUEUED, RunStatus.RUNNING}
        else RecoveryState.TERMINAL
    )
    return RecoverySnapshot(
        state=state,
        failure_category="EXECUTION" if state == RecoveryState.RETRYABLE else None,
        failure_kind="pending_host_job" if state == RecoveryState.RETRYABLE else None,
    )


def _corrupted_without_transcript(kind: str) -> RecoverySnapshot:
    return RecoverySnapshot(
        state=RecoveryState.CORRUPTED,
        failure_category="REPLAY",
        failure_kind=kind,
    )


def _recovery_status(state: RecoveryState) -> RecoveryStatus:
    return RecoveryStatus(state.value)


def _public_decision_status(value: str) -> str:
    normalized = value.casefold()
    if normalized in {"confirmed_root_cause", "discovered_verified", "verified"}:
        return "VERIFIED"
    if normalized in {"supported", "supported_hypothesis", "multiple_supported"}:
        return "SUPPORTED"
    if normalized == "out_of_scope":
        return "OUT_OF_SCOPE"
    if normalized == "query_failed":
        return "QUERY_FAILED"
    if normalized == "inconclusive":
        return "INCONCLUSIVE"
    return normalized.upper()


def _duration_ms(started_at: str, finished_at: str | None) -> int | None:
    if finished_at is None:
        return None
    from datetime import datetime

    try:
        return max(0, int((datetime.fromisoformat(finished_at) - datetime.fromisoformat(started_at)).total_seconds() * 1000))
    except ValueError:
        return None
