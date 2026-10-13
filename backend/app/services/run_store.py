"""Read recorded investigation runs without reading evaluation or Ground Truth data."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from backend.app.schemas import ReplayStatus, RunStatus, RunSummary
from rca_agent.models import AnswerArtifact
from rca_agent.runner import RecordedReplay, ReplayValidationError
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


@dataclass(frozen=True, slots=True)
class StoredRun:
    manifest: RunManifest
    events: tuple[TranscriptEvent, ...]
    answer: AnswerArtifact | None
    replay_status: ReplayStatus
    replay_error: str | None


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
        if transcript_path.is_file():
            events = load_jsonl_events(transcript_path)
            if manifest.status == RunStatus.SUCCEEDED:
                try:
                    RecordedReplay.verify(events)
                    replay_status = ReplayStatus.VALID
                except ReplayValidationError as exc:
                    replay_status = ReplayStatus.INVALID
                    replay_error = type(exc).__name__
        answer = AnswerArtifact.model_validate_json(answer_path.read_text("utf-8")) \
            if answer_path.is_file() else None
        if answer is not None and not answer.verify_digest():
            replay_status = ReplayStatus.INVALID
            replay_error = "AnswerArtifactDigestInvalid"
        return StoredRun(manifest, events, answer, replay_status, replay_error)

    @staticmethod
    def _summary(stored: StoredRun) -> RunSummary:
        tool_calls = sum(1 for event in stored.events if event.event_type.value == "tool_response")
        answer = stored.answer
        return RunSummary(
            run_id=stored.manifest.run_id,
            case_id=stored.manifest.case_id,
            status=stored.manifest.status,
            replay_status=stored.replay_status,
            started_at=stored.manifest.started_at,
            finished_at=stored.manifest.finished_at,
            duration_ms=_duration_ms(stored.manifest.started_at, stored.manifest.finished_at),
            query_count=tool_calls,
            final_status=answer.authoritative_decision.status.value if answer else None,
            profile_id=stored.manifest.profile_id,
            model_label=stored.manifest.model_label,
            error_code=stored.manifest.error_code,
        )


def _duration_ms(started_at: str, finished_at: str | None) -> int | None:
    if finished_at is None:
        return None
    from datetime import datetime

    try:
        return max(0, int((datetime.fromisoformat(finished_at) - datetime.fromisoformat(started_at)).total_seconds() * 1000))
    except ValueError:
        return None
