"""Safe, offline recovery inspection for interrupted Episodes.

Recovery never reconstructs facts from a partial record and never calls a model,
provider, Environment, or worker.  A live EpisodeRunner can be resumed only
after its current transcript passes this inspection; raw historical records are
returned as a plan, not silently repaired into a new runtime.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from rca_agent.models import EpisodeStatus, StopReason
from rca_agent.runner.replay import RecordedReplay, ReplayReport, ReplayValidationError
from rca_agent.runner.transcript import (
    GENESIS_DIGEST,
    EventRecord,
    EventType,
    TranscriptEvent,
)
from rca_agent.tools.fingerprint import sha256_digest


class RecoveryState(StrEnum):
    RESUMABLE = "resumable"
    RETRYABLE = "retryable"
    TERMINAL = "terminal"
    FROZEN = "frozen"
    CORRUPTED = "corrupted"


class PendingOperation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    operation_id: str = Field(min_length=1, max_length=160)
    kind: Literal["query", "adapter_call", "handoff"]
    disposition: Literal["retryable", "terminate"]
    source_event_id: str = Field(pattern=r"^[a-f0-9]{64}$")


class RecoverySnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    state: RecoveryState
    episode_id: str | None = Field(default=None, pattern=r"^ep-[a-f0-9]{32}$")
    event_count: int = Field(default=0, ge=0)
    next_sequence: int = Field(default=1, ge=1)
    head_digest: str = Field(default=GENESIS_DIGEST, pattern=r"^[a-f0-9]{64}$")
    budget_remaining: int | None = Field(default=None, ge=0)
    workspace_digest: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    candidate_frontier_digest: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    answer_digest: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    stop_reason: StopReason | None = None
    pending_operations: tuple[PendingOperation, ...] = ()
    replay_valid: bool = False
    failure_category: str | None = None
    failure_kind: str | None = None
    core_binding_digest: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")

    @property
    def can_continue(self) -> bool:
        return self.state in {RecoveryState.RESUMABLE, RecoveryState.RETRYABLE}


class RecoveryValidationError(ValueError):
    pass


class EpisodeRecovery:
    """Inspect a transcript or a fully populated EventRecord stream offline."""

    @classmethod
    def inspect(
        cls,
        events: Sequence[TranscriptEvent | EventRecord] | Iterable[TranscriptEvent | EventRecord],
    ) -> RecoverySnapshot:
        raw_events = tuple(events)
        if not raw_events:
            return RecoverySnapshot(
                state=RecoveryState.CORRUPTED,
                failure_category="REPLAY",
                failure_kind="empty_transcript",
            )
        try:
            normalized = tuple(cls._normalize_event(event) for event in raw_events)
        except (TypeError, ValueError):
            episode_id = getattr(raw_events[0], "episode_id", None)
            return RecoverySnapshot(
                state=RecoveryState.CORRUPTED,
                episode_id=episode_id if isinstance(episode_id, str) else None,
                event_count=len(raw_events),
                failure_category="REPLAY",
                failure_kind="event_record_missing_integrity_fields",
            )
        try:
            cls._validate_chain(normalized)
        except RecoveryValidationError as exc:
            return cls._corrupted(normalized, str(exc))

        pending = cls._pending_operations(normalized)
        replay: ReplayReport | None = None
        replay_error: str | None = None
        try:
            replay = RecordedReplay.verify(normalized)
        except ReplayValidationError as exc:
            replay_error = _safe_replay_error(str(exc))

        first = normalized[0]
        head = normalized[-1]
        budget = replay.budget_remaining if replay is not None else _budget(normalized)
        answer = next(
            (event.payload for event in reversed(normalized)
             if event.event_type == EventType.ANSWER_FROZEN),
            None,
        )
        answer_digest = answer.get("artifact_digest") if isinstance(answer, dict) else None
        workspace_digest = (
            answer.get("investigation_workspace", {}).get("snapshot_digest")
            if isinstance(answer, dict) and isinstance(answer.get("investigation_workspace"), dict)
            else None
        )
        state = _state_for(replay, pending, normalized)
        if replay is None and not pending:
            state = RecoveryState.CORRUPTED
        stop_reason = replay.stop_reason if replay is not None else _stop_reason(normalized)
        failure_kind = replay_error
        if state == RecoveryState.RETRYABLE and failure_kind is None:
            failure_kind = "pending_operation"
        return RecoverySnapshot(
            state=state,
            episode_id=first.episode_id,
            event_count=len(normalized),
            next_sequence=head.sequence + 1,
            head_digest=head.digest,
            budget_remaining=budget,
            workspace_digest=workspace_digest or (replay.workspace_digest if replay else None),
            candidate_frontier_digest=(replay.candidate_frontier_digest if replay else None),
            answer_digest=answer_digest or (replay.answer_digest if replay else None),
            stop_reason=stop_reason,
            pending_operations=pending,
            replay_valid=replay is not None and replay.valid,
            failure_category=None if replay is not None else "REPLAY",
            failure_kind=failure_kind,
            core_binding_digest=_core_binding_digest(first),
        )

    @classmethod
    def attach(cls, runner: Any) -> RecoverySnapshot:
        """Validate a live Runner before allowing its caller to continue it."""

        snapshot = cls.inspect(runner.transcript.events)
        if snapshot.episode_id != runner.episode_id:
            raise RecoveryValidationError("episode_id_mismatch")
        if snapshot.state != RecoveryState.RESUMABLE:
            raise RecoveryValidationError(f"episode_is_{snapshot.state.value}")
        return snapshot

    @staticmethod
    def _normalize_event(event: TranscriptEvent | EventRecord) -> TranscriptEvent:
        if isinstance(event, TranscriptEvent):
            return event
        if isinstance(event, EventRecord):
            return event.to_transcript_event()
        raise TypeError("recovery accepts TranscriptEvent or EventRecord")

    @staticmethod
    def _validate_chain(events: tuple[TranscriptEvent, ...]) -> None:
        episode_id = events[0].episode_id
        previous = GENESIS_DIGEST
        for expected, event in enumerate(events, start=1):
            if event.episode_id != episode_id:
                raise RecoveryValidationError("episode_id_mismatch")
            if event.sequence != expected:
                raise RecoveryValidationError("duplicate_or_noncontiguous_event")
            if event.previous_digest != previous:
                raise RecoveryValidationError("digest_chain_mismatch")
            if not event.verify_digest():
                raise RecoveryValidationError("event_digest_mismatch")
            previous = event.digest

    @staticmethod
    def _pending_operations(events: tuple[TranscriptEvent, ...]) -> tuple[PendingOperation, ...]:
        pending: list[PendingOperation] = []
        requests = {
            event.payload.get("query_ref"): event
            for event in events
            if event.event_type == EventType.TOOL_REQUEST and event.payload.get("query_ref")
        }
        responses = {
            event.payload.get("query_ref")
            for event in events
            if event.event_type == EventType.TOOL_RESPONSE
        }
        for query_ref, event in requests.items():
            if query_ref not in responses:
                pending.append(PendingOperation(
                    operation_id=f"query-{query_ref}",
                    kind="query",
                    disposition="retryable",
                    source_event_id=event.event_id,
                ))
        started = {
            EventType.DISCOVERY_STARTED: ("handoff", EventType.DISCOVERY_COMPLETED),
            EventType.MECHANISM_FORMALIZATION_STARTED: (
                "adapter_call", EventType.MECHANISM_FORMALIZATION_COMPLETED,
            ),
        }
        for start_type, (kind, complete_type) in started.items():
            starts = [event for event in events if event.event_type == start_type]
            completes = [event for event in events if event.event_type == complete_type]
            if len(starts) > len(completes):
                event = starts[-1]
                pending.append(PendingOperation(
                    operation_id=f"{kind}-{event.sequence}",
                    kind=kind, disposition="retryable", source_event_id=event.event_id,
                ))
        for event in events:
            payload = event.payload
            if payload.get("operation_status") in {"pending", "started", "requested"}:
                kind = payload.get("operation_kind", "adapter_call")
                if kind in {"query", "adapter_call", "handoff"}:
                    pending.append(PendingOperation(
                        operation_id=str(payload.get("operation_id", f"operation-{event.sequence}")),
                        kind=kind,
                        disposition="retryable",
                        source_event_id=event.event_id,
                    ))
        unique = {item.operation_id: item for item in pending}
        return tuple(unique[key] for key in sorted(unique))

    @staticmethod
    def _corrupted(events: tuple[TranscriptEvent, ...], kind: str) -> RecoverySnapshot:
        return RecoverySnapshot(
            state=RecoveryState.CORRUPTED,
            episode_id=events[0].episode_id if events else None,
            event_count=len(events),
            next_sequence=(events[-1].sequence + 1 if events else 1),
            head_digest=(events[-1].digest if events else GENESIS_DIGEST),
            failure_category="REPLAY",
            failure_kind=kind,
        )


def _state_for(
    replay: ReplayReport | None,
    pending: tuple[PendingOperation, ...],
    events: tuple[TranscriptEvent, ...],
) -> RecoveryState:
    if replay is not None and replay.final_status in {EpisodeStatus.ANSWER_FROZEN, EpisodeStatus.EVALUATED}:
        return RecoveryState.FROZEN
    if pending:
        return RecoveryState.RETRYABLE
    if replay is not None and replay.final_status == EpisodeStatus.ACTIVE:
        return RecoveryState.RESUMABLE
    if any(event.event_type == EventType.ANSWER_FROZEN for event in events):
        return RecoveryState.FROZEN
    return RecoveryState.TERMINAL


def _budget(events: tuple[TranscriptEvent, ...]) -> int | None:
    values = [event.payload.get("budget_after") for event in events if event.event_type == EventType.QUERY_COST]
    return values[-1] if values and type(values[-1]) is int and values[-1] >= 0 else None


def _stop_reason(events: tuple[TranscriptEvent, ...]) -> StopReason | None:
    for event in reversed(events):
        value = event.payload.get("reason")
        try:
            return StopReason(value)
        except (TypeError, ValueError):
            continue
    return None


def _core_binding_digest(event: TranscriptEvent) -> str | None:
    payload = event.payload
    if not isinstance(payload, dict):
        return None
    return sha256_digest({
        "episode_id": event.episode_id,
        "engineering_contract_revision": payload.get("engineering_contract_revision"),
        "bound_handoff_version": payload.get("bound_handoff_version"),
        "inquiry_contract_version": payload.get("inquiry_contract_version"),
        "excluded_catalog_candidate_ids": payload.get("excluded_catalog_candidate_ids", ()),
    })


def _safe_replay_error(message: str) -> str:
    lowered = message.casefold()
    if "incomplete tool call" in lowered:
        return "pending_query"
    if "incomplete discovery" in lowered:
        return "pending_handoff"
    if "coverage shadow" in lowered:
        return "pending_answer_freeze"
    if "answer freeze" in lowered:
        return "freeze_validation_failure"
    if "digest" in lowered:
        return "digest_mismatch"
    if "sequence" in lowered or "episode id" in lowered:
        return "event_identity_mismatch"
    return "replay_validation_failure"
