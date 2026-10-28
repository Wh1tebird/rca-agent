"""Host-owned operational metrics and redacted audit projections.

The objects in this module are deliberately observational.  They do not feed
policy, evidence admission, Core decisions, stopping, or answer construction.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from datetime import datetime
from enum import StrEnum
from threading import Lock
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class OperationalMetricName(StrEnum):
    EPISODE_STARTED = "episode_started"
    EPISODE_COMPLETED = "episode_completed"
    EPISODE_FAILED = "episode_failed"
    WORKER_READY = "worker_ready"
    WORKER_STARTUP_TIMEOUT = "worker_startup_timeout"
    WORKER_REQUEST_TIMEOUT = "worker_request_timeout"
    WORKER_CRASH = "worker_crash"
    WORKER_EXCEPTION = "worker_exception"
    CANCELLATION = "cancellation"
    WORKER_TERMINATE = "worker_terminate"
    WORKER_KILL = "worker_kill"
    WORKER_REAP = "worker_reap"
    LATE_RESULT_REJECTED = "late_result_rejected"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    QUERY_REJECTED = "query_rejected"
    QUERY_FAILED = "query_failed"
    QUERY_COMPLETED = "query_completed"
    BUDGET_EXHAUSTED = "budget_exhausted"
    NO_VALID_QUERY_HANDOFF = "no_valid_query_handoff"
    REPLAY_FAILURE = "replay_failure"
    ANSWER_FREEZE_FAILURE = "answer_freeze_failure"
    RECOVERY_SUCCESS = "recovery_success"
    RECOVERY_FAILURE = "recovery_failure"


class OperationalMetric(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: OperationalMetricName
    value: float = Field(default=1, ge=0)
    episode_id: str | None = Field(default=None, pattern=r"^ep-[a-f0-9]{32}$")
    capability: str | None = Field(default=None, max_length=120)
    provider: str | None = Field(default=None, max_length=120)
    result_status: str | None = Field(default=None, max_length=80)
    failure_category: str | None = Field(default=None, max_length=80)
    query_ref: str | None = Field(default=None, pattern=r"^qry-[a-f0-9]{16}$")
    cost_units: int | None = Field(default=None, ge=0)
    latency_ms: float | None = Field(default=None, ge=0)
    recorded_at: datetime | None = None


class OperationalMetricsSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    counters: dict[str, int]
    samples: tuple[OperationalMetric, ...] = ()


class OperationalMetrics:
    """A small in-memory, thread-safe metrics sink owned by the host."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._counters: Counter[str] = Counter()
        self._samples: list[OperationalMetric] = []

    def record(
        self,
        name: OperationalMetricName,
        *,
        value: float = 1,
        episode_id: str | None = None,
        capability: str | None = None,
        provider: str | None = None,
        result_status: str | None = None,
        failure_category: str | None = None,
        query_ref: str | None = None,
        cost_units: int | None = None,
        latency_ms: float | None = None,
    ) -> None:
        point = OperationalMetric(
            name=name,
            value=value,
            episode_id=episode_id,
            capability=capability,
            provider=provider,
            result_status=result_status,
            failure_category=failure_category,
            query_ref=query_ref,
            cost_units=cost_units,
            latency_ms=latency_ms,
        )
        with self._lock:
            self._counters[name.value] += int(value)
            self._samples.append(point)

    def count(self, name: OperationalMetricName, **kwargs: Any) -> None:
        self.record(name, **kwargs)

    def snapshot(self) -> OperationalMetricsSnapshot:
        with self._lock:
            return OperationalMetricsSnapshot(
                counters=dict(self._counters),
                samples=tuple(self._samples),
            )


class AuditRecord(BaseModel):
    """A redacted event projection safe for operational diagnostics."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    episode_id: str = Field(pattern=r"^ep-[a-f0-9]{32}$")
    event_id: str = Field(pattern=r"^[a-f0-9]{64}$")
    sequence: int = Field(ge=1)
    previous_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    capability: str | None = Field(default=None, max_length=120)
    provider: str | None = Field(default=None, max_length=120)
    query_ref: str | None = Field(default=None, pattern=r"^qry-[a-f0-9]{16}$")
    entity_scope_digest: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    time_range: dict[str, Any] | None = None
    cost_units: int | None = Field(default=None, ge=0)
    adapter_ref: str | None = Field(default=None, max_length=160)
    core_decision: str | None = Field(default=None, max_length=120)
    stop_reason: str | None = Field(default=None, max_length=120)
    handoff: str | None = Field(default=None, max_length=120)
    answer_artifact_digest: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    replay_result: str | None = Field(default=None, max_length=80)
    failure_category: str | None = Field(default=None, max_length=80)

    @field_validator("time_range")
    @classmethod
    def _time_range_is_redacted(cls, value: dict[str, Any] | None) -> dict[str, Any] | None:
        if value is None:
            return None
        forbidden = {"path", "filesystem", "token", "secret", "prompt", "trace_id"}
        if forbidden & {str(key).casefold() for key in value}:
            raise ValueError("audit time range contains a forbidden field")
        return value


def audit_record_from_event(event: Any) -> AuditRecord:
    """Project only host-approved metadata from one TranscriptEvent.

    Raw request bodies, observations, prompts, credentials, and provider error
    messages are intentionally ignored.
    """

    payload = event.payload if isinstance(event.payload, dict) else {}
    decision = payload.get("authoritative_decision")
    if isinstance(decision, dict):
        decision = decision.get("status")
    answer_digest = payload.get("artifact_digest")
    if not isinstance(answer_digest, str):
        answer_digest = None
    handoff = payload.get("handoff")
    if not isinstance(handoff, str):
        handoff = None
    return AuditRecord(
        episode_id=event.episode_id,
        event_id=event.event_id,
        sequence=event.sequence,
        previous_digest=event.previous_digest,
        capability=_string(payload.get("evidence_capability") or payload.get("capability")),
        provider=_string(payload.get("evidence_provider") or payload.get("provider")),
        query_ref=_query_ref(payload.get("query_ref")),
        entity_scope_digest=_digest(payload.get("evidence_scope_digest")
                                    or payload.get("entity_scope_digest")),
        time_range=_time_range(payload.get("evidence_time_range") or payload.get("time_range")),
        cost_units=_cost(payload.get("evidence_cost_units") or payload.get("units")),
        adapter_ref=_opaque_ref(payload.get("adapter_ref") or payload.get("reasoning_ref")),
        core_decision=_string(decision),
        stop_reason=_string(
            payload.get("reason")
            or payload.get("stop_reason")
            or payload.get("termination_reason")
        ),
        handoff=handoff or (
            event.event_type.value if "handoff" in event.event_type.value else None
        ),
        answer_artifact_digest=answer_digest,
        replay_result=_string(payload.get("replay_result")),
        failure_category=_string(payload.get("failure_category")),
    )


def audit_transcript(
    events: Iterable[Any],
    *,
    replay_result: str | None = None,
) -> tuple[AuditRecord, ...]:
    records = tuple(audit_record_from_event(event) for event in events)
    if replay_result is None or not records:
        return records
    return records[:-1] + (records[-1].model_copy(update={"replay_result": replay_result}),)


def _string(value: Any) -> str | None:
    return value if isinstance(value, str) and len(value) <= 120 else None


def _query_ref(value: Any) -> str | None:
    return value if isinstance(value, str) and value.startswith("qry-") else None


def _digest(value: Any) -> str | None:
    return value if isinstance(value, str) and len(value) == 64 and all(
        char in "0123456789abcdef" for char in value
    ) else None


def _cost(value: Any) -> int | None:
    return value if type(value) is int and value >= 0 else None


def _opaque_ref(value: Any) -> str | None:
    if not isinstance(value, str) or len(value) > 160:
        return None
    lowered = value.casefold()
    if any(token in lowered for token in ("/", "\\", "token", "secret", "prompt")):
        return None
    return value


def _time_range(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    allowed = {"start", "end", "seconds", "timezone"}
    if not set(value) <= allowed:
        return None
    return {key: value[key] for key in value}
