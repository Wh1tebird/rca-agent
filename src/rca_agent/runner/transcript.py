from __future__ import annotations

import json
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from rca_agent.tools.fingerprint import sha256_digest

GENESIS_DIGEST = "0" * 64


class EventType(StrEnum):
    EPISODE_CREATED = "episode_created"
    ENVIRONMENT_RESET = "environment_reset"
    POLICY_PROCESS_STARTED = "policy_process_started"
    POLICY_PROCESS_EXITED = "policy_process_exited"
    POLICY_ACTION = "policy_action"
    ACTION_REJECTED = "action_rejected"
    TOOL_REQUEST = "tool_request"
    STARTUP_ACTION_SELECTED = "startup_action_selected"
    TOOL_RESPONSE = "tool_response"
    QUERY_COST = "query_cost"
    HYPOTHESIS_SUBMITTED = "hypothesis_submitted"
    CANDIDATE_EVIDENCE_UPDATED = "candidate_evidence_updated"
    INQUIRY_ROUND_RECORDED = "inquiry_round_recorded"
    INQUIRY_LOOP_VIEW_RECORDED = "inquiry_loop_view_recorded"
    INQUIRY_LOOP_STEP_ACCEPTED = "inquiry_loop_step_accepted"
    INQUIRY_LOOP_ATTEMPT_RECORDED = "inquiry_loop_attempt_recorded"
    INQUIRY_ATTEMPT_RECORDED = "inquiry_attempt_recorded"
    INQUIRY_QUERY_CANCELLED = "inquiry_query_cancelled"
    INVESTIGATION_CANDIDATE_ADDED = "investigation_candidate_added"
    CANDIDATE_TRANSITIONED = "candidate_transitioned"
    CANDIDATE_SUPERSEDED = "candidate_superseded"
    CAPABILITY_REGISTRY_RESOLVED = "capability_registry_resolved"
    DISCRIMINATION_TASK_CREATED = "discrimination_task_created"
    DISCRIMINATION_TASK_SELECTED = "discrimination_task_selected"
    DISCRIMINATION_TASK_COMPLETED = "discrimination_task_completed"
    DISCRIMINATION_TASK_UNVERIFIABLE = "discrimination_task_unverifiable"
    DISCRIMINATION_SHADOW_SELECTION_RECORDED = (
        "discrimination_shadow_selection_recorded"
    )
    DISCRIMINATION_ACTIVE_SELECTION_RECORDED = (
        "discrimination_active_selection_recorded"
    )
    DISCOVERY_STARTED = "discovery_started"
    BOUND_CANDIDATE_FORMALIZATION_RECORDED = "bound_candidate_formalization_recorded"
    DISCOVERY_HANDOFF_REPAIR_REQUESTED = "discovery_handoff_repair_requested"
    DISCOVERY_HANDOFF_VALIDATED = "discovery_handoff_validated"
    DISCOVERY_FORMAT_REPAIR_REQUESTED = "discovery_format_repair_requested"
    DISCOVERY_FORMAT_REPAIR_COMPLETED = "discovery_format_repair_completed"
    DISCOVERY_FORMAT_REPAIR_REJECTED = "discovery_format_repair_rejected"
    MECHANISM_FORMALIZATION_STARTED = "mechanism_formalization_started"
    MECHANISM_FORMALIZATION_RETRIED = "mechanism_formalization_retried"
    MECHANISM_FORMALIZATION_COMPLETED = "mechanism_formalization_completed"
    MECHANISM_FORMALIZATION_REJECTED = "mechanism_formalization_rejected"
    MECHANISM_COMPLETION_REQUESTED = "mechanism_completion_requested"
    MECHANISM_COMPLETION_COMPLETED = "mechanism_completion_completed"
    MECHANISM_COMPLETION_REJECTED = "mechanism_completion_rejected"
    MECHANISM_DRAFT_GENERATED = "mechanism_draft_generated"
    MECHANISM_DRAFT_REJECTED = "mechanism_draft_rejected"
    MECHANISM_DRAFT_FROZEN = "mechanism_draft_frozen"
    VERIFICATION_PLAN_GENERATED = "verification_plan_generated"
    VERIFICATION_PLAN_REJECTED = "verification_plan_rejected"
    VERIFICATION_PLAN_RETRIED = "verification_plan_retried"
    VERIFICATION_PLAN_COMPILED = "verification_plan_compiled"
    VERIFICATION_PLAN_EXTENSION_STARTED = "verification_plan_extension_started"
    VERIFICATION_PLAN_EXTENSION_REJECTED = "verification_plan_extension_rejected"
    VERIFICATION_PLAN_EXTENDED = "verification_plan_extended"
    DISCOVERY_PAYLOAD_REJECTED = "discovery_payload_rejected"
    DISCOVERY_DRAFT_EVALUATED = "discovery_draft_evaluated"
    DISCOVERY_COMPLETED = "discovery_completed"
    DYNAMIC_CANDIDATE_REGISTERED = "dynamic_candidate_registered"
    CATALOG_CANDIDATE_ALIGNED = "catalog_candidate_aligned"
    DYNAMIC_VERIFICATION = "dynamic_verification"
    MECHANISM_COVERAGE_SHADOW_RECORDED = "mechanism_coverage_shadow_recorded"
    STOP_REQUESTED = "stop_requested"
    ANSWER_FROZEN = "answer_frozen"
    EVALUATION_STARTED = "evaluation_started"
    EVALUATION_COMPLETED = "evaluation_completed"
    POLICY_EXCEPTION = "policy_exception"
    POLICY_TIMEOUT = "policy_timeout"
    TOOL_EXCEPTION = "tool_exception"


class TranscriptEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    episode_id: str = Field(pattern=r"^ep-[a-f0-9]{32}$")
    sequence: int = Field(ge=1)
    event_type: EventType
    payload: dict[str, Any]
    recorded_at: datetime
    previous_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    digest: str = Field(pattern=r"^[a-f0-9]{64}$")

    def digest_payload(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude={"digest"})

    def verify_digest(self) -> bool:
        return sha256_digest(self.digest_payload()) == self.digest


class JsonlTranscriptStore:
    """Append-only local persistence; an existing transcript is never overwritten."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=False)

    def append(self, event: TranscriptEvent) -> None:
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(event.model_dump_json())
            stream.write("\n")
            stream.flush()


class Transcript:
    def __init__(
        self,
        episode_id: str,
        store: JsonlTranscriptStore | None = None,
    ) -> None:
        self.episode_id = episode_id
        self.__events: list[TranscriptEvent] = []
        self.__store = store

    @property
    def events(self) -> tuple[TranscriptEvent, ...]:
        return tuple(self.__events)

    @property
    def head_digest(self) -> str:
        return self.__events[-1].digest if self.__events else GENESIS_DIGEST

    def append(self, event_type: EventType, payload: BaseModel | dict[str, Any]) -> TranscriptEvent:
        raw = payload.model_dump(mode="json") if isinstance(payload, BaseModel) else payload
        normalized = json.loads(json.dumps(raw, ensure_ascii=False, sort_keys=True))
        base = {
            "episode_id": self.episode_id,
            "sequence": len(self.__events) + 1,
            "event_type": event_type,
            "payload": normalized,
            "recorded_at": datetime.now(UTC),
            "previous_digest": self.head_digest,
        }
        digest_payload = TypeAdapter(dict[str, Any]).dump_python(base, mode="json")
        event = TranscriptEvent(**base, digest=sha256_digest(digest_payload))
        self.__events.append(event)
        if self.__store is not None:
            self.__store.append(event)
        return event


def load_jsonl_events(path: Path) -> tuple[TranscriptEvent, ...]:
    events = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise ValueError(f"blank transcript line at {line_number}")
            events.append(TranscriptEvent.model_validate_json(line))
    return tuple(events)
