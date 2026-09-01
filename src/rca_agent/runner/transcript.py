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
    TOOL_RESPONSE = "tool_response"
    QUERY_COST = "query_cost"
    HYPOTHESIS_SUBMITTED = "hypothesis_submitted"
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
