from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

CASE_ID_RE = re.compile(r"^t\d{3}$")
OBSERVATION_FILES = {
    "metrics": "metrics.parquet",
    "logs": "logs.parquet",
    "traces": "traces.parquet",
    "events": "events.parquet",
    "alerts": "alerts.parquet",
    "topology": "topology.json",
}


class AlertWindow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start: str
    end: str


class AlertEntity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entity_id: str
    entity_name: str
    entity_type: str
    entity_domain: str


class BlindTask(BaseModel):
    """Agent-visible case contract with no ground-truth fields."""

    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(pattern=r"^t\d{3}$")
    task_version: str
    alert_event_id: str
    alert_title: str
    alert_trigger_time: str
    alert_window: AlertWindow
    alert_entity: AlertEntity
    prompt_text: str
    workspace: str
    region_id: str
    available_modalities: list[str]
    scoring_note: str
    alert_trans_id: str | None = None


@dataclass(frozen=True, slots=True)
class BlindCase:
    task: BlindTask
    observations: dict[str, Path]


class FlatBlindCaseRepository:
    """Load an RCA100 flat case without exposing answer-key data."""

    def __init__(self, cases_root: Path) -> None:
        self._cases_root = cases_root.resolve()

    def load(self, case_id: str) -> BlindCase:
        case_dir = self._case_dir(case_id)
        raw: dict[str, Any] = json.loads((case_dir / "task.json").read_text())
        task = BlindTask.model_validate(raw)
        if task.task_id != case_id:
            raise ValueError(f"task id mismatch: requested {case_id}, found {task.task_id}")
        observations = {
            name: case_dir / filename for name, filename in OBSERVATION_FILES.items()
        }
        missing = [str(path) for path in observations.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"missing observation files: {missing}")
        return BlindCase(task=task, observations=observations)

    def _case_dir(self, case_id: str) -> Path:
        if not CASE_ID_RE.fullmatch(case_id):
            raise ValueError(f"invalid case id: {case_id!r}")
        case_dir = (self._cases_root / case_id).resolve()
        if case_dir.parent != self._cases_root:
            raise ValueError("case path escapes cases root")
        if not case_dir.is_dir():
            raise FileNotFoundError(case_dir)
        return case_dir

