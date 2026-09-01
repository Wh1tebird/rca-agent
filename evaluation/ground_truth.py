from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

CASE_ID_RE = re.compile(r"^t\d{3}$")


class ExpectedValue(BaseModel):
    model_config = ConfigDict(extra="allow")

    comparator: Literal[">=", "<=", "==", ">", "<"]
    value: float
    unit: str = ""


class ExpectedCheckpoint(BaseModel):
    model_config = ConfigDict(extra="allow")

    source_type: str
    signal: str
    required: bool = True
    expected: ExpectedValue


class GroundTruthStep(BaseModel):
    model_config = ConfigDict(extra="allow")

    step: int
    step_type: Literal["cause", "propagation", "impact"]
    target: str
    fault_id: str
    required: bool = True
    queryable: bool = True
    observability: list[ExpectedCheckpoint] = Field(default_factory=list)


class GroundTruthCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    root_cause_entities: list[str]
    root_cause_types: list[str]
    expected_fault_id: str
    steps: list[GroundTruthStep]


class FlatGroundTruthRepository:
    """Evaluator-only loader for the RCA100 answer-key layout."""

    def __init__(self, answer_key_root: Path) -> None:
        self._answer_key_root = answer_key_root.resolve()

    def load(self, case_id: str) -> GroundTruthCase:
        if not CASE_ID_RE.fullmatch(case_id):
            raise ValueError(f"invalid case id: {case_id!r}")
        path = (self._answer_key_root / f"{case_id}.gt.json").resolve()
        if path.parent != self._answer_key_root:
            raise ValueError("ground-truth path escapes answer-key root")
        payload = json.loads(path.read_text())
        raw = json.loads(payload["raw_ground_truth"])
        return GroundTruthCase(
            case_id=case_id,
            root_cause_entities=payload["root_cause_entities"],
            root_cause_types=payload["root_cause_types"],
            expected_fault_id=raw["outcome"]["expected_fault_id"],
            steps=raw["reasoning"]["steps"],
        )

