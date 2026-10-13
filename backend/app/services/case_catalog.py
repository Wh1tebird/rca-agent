"""Safe, host-side enumeration of RCA100 cases.

Case selection belongs to this host service. It is never serialized into Agent context.
"""
from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

from backend.app.schemas import CaseDetail, CaseReadiness, CaseSummary
from rca_agent.core.cases import OBSERVATION_FILES, BlindTask


class CaseCatalogUnavailable(RuntimeError):
    pass


class CaseNotFound(KeyError):
    pass


class Rca100CaseCatalog:
    def __init__(self, cases_root: Path) -> None:
        self._cases_root = cases_root

    def list_cases(self) -> tuple[CaseSummary, ...]:
        self._require_root()
        result = []
        for case_dir in sorted(self._case_directories(), key=lambda item: item.name):
            parsed = self._parse_case(case_dir)
            if isinstance(parsed, CaseDetail):
                result.append(CaseSummary(**parsed.model_dump(exclude={"prompt_text"})))
            else:
                result.append(parsed)
        return tuple(result)

    def get_case(self, case_id: str) -> CaseDetail:
        self._require_root()
        if not _is_case_id(case_id):
            raise CaseNotFound(case_id)
        case_dir = self._cases_root / case_id
        if not case_dir.is_dir():
            raise CaseNotFound(case_id)
        parsed = self._parse_case(case_dir)
        if isinstance(parsed, CaseDetail):
            return parsed
        return CaseDetail(
            **parsed.model_dump(),
            prompt_text="This Case cannot initialize an investigation until its task metadata is repaired.",
        )

    def _case_directories(self) -> Iterable[Path]:
        return (path for path in self._cases_root.iterdir() if path.is_dir())

    def _parse_case(self, case_dir: Path) -> CaseDetail | CaseSummary:
        task_path = case_dir / "task.json"
        if not task_path.is_file():
            return _invalid_case(case_dir.name, "missing task.json")
        try:
            task = BlindTask.model_validate(json.loads(task_path.read_text(encoding="utf-8")))
        except (OSError, ValueError, json.JSONDecodeError):
            return _invalid_case(case_dir.name, "task.json cannot be validated")
        if task.task_id != case_dir.name:
            return _invalid_case(case_dir.name, "task ID does not match its directory")
        try:
            triggered_at = _parse_datetime(task.alert_trigger_time)
            window_start = _parse_datetime(task.alert_window.start)
            window_end = _parse_datetime(task.alert_window.end)
        except ValueError:
            return _invalid_case(case_dir.name, "alert timestamp is missing or invalid")
        missing = tuple(name for name, filename in OBSERVATION_FILES.items()
                        if not (case_dir / filename).is_file())
        readiness = CaseReadiness.READY if not missing else CaseReadiness.INCOMPLETE
        return CaseDetail(
            case_id=task.task_id,
            title=task.alert_title,
            alert_entity=task.alert_entity.entity_name,
            alert_entity_type=task.alert_entity.entity_type,
            triggered_at=triggered_at,
            window_start=window_start,
            window_end=window_end,
            modalities=tuple(task.available_modalities),
            readiness=readiness,
            missing_modalities=missing,
            run_count=0,
            prompt_text=task.prompt_text,
        )

    def _require_root(self) -> None:
        if not self._cases_root.is_dir():
            raise CaseCatalogUnavailable("RCA100 cases directory is unavailable")


def _invalid_case(case_id: str, reason: str) -> CaseSummary:
    return CaseSummary(
        case_id=case_id if _is_case_id(case_id) else "invalid",
        title="Invalid RCA100 case",
        alert_entity="Unavailable",
        alert_entity_type="unknown",
        triggered_at="1970-01-01T00:00:00+00:00",
        window_start="1970-01-01T00:00:00+00:00",
        window_end="1970-01-01T00:00:00+00:00",
        modalities=(),
        readiness=CaseReadiness.INVALID,
        missing_modalities=(reason,),
        run_count=0,
    )


def _is_case_id(value: str) -> bool:
    return len(value) == 4 and value.startswith("t") and value[1:].isdigit()


def _parse_datetime(value: str) -> datetime:
    if not value.strip():
        raise ValueError("timestamp is empty")
    return datetime.fromisoformat(value)
