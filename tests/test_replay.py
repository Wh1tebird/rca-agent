from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from rca_agent.environment.rca100 import Rca100Environment
from rca_agent.models import (
    CallToolAction,
    GetMetricSeriesRequest,
    RequestStopAction,
    TimeWindow,
    ToolName,
)
from rca_agent.runner import EpisodeRunner, RecordedReplay, ReplayValidationError, TranscriptEvent


def _finished_runner(cases_root: Path, path: Path) -> EpisodeRunner:
    runner = EpisodeRunner(
        Rca100Environment.from_case(cases_root, "t025", query_budget=3),
        transcript_path=path,
    )
    bootstrap = runner.start()
    duration = (bootstrap.observation_horizon.end - bootstrap.observation_horizon.start) / 2
    window = TimeWindow(
        start=bootstrap.observation_horizon.start + duration,
        end=bootstrap.observation_horizon.end,
    )
    runner.handle(
        CallToolAction(
            tool_name=ToolName.GET_METRIC_SERIES,
            request=GetMetricSeriesRequest(
                entity_ref=bootstrap.alert_entity.entity_ref,
                signal="request_count",
                window=window,
                max_points=3,
            ),
        )
    )
    runner.handle(RequestStopAction(summary="recorded replay fixture complete"))
    return runner


def test_recorded_replay_validates_without_backend_access(cases_root: Path, tmp_path: Path):
    path = tmp_path / "episode.jsonl"
    runner = _finished_runner(cases_root, path)

    report_from_memory = RecordedReplay.verify(runner.transcript.events)
    report_from_disk = RecordedReplay.from_path(path)

    assert report_from_memory == report_from_disk
    assert report_from_memory.valid
    assert report_from_memory.tool_calls == 1
    assert report_from_memory.final_status.value == "answer_frozen"
    assert RecordedReplay.verify(runner.transcript.events) == report_from_memory


@pytest.mark.parametrize("tamper", ["delete", "insert", "reorder", "modify"])
def test_transcript_tampering_is_detected(cases_root: Path, tmp_path: Path, tamper: str):
    events = list(_finished_runner(cases_root, tmp_path / f"{tamper}.jsonl").transcript.events)
    if tamper == "delete":
        del events[2]
    elif tamper == "insert":
        events.insert(2, events[1])
    elif tamper == "reorder":
        events[2], events[3] = events[3], events[2]
    else:
        raw = deepcopy(events[2].model_dump(mode="python"))
        raw["payload"]["action_type"] = "modified"
        events[2] = TranscriptEvent.model_validate(raw)

    with pytest.raises(ReplayValidationError):
        RecordedReplay.verify(events)


def test_query_fingerprint_or_budget_tampering_fails(cases_root: Path, tmp_path: Path):
    events = list(_finished_runner(cases_root, tmp_path / "budget.jsonl").transcript.events)
    cost_index = next(
        index for index, event in enumerate(events) if event.event_type.value == "query_cost"
    )
    raw = events[cost_index].model_dump(mode="python")
    raw["payload"]["budget_after"] += 1
    events[cost_index] = TranscriptEvent.model_validate(raw)

    with pytest.raises(ReplayValidationError):
        RecordedReplay.verify(events)
