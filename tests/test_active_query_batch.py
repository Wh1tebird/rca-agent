from types import SimpleNamespace

import pytest

from evaluation import record_active_formalizer_qwen as recorder
from rca_agent.runner.transcript import load_jsonl_events


def test_baseline_metrics_count_executed_queries_without_active_events():
    from test_bound_formalization import run_bound

    runner, _, _ = run_bound()
    assert recorder._evidence(runner.transcript.events)["active_selections"] == 0


def test_paired_preflight_skips_invalid_cases_and_reuses_recorder(monkeypatch, tmp_path):
    monkeypatch.setattr(recorder.Settings, "from_environment", lambda: SimpleNamespace(
        cases_root=tmp_path, qwen_query_budget=20))
    class Environment:
        @staticmethod
        def from_case(root, case_id, query_budget):
            if case_id == "t001":
                raise ValueError("private path must not be saved")
            return SimpleNamespace(reset=lambda: None)
    monkeypatch.setattr(recorder, "Rca100Environment", Environment)
    calls = []
    def record(case_ids, runs_root, **kwargs):
        calls.append((case_ids, kwargs["workspace_mode"]))
        return [{"case_id": case_ids[0], "workspace_mode": kwargs["workspace_mode"]}]
    monkeypatch.setattr(recorder, "record_cases", record)
    rows = recorder.record_paired(("t001", "t007"), tmp_path / "output", workers=1)
    assert len(rows) == 2
    assert calls == [(("t007",), "shadow"), (("t007",), "active")]
    assert "private path" not in (tmp_path / "output/preflight.json").read_text()
    with pytest.raises(ValueError):
        recorder.record_paired(("t007", "t007"), tmp_path)


def test_metrics_read_current_frozen_fixture_if_available():
    from pathlib import Path
    fixtures = list(Path("analysis/bound_discovery_v6_qwen_20260928").rglob("transcript.jsonl"))
    if not fixtures:
        pytest.skip("local non-distributed smoke artifacts")
    for path in fixtures:
        events = load_jsonl_events(path)
        metrics = recorder._evidence(events)
        assert metrics["inquiry_queries_executed"] == metrics["active_selections"]
        legacy_view = tuple(e for e in events if e.event_type.value != "discrimination_active_selection_recorded")
        assert recorder._evidence(legacy_view)["inquiry_queries_executed"] == metrics["inquiry_queries_executed"]
