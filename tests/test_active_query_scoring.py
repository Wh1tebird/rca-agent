from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from evaluation.active_query_comparison import direction_match, evaluate, query_metrics, score_run
from evaluation.record_active_formalizer_qwen import _write_json


def event(kind, payload, seconds=0):
    return SimpleNamespace(event_type=SimpleNamespace(value=kind), payload=payload,
                           recorded_at=datetime(2026, 9, 28, tzinfo=UTC) + timedelta(seconds=seconds))


def test_exact_subtype_not_generic_family_and_pool_not_sql_execution():
    assert direction_match("fullGC", "GC pauses", exact=False)
    assert not direction_match("fullGC", "GC pauses", exact=True)
    assert direction_match("cpuDeadLoop", "high CPU", exact=False)
    assert not direction_match("cpuDeadLoop", "high CPU", exact=True)
    assert not direction_match("slowSQL", "High SQL duration due to connection pool exhaustion", exact=True)
    assert not direction_match("slowSQL", "thread pool exhaustion causing slow SQL queries", exact=True)
    assert not direction_match("threadExhaustion", "GC pauses causing temporary thread starvation", exact=True)
    assert direction_match("slowSQL", "SQL cross-join execution", exact=True)


def test_success_missing_is_not_effective_and_zero_is_valid():
    observation = {"data_status": "observed", "value": 0, "signal": "cpu", "entity_ref": "ent-service",
                   "observation_id": "obs-1", "modality": "metric"}
    events = [event("capability_registry_resolved", {"capabilities": [
        {"entity_ref": "ent-service", "signal": "cpu", "support_eligible": True,
         "independent_group": "cpu"},
        {"entity_ref": "ent-service", "signal": "cpu", "support_eligible": False,
         "independent_group": "cpu"},
    ]}), event("tool_response", {"query_ref": "q1", "status": "success",
        "observations": [observation]}), event("tool_response", {
        "query_ref": "q2", "status": "success", "observations": [
            {**observation, "observation_id": "obs-2", "data_status": "missing", "value": None},
        ]}, seconds=2)]
    metrics = query_metrics(events)
    assert metrics["status_success_queries"] == 2
    assert metrics["trusted_nonempty_queries"] == 1
    assert metrics["eligible_groups"] == ["cpu"]


def test_gt_loader_is_never_called_before_replay_pass(monkeypatch, tmp_path):
    from test_bound_formalization import run_bound

    from evaluation import active_query_comparison as scoring
    runner, _, _ = run_bound()
    _write_json(tmp_path / "config.json", {"case_id": "t007"})
    _write_json(tmp_path / "answer.json", runner.answer.model_dump(mode="json"))
    (tmp_path / "transcript.jsonl").write_text("\n".join(
        e.model_dump_json() for e in runner.transcript.events) + "\n")
    monkeypatch.setattr(scoring.RecordedReplay, "verify", lambda events: SimpleNamespace(valid=False))
    repository = SimpleNamespace(load=lambda case_id: pytest.fail("GT accessed before valid Replay"))
    with pytest.raises(ValueError, match="GT must not be loaded"):
        score_run(tmp_path, repository)


def test_shadow_counts_use_handoff_links_not_all_tools():
    rounds = event("inquiry_round_recorded", {"draft": {"choices": [{}, {}]}})
    handoff = event("discovery_started", {"inquiry_handoff": {
        "query_links": [{"query_ref": "inquiry-1"}, {"query_ref": "inquiry-2"}]}})
    requests = [event("tool_request", {"query_ref": ref})
                for ref in ("initial-trace", "inquiry-1", "inquiry-2", "planner-1")]
    events = [requests[0], rounds, requests[1], requests[2], handoff, requests[3]]
    for sequence, e in enumerate(events):
        e.sequence = sequence
    metrics = query_metrics(events)
    assert metrics["inquiry_queries_offered"] == 2
    assert metrics["inquiry_queries_executed"] == 2
    assert metrics["inquiry_queries_not_executed"] == 0
    assert metrics["inquiry_count_source"] == "validated_handoff"


def test_batch_incomplete_blocks_gt_repository_creation(monkeypatch, tmp_path):
    from evaluation import active_query_comparison as scoring

    _write_json(tmp_path / "preflight.json", [{"case_id": "t007", "valid": True}])
    _write_json(tmp_path / "batch_results.json", [{"case_id": "t007", "workspace_mode": "active"}])
    monkeypatch.setattr(scoring, "FlatGroundTruthRepository",
                        lambda root: pytest.fail("GT repository created before batch completed"))
    with pytest.raises(ValueError, match="incomplete"):
        evaluate(tmp_path, tmp_path)
