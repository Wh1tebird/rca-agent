from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from evaluation.artifacts import (
    OfflineArtifactEvaluator,
    UsageCall,
    persist_benchmark_result,
)
from rca_agent.runner.replay import RecordedReplay, ReplayValidationError
from rca_agent.runner.transcript import EventType, Transcript, load_jsonl_events


def _result() -> dict:
    return {
        "case_id": "t030", "acquisition_mode": "fixed",
        "episode_id": "ep-" + "a" * 32, "status": "supported_hypothesis",
        "answer_digest": "b" * 64, "replay_status": "valid",
        "transcript_head_digest": "c" * 64, "stop_reason": "max_rounds",
        "policy_proposals": [], "registered_candidates": [],
        "observation_refs": ["obs-abc"],
    }


def test_frozen_answer_projection_does_not_copy_prompt_or_path(tmp_path: Path):
    result = _result()
    result["prompt"] = "secret prompt"
    persist_benchmark_result(result, tmp_path)
    payload = json.loads((tmp_path / "frozen_answer.json").read_text())
    encoded = json.dumps(payload)
    assert "secret prompt" not in encoded
    assert "/Users/" not in encoded
    assert payload["answer_digest_status"] == "not_evaluable"


def test_answer_digest_mismatch_is_not_claimed_verified(tmp_path: Path):
    result = _result()
    persist_benchmark_result(result, tmp_path, answer_payload={"status": "different"})
    payload = json.loads((tmp_path / "frozen_answer.json").read_text())
    assert payload["answer_digest_status"] == "not_evaluable"


def test_evaluator_input_requires_answer_freeze(tmp_path: Path):
    result = _result()
    result.pop("answer_digest")
    with pytest.raises(ValueError, match="Answer Freeze"):
        persist_benchmark_result(result, tmp_path)


def test_corrupted_transcript_is_rejected(tmp_path: Path):
    transcript_path = tmp_path / "events.jsonl"
    transcript = Transcript("ep-" + "d" * 32)
    event = transcript.append(EventType.EPISODE_CREATED, {"episode_id": "ep-" + "d" * 32})
    transcript_path.write_text(event.model_dump_json() + "\n", encoding="utf-8")
    transcript_path.write_text(transcript_path.read_text().replace(event.digest, "0" * 64), encoding="utf-8")
    with pytest.raises(ReplayValidationError):
        RecordedReplay.verify(load_jsonl_events(transcript_path))


def test_usage_available_and_unavailable_round_trip(tmp_path: Path):
    calls = [UsageCall(provider="deepseek", model="deepseek-flash", call_id="c1", phase="discovery",
                       started_at=datetime.now(UTC), finished_at=datetime.now(UTC), duration_seconds=1.2,
                       input_tokens=10, output_tokens=5, total_tokens=15, usage_source="provider"),
             UsageCall(provider="deepseek", model="deepseek-flash", call_id="c2", phase="planner",
                       duration_seconds="unavailable", usage_source="unavailable", unavailable_reason="provider omitted usage")]
    persist_benchmark_result(_result(), tmp_path, usage_calls=calls)
    data=json.loads((tmp_path/"usage.json").read_text())
    assert data["calls"][0]["total_tokens"] == 15
    assert data["calls"][1]["total_tokens"] == "unavailable"


def test_offline_evaluator_returns_not_evaluable_without_ground_truth(tmp_path: Path):
    persist_benchmark_result(_result(), tmp_path)
    report=OfflineArtifactEvaluator().load_report(tmp_path)
    assert report.outcome.result.value == "not_evaluable"
    assert report.outcome.detail_code == "answer_digest_not_verifiable_from_persisted_projection"


def test_transcript_round_trip_and_corruption_detection(tmp_path: Path):
    transcript_path=tmp_path/"source.jsonl"
    store=type("Store", (), {"path": transcript_path, "append": lambda self, event: self.path.open("a").write(event.model_dump_json()+"\n")})()
    transcript=Transcript("ep-"+"d"*32, store=store)
    transcript.append(EventType.EPISODE_CREATED, {"episode_id":"ep-"+"d"*32})
    result=_result() | {"episode_id":"ep-"+"d"*32}
    out=tmp_path/"artifact"
    persist_benchmark_result(result, out, transcript_path=transcript_path)
    meta=json.loads((out/"transcript_metadata.json").read_text())
    assert meta["complete"] is True
    assert meta["replay_status"] == "valid"
