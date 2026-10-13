from __future__ import annotations

import json
import time
from pathlib import Path

from fastapi.testclient import TestClient

from backend.app.config import Settings
from backend.app.main import create_app
from backend.app.services import run_executor
from rca_agent.environment import Rca100Environment
from rca_agent.models import RequestStopAction
from rca_agent.runner import EpisodeRunner


def test_case_api_lists_dynamic_rca100_cases_without_private_fields(cases_root: Path, tmp_path: Path):
    client = TestClient(create_app(Settings(cases_root=cases_root, runs_root=tmp_path / "runs")))

    response = client.get("/api/v1/cases")

    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] >= 100
    assert any(item["case_id"] == "t021" for item in payload["items"])
    encoded = response.text.lower()
    assert "ground_truth" not in encoded
    assert "answer_key" not in encoded
    assert str(cases_root).lower() not in encoded


def test_recorded_run_is_projected_only_after_replay(cases_root: Path, tmp_path: Path):
    runs_root = tmp_path / "runs"
    run_dir = runs_root / "run-demo-001"
    run_dir.mkdir(parents=True)
    transcript_path = run_dir / "transcript.jsonl"
    runner = EpisodeRunner(
        Rca100Environment.from_case(cases_root, "t021"),
        transcript_path=transcript_path,
    )
    runner.start()
    answer = runner.handle(RequestStopAction(summary="recorded console fixture"))
    assert answer is not None
    (run_dir / "answer.json").write_text(answer.model_dump_json(), encoding="utf-8")
    (run_dir / "manifest.json").write_text(json.dumps({
        "schema_version": 1,
        "run_id": "run-demo-001",
        "case_id": "t021",
        "status": "succeeded",
        "started_at": "2026-03-17T02:44:09+08:00",
        "finished_at": "2026-03-17T02:44:10+08:00",
        "profile_id": "scripted-fixture",
        "model_label": "Fake/Scripted",
    }), encoding="utf-8")
    client = TestClient(create_app(Settings(cases_root=cases_root, runs_root=runs_root)))

    response = client.get("/api/v1/runs/run-demo-001")

    assert response.status_code == 200
    payload = response.json()
    assert payload["run"]["replay_status"] == "valid"
    assert payload["case"]["case_id"] == "t021"
    assert payload["final_decision"] is not None
    assert "ground_truth" not in response.text.lower()


def test_live_scripted_run_persists_state_and_replays(cases_root: Path, tmp_path: Path):
    runs_root = tmp_path / "runs"
    app = create_app(Settings(cases_root=cases_root, runs_root=runs_root))
    with TestClient(app) as client:
        created = client.post(
            "/api/v1/cases/t021/runs",
            json={"profile_id": "scripted-trace-demo"},
        )
        assert created.status_code == 202
        run_id = created.json()["run_id"]

        summary = None
        for _ in range(100):
            response = client.get(f"/api/v1/runs/{run_id}/status")
            assert response.status_code == 200
            summary = response.json()
            if summary["status"] in {"succeeded", "failed"}:
                break
            time.sleep(0.01)

        assert summary is not None
        assert summary["status"] == "succeeded"
        assert summary["replay_status"] == "valid"
        detail = client.get(f"/api/v1/runs/{run_id}")
        assert detail.status_code == 200
        payload = detail.json()
        assert payload["run"]["query_count"] >= 1
        assert payload["evidence"]
        assert "ground_truth" not in detail.text.lower()
        assert "answer_key" not in detail.text.lower()


def test_live_run_rejects_unapproved_profile(cases_root: Path, tmp_path: Path):
    client = TestClient(create_app(Settings(cases_root=cases_root, runs_root=tmp_path / "runs")))

    response = client.post(
        "/api/v1/cases/t021/runs",
        json={"profile_id": "arbitrary-provider"},
    )

    assert response.status_code == 422


def test_qwen_live_run_requires_host_credential(cases_root: Path, tmp_path: Path, monkeypatch):
    monkeypatch.delenv("RCA_LOCAL_API_KEY", raising=False)
    client = TestClient(create_app(Settings(cases_root=cases_root, runs_root=tmp_path / "runs")))

    response = client.post(
        "/api/v1/cases/t021/runs",
        json={"profile_id": "qwen-investigation"},
    )

    assert response.status_code == 503
    assert "RCA_LOCAL_API_KEY" not in response.text


def test_qwen_live_run_uses_server_profile_and_replays(
    cases_root: Path,
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setenv("RCA_LOCAL_API_KEY", "test-only-secret")

    def fake_qwen(settings, case_id, transcript_path):
        return run_executor._run_scripted_trace_profile(
            settings.cases_root,
            case_id,
            transcript_path,
        )

    monkeypatch.setattr(run_executor, "_run_qwen_investigation_profile", fake_qwen)
    settings = Settings(
        cases_root=cases_root,
        runs_root=tmp_path / "runs",
        qwen_model="Qwen test profile",
    )
    app = create_app(settings)
    with TestClient(app) as client:
        created = client.post(
            "/api/v1/cases/t021/runs",
            json={"profile_id": "qwen-investigation"},
        )
        assert created.status_code == 202
        run_id = created.json()["run_id"]

        summary = None
        for _ in range(100):
            summary = client.get(f"/api/v1/runs/{run_id}/status").json()
            if summary["status"] in {"succeeded", "failed"}:
                break
            time.sleep(0.01)

        assert summary is not None
        assert summary["status"] == "succeeded"
        assert summary["profile_id"] == "qwen-investigation"
        assert summary["model_label"] == "Qwen test profile"
        assert summary["replay_status"] == "valid"
        assert "test-only-secret" not in json.dumps(summary)


def test_interrupted_live_run_is_recovered_as_failed(cases_root: Path, tmp_path: Path):
    run_dir = tmp_path / "runs" / "run-live-interrupted"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(json.dumps({
        "schema_version": 1,
        "run_id": "run-live-interrupted",
        "case_id": "t021",
        "status": "running",
        "started_at": "2026-03-17T02:44:09+08:00",
        "finished_at": None,
        "profile_id": "scripted-trace-demo",
        "model_label": "No model",
        "error_code": None,
    }), encoding="utf-8")

    client = TestClient(create_app(Settings(cases_root=cases_root, runs_root=tmp_path / "runs")))
    response = client.get("/api/v1/runs/run-live-interrupted/status")

    assert response.status_code == 200
    assert response.json()["status"] == "failed"
    assert response.json()["error_code"] == "HostRestarted"
