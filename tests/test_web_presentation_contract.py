from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

from backend.app.services.run_projector import _decision_status, _evidence
from backend.app.services.run_store import RunStore
from rca_agent.models import (
    AnswerArtifact,
    AuthoritativeDecision,
    DecisionSource,
    DecisionStatus,
    HypothesisProposal,
    Observation,
    StopReason,
)
from rca_agent.runner.transcript import EventType, Transcript


def test_observation_projection_keeps_candidate_admission_explicit_and_neutral():
    episode_id = "ep-" + "a" * 32
    observation_id = "obs-" + "b" * 16
    query_ref = "qry-" + "c" * 16
    transcript = Transcript(episode_id)
    observation = Observation(
        observation_id=observation_id,
        modality="trace",
        entity_ref="ent-app",
        signal="connection_duration_ms",
        value=0.1,
        data_status="observed",
        details={"trace_ref": "trace-" + "d" * 16, "path": "/private/case"},
    )
    event = transcript.append(EventType.TOOL_RESPONSE, {
        "tool_name": "get_trace_stage_breakdown",
        "query_ref": query_ref,
        "status": "success",
        "observations": [observation.model_dump(mode="json")],
    })
    proposal = HypothesisProposal(
        hypothesis_id="h-one",
        candidate_id="CandidateOne",
        mechanism="bounded mechanism",
        causal_entity_ref="ent-app",
        rationale="proposal only",
        observation_refs=(observation_id,),
    )
    answer = AnswerArtifact(
        episode_id=episode_id,
        termination_reason=StopReason.POLICY_REQUEST,
        policy_proposals=(proposal,),
        authoritative_decision=AuthoritativeDecision(
            status=DecisionStatus.INCONCLUSIVE,
            source=DecisionSource.CORE,
            reason="insufficient evidence",
        ),
        observation_refs=(observation_id,),
        missing_evidence=("independent signal",),
        transcript_digest=event.digest,
        component_versions={},
        frozen_at=datetime.now(UTC),
        artifact_digest="e" * 64,
    )
    case = SimpleNamespace(
        window_start=datetime(2026, 1, 1, tzinfo=UTC),
        window_end=datetime(2026, 1, 1, 0, 5, tzinfo=UTC),
    )

    evidence, _ = _evidence(transcript.events, case, answer)

    assert len(evidence) == 1
    assert evidence[0].role == "admission"
    assert evidence[0].candidate_ids == ("h-one",)
    assert evidence[0].observation_note.startswith("Neutral Observation")
    assert evidence[0].admission_context == "Explicitly admitted by Candidate refs: h-one"
    assert "path" not in evidence[0].raw
    assert evidence[0].raw["details"]["trace_ref"].startswith("trace-")


def test_decision_projection_preserves_core_statuses():
    assert _decision_status("inconclusive") == "INCONCLUSIVE"
    assert _decision_status("out_of_scope") == "OUT_OF_SCOPE"
    assert _decision_status("query_failed") == "QUERY_FAILED"
    assert _decision_status("supported_hypothesis") == "SUPPORTED"
    assert _decision_status("confirmed_root_cause") == "VERIFIED"


def test_malformed_transcript_is_corrupted_projection(tmp_path):
    run_dir = tmp_path / "run-corrupted-001"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(
        '{"schema_version": 1, "run_id": "run-corrupted-001", "case_id": "t001", '
        '"status": "failed", "started_at": "2026-01-01T00:00:00+00:00", '
        '"profile_id": "scripted-trace-demo"}',
        encoding="utf-8",
    )
    (run_dir / "transcript.jsonl").write_text("not-json\n", encoding="utf-8")

    stored = RunStore(tmp_path).get_run("run-corrupted-001")

    assert stored.replay_status.value == "invalid"
    assert stored.recovery.state.value == "corrupted"
    assert stored.recovery.failure_kind == "transcriptinvalid"
