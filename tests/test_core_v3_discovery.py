from __future__ import annotations

from rca_agent.core.v3.capabilities import rca100_capabilities
from rca_agent.core.v3.catalog import mechanism_catalog
from rca_agent.core.v3.discovery import (
    AlignmentDecision,
    AlignmentStatus,
    CandidateAlignmentResult,
    DeepSeekJsonClient,
    DeepSeekMechanismDiscovery,
    MechanismDiscoveryResult,
    MechanismHypothesis,
    PredictedObservation,
    compile_aligned_proposal,
)
from rca_agent.core.v3.models import Comparison, Incident, MechanismFamily, Observation


def _incident() -> Incident:
    return Incident(
        incident_id="secret-case-id",
        alert_title="inventory latency",
        alert_entity="inventory::operation",
        window_start="start",
        window_end="end",
    )


def _hypothesis() -> MechanismHypothesis:
    return MechanismHypothesis(
        hypothesis_id="abnormal_query_execution",
        mechanism="abnormal database query execution",
        family=MechanismFamily.DATABASE_DEPENDENCY,
        causal_explanation="Request latency concentrates in database query execution.",
        explained_observations=["sql_duration_share", "sql_shape"],
        unexplained_observations=["specific SQL optimization cause remains unknown"],
        predicted_observations=[
            PredictedObservation(
                signal="sql_duration_share",
                comparison=Comparison.GTE,
                threshold=0.8,
                independent_group="execution_duration",
                rationale="SQL execution should dominate the trace.",
            ),
            PredictedObservation(
                signal="sql_shape",
                comparison=Comparison.EQ,
                threshold="complex_cross_join",
                independent_group="query_structure",
                rationale="A complex query shape independently supports the mechanism.",
            ),
        ],
        confidence=0.9,
    )


def test_discovery_request_has_no_catalog_candidate_or_case_id():
    captured = {}

    def transport(request, timeout):
        import json

        body = json.loads(request.data)
        captured["system"] = body["messages"][0]["content"]
        captured["user"] = body["messages"][1]["content"]
        content = json.dumps({"hypotheses": [_hypothesis().model_dump(mode="json")]})
        return json.dumps({"choices": [{"message": {"content": content}}]}).encode()

    discoverer = DeepSeekMechanismDiscovery(
        DeepSeekJsonClient(api_key="test", transport=transport)
    )
    result = discoverer.discover(
        _incident(),
        rca100_capabilities(),
        [
            Observation(
                signal="sql_duration_share",
                value=0.998,
                query_ref="private-ref",
                observed_entity="inventory",
            ),
            Observation(
                signal="sql_shape",
                value="complex_cross_join",
                query_ref="private-ref",
                observed_entity="inventory",
            ),
        ],
    )

    assert result.hypotheses[0].mechanism == "abnormal database query execution"
    assert "candidate_catalog" not in captured["user"]
    assert "slowSQL" not in captured["user"]
    assert "secret-case-id" not in captured["user"]
    assert "private-ref" not in captured["user"]


def test_discovery_rejects_explained_or_predicted_invented_signals():
    import json

    raw = _hypothesis().model_dump(mode="json")
    raw["predicted_observations"][0]["signal"] = "invented_signal"

    def transport(request, timeout):
        content = json.dumps({"hypotheses": [raw]})
        return json.dumps({"choices": [{"message": {"content": content}}]}).encode()

    result = DeepSeekMechanismDiscovery(
        DeepSeekJsonClient(api_key="test", transport=transport)
    ).discover(
        _incident(),
        rca100_capabilities(),
        [Observation(signal="sql_shape", value="complex_cross_join", query_ref="q", observed_entity="x")],
    )

    assert result.hypotheses == []
    assert result.rejected_hypotheses


def test_alignment_can_reuse_existing_candidate():
    discovery = MechanismDiscoveryResult(generator="test", hypotheses=[_hypothesis()])
    alignment = CandidateAlignmentResult(
        generator="test",
        decisions=[
            AlignmentDecision(
                hypothesis_id="abnormal_query_execution",
                status=AlignmentStatus.MATCH_EXISTING,
                candidate_id="slowSQL",
                rationale="same causal mechanism",
            )
        ],
    )
    proposal = compile_aligned_proposal(
        discovery,
        alignment,
        list(mechanism_catalog("inventory")),
        rca100_capabilities(),
    )

    assert [item.candidate_id for item in proposal.candidates] == ["slowSQL"]


def test_hidden_catalog_compiles_new_candidate_with_two_owned_conditions():
    discovery = MechanismDiscoveryResult(generator="test", hypotheses=[_hypothesis()])
    alignment = CandidateAlignmentResult(
        generator="test",
        decisions=[
            AlignmentDecision(
                hypothesis_id="abnormal_query_execution",
                status=AlignmentStatus.CREATE_NEW,
                rationale="no semantically equivalent catalog mechanism",
            )
        ],
    )
    proposal = compile_aligned_proposal(
        discovery,
        alignment,
        [item for item in mechanism_catalog("inventory") if item.candidate_id != "slowSQL"],
        rca100_capabilities(),
    )

    candidate = proposal.candidates[0]
    assert candidate.candidate_id == "abnormal_query_execution"
    assert len(candidate.positive_evidence_requirements) == 2
    assert {item.independent_group for item in candidate.conditions} == {
        "database_query_duration",
        "database_query_structure",
    }


def test_new_candidate_with_one_evidence_group_is_rejected():
    hypothesis = _hypothesis().model_copy(
        update={
            "explained_observations": ["sql_duration_share"],
            "predicted_observations": [_hypothesis().predicted_observations[0]],
        }
    )
    proposal = compile_aligned_proposal(
        MechanismDiscoveryResult(generator="test", hypotheses=[hypothesis]),
        CandidateAlignmentResult(
            generator="test",
            decisions=[
                AlignmentDecision(
                    hypothesis_id=hypothesis.hypothesis_id,
                    status=AlignmentStatus.CREATE_NEW,
                    rationale="new",
                )
            ],
        ),
        [],
        rca100_capabilities(),
    )

    assert proposal.candidates == []
    assert proposal.rejected_candidates


def test_ai_cannot_turn_low_cpu_into_positive_resource_threshold():
    hypothesis = _hypothesis().model_copy(
        update={
            "hypothesis_id": "resource_contention",
            "mechanism": "generic resource contention",
            "family": MechanismFamily.RESOURCE,
            "explained_observations": ["node_cpu_peak", "service_cpu_ratio"],
            "predicted_observations": [
                PredictedObservation(
                    signal="node_cpu_peak",
                    comparison=Comparison.LTE,
                    threshold=5.0,
                    independent_group="observed_low_cpu",
                    rationale="AI tries to fit the observed value",
                ),
                PredictedObservation(
                    signal="service_cpu_ratio",
                    comparison=Comparison.GTE,
                    threshold=1.6,
                    independent_group="observed_service_cpu",
                    rationale="AI tries to fit the observed value",
                ),
            ],
        }
    )
    proposal = compile_aligned_proposal(
        MechanismDiscoveryResult(generator="test", hypotheses=[hypothesis]),
        CandidateAlignmentResult(
            generator="test",
            decisions=[
                AlignmentDecision(
                    hypothesis_id=hypothesis.hypothesis_id,
                    status=AlignmentStatus.CREATE_NEW,
                    rationale="new",
                )
            ],
        ),
        [],
        rca100_capabilities(),
    )

    assert proposal.candidates == []


def test_alignment_signal_compatibility_corrects_connection_stage_mismatch():
    hypothesis = _hypothesis().model_copy(
        update={
            "hypothesis_id": "connection_delay",
            "mechanism": "database connection acquisition delay",
            "explained_observations": ["connection_duration_ms"],
            "predicted_observations": [
                PredictedObservation(
                    signal="connection_duration_ms",
                    comparison=Comparison.GTE,
                    threshold=100.0,
                    independent_group="connection_stage",
                    rationale="connection time should dominate",
                )
            ],
        }
    )
    proposal = compile_aligned_proposal(
        MechanismDiscoveryResult(generator="test", hypotheses=[hypothesis]),
        CandidateAlignmentResult(
            generator="test",
            decisions=[
                AlignmentDecision(
                    hypothesis_id=hypothesis.hypothesis_id,
                    status=AlignmentStatus.MATCH_EXISTING,
                    candidate_id="slowSQL",
                    rationale="bad semantic match",
                )
            ],
        ),
        list(mechanism_catalog("inventory")),
        rca100_capabilities(),
    )

    assert [item.candidate_id for item in proposal.candidates] == ["dbNetworkLatency"]


def test_alignment_prefers_strong_connection_prediction_over_weak_sql_overlap():
    hypothesis = _hypothesis().model_copy(
        update={
            "hypothesis_id": "pool_wait",
            "mechanism": "database connection acquisition delay",
            "explained_observations": ["connection_duration_ms", "sql_duration_share"],
            "predicted_observations": [
                PredictedObservation(
                    signal="connection_duration_ms",
                    comparison=Comparison.GTE,
                    threshold=5000.0,
                    independent_group="connection_stage",
                    rationale="strong connection delay",
                ),
                PredictedObservation(
                    signal="sql_duration_share",
                    comparison=Comparison.GTE,
                    threshold=0.2,
                    independent_group="sql_stage",
                    rationale="weak SQL overlap",
                ),
            ],
        }
    )
    proposal = compile_aligned_proposal(
        MechanismDiscoveryResult(generator="test", hypotheses=[hypothesis]),
        CandidateAlignmentResult(
            generator="test",
            decisions=[
                AlignmentDecision(
                    hypothesis_id=hypothesis.hypothesis_id,
                    status=AlignmentStatus.MATCH_EXISTING,
                    candidate_id="slowSQL",
                    rationale="semantic model chose the wrong neighbor",
                )
            ],
        ),
        list(mechanism_catalog("inventory")),
        rca100_capabilities(),
    )

    assert [item.candidate_id for item in proposal.candidates] == ["dbNetworkLatency"]
