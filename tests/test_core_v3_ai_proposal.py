import json

from rca_agent.core.v3.capabilities import rca100_capabilities
from rca_agent.core.v3.catalog import mechanism_catalog
from rca_agent.core.v3.models import Incident, Observation
from rca_agent.core.v3.proposal import (
    ConstrainedAiProposalGenerator,
    DeepSeekProposalGenerator,
    FilteredRuleProposalGenerator,
    HybridProposalGenerator,
    RuleProposalGenerator,
)


def _incident() -> Incident:
    return Incident(
        incident_id="demo",
        alert_title="frontend error",
        alert_entity="frontend",
        window_start="2026-01-01T00:00:00Z",
        window_end="2026-01-01T00:05:00Z",
    )


def test_valid_ai_candidate_passes_schema_and_capability_gate():
    candidate = mechanism_catalog("frontend")[0]
    proposal = ConstrainedAiProposalGenerator().validate_or_fallback(
        {"candidates": [candidate.model_dump()]},
        _incident(),
        rca100_capabilities(),
        [],
    )

    assert proposal.generator == "ai_schema_gated_v1"
    assert proposal.candidates == [candidate]


def test_ai_candidate_without_positive_contract_falls_back_to_rules():
    raw = mechanism_catalog("frontend")[0].model_dump()
    raw["positive_evidence_requirements"] = []
    proposal = ConstrainedAiProposalGenerator().validate_or_fallback(
        {"candidates": [raw]},
        _incident(),
        rca100_capabilities(),
        [],
    )

    assert proposal.generator == "rule_catalog_v1_after_ai_rejection"
    assert proposal.rejected_candidates


def test_ai_candidate_cannot_invent_query_capability():
    raw = mechanism_catalog("frontend")[0].model_dump()
    raw["required_capabilities"] = ["arbitrary_sql"]
    proposal = ConstrainedAiProposalGenerator().validate_or_fallback(
        {"candidates": [raw]},
        _incident(),
        rca100_capabilities(),
        [],
    )

    assert proposal.generator == "rule_catalog_v1_after_ai_rejection"
    assert all(
        item.knowledge_provenance.startswith("human-authored")
        for item in proposal.candidates
    )


def test_ai_candidate_cannot_hide_invented_signal_under_known_capability():
    raw = mechanism_catalog("frontend")[0].model_dump()
    raw["conditions"][0]["signal"] = "invented_cpu_magic"
    proposal = ConstrainedAiProposalGenerator().validate_or_fallback(
        {"candidates": [raw]},
        _incident(),
        rca100_capabilities(),
        [],
    )

    assert proposal.generator == "rule_catalog_v1_after_ai_rejection"
    assert "invalid condition signals" in proposal.rejected_candidates[0]


def test_deepseek_selects_seed_candidate_without_leaking_key_into_body():
    captured = {}

    def transport(request, timeout):
        captured["authorization"] = request.headers["Authorization"]
        captured["body"] = request.data.decode("utf-8")
        captured["timeout"] = timeout
        content = json.dumps({"selected_candidate_ids": ["nodeCpuHigh"], "candidates": []})
        return json.dumps(
            {"choices": [{"message": {"content": content}}]}
        ).encode("utf-8")

    proposal = DeepSeekProposalGenerator(
        api_key="test-secret", transport=transport
    ).propose(
        _incident(),
        rca100_capabilities(),
        [
            Observation(
                signal="node_cpu_peak",
                value=0.96,
                query_ref="private-query-ref",
                observed_entity="frontend",
            )
        ],
    )

    assert proposal.generator == "deepseek_schema_gated_v1"
    assert [item.candidate_id for item in proposal.candidates] == ["nodeCpuHigh"]
    assert captured["authorization"] == "Bearer test-secret"
    assert "test-secret" not in captured["body"]
    assert "private-query-ref" not in captured["body"]


def test_deepseek_failure_falls_back_without_exposing_error_text():
    def transport(request, timeout):
        raise RuntimeError("response accidentally contained a secret")

    proposal = DeepSeekProposalGenerator(
        api_key="test-secret", transport=transport
    ).propose(_incident(), rca100_capabilities(), [])

    assert proposal.generator == "rule_catalog_v1_after_ai_error"
    assert proposal.rejected_candidates == ["AI proposal unavailable: RuntimeError"]
    assert "secret" not in str(proposal.model_dump())


def test_hybrid_ai_can_add_but_not_remove_rule_candidates():
    cpu = mechanism_catalog("frontend")[0]
    rules = RuleProposalGenerator()

    class OnlyCpuAi:
        def propose(self, incident, capabilities, scout_observations):
            from rca_agent.core.v3.models import CandidateProposal

            return CandidateProposal(generator="test_ai", candidates=[cpu])

    rule_candidates = {
        item.candidate_id
        for item in rules.propose(_incident(), rca100_capabilities(), []).candidates
    }
    proposal = HybridProposalGenerator(ai=OnlyCpuAi()).propose(
        _incident(), rca100_capabilities(), []
    )

    assert proposal.generator == "hybrid_deepseek_plus_rule_v1"
    assert rule_candidates <= {item.candidate_id for item in proposal.candidates}
    assert "nodeCpuHigh" in {item.candidate_id for item in proposal.candidates}


def test_filtered_rule_candidate_is_really_absent_from_seed_proposal():
    proposal = FilteredRuleProposalGenerator({"slowSQL"}).propose(
        _incident(), rca100_capabilities(), []
    )

    assert "slowSQL" not in {item.candidate_id for item in proposal.candidates}
