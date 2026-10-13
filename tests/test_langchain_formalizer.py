"""No-provider check of the opt-in structured Formalizer boundary."""

import pytest
from langchain_core.runnables import RunnableLambda

from rca_agent.core.dynamic import MechanismDiscoveryPayload
from rca_agent.investigation.discovery_format import repair_context_from_raw
from rca_agent.investigation.formalizer import (
    FormalizationCandidate,
    FormalizationSkipReason,
    MechanismFormalizationContext,
    MechanismFormalizationPayload,
    SkippedFormalization,
)
from rca_agent.investigation.models import InvestigationAlternative
from rca_agent.langchain_formalizer import (
    FormalizerBoundaryError,
    LangChainDiscoveryFormatFormalizerAdapter,
    LangChainMechanismFormalizerAdapter,
)


def test_qwen_formalizer_uses_json_schema_and_preserves_track_context():
    seen = {}

    class FakeModel:
        def with_structured_output(self, schema, *, method, include_raw):
            seen.update({"schema": schema, "method": method, "include_raw": include_raw})

            def respond(messages):
                content = messages.messages[-1].content
                seen["content"] = content
                return {"parsed": {"mechanisms": []}, "raw": None}

            return RunnableLambda(respond)

    adapter = LangChainMechanismFormalizerAdapter.from_chat_model(
        FakeModel(), structured_method="json_schema",
    )
    context = MechanismFormalizationContext(
        alert_summary="latency rose",
        candidates=(FormalizationCandidate(
            candidate_track_id="track-" + "a" * 20,
            semantic_digest="b" * 64,
            mechanism_summary="slow SQL execution",
            alternatives=(InvestigationAlternative(
                alternative_id="ialt-" + "c" * 20,
                semantic_digest="d" * 64,
                mechanism_summary="network delay",
            ),),
        ),),
        entity_graph={"nodes": [{"entity_ref": "ent-example"}]},
        proposal_ids=("h-runtime-discovery",),
    )

    result = adapter.formalize(context)

    assert result == MechanismFormalizationPayload()
    assert seen["schema"] is MechanismFormalizationPayload
    assert seen["method"] == "json_schema"
    assert seen["include_raw"] is True
    assert "track-" + "a" * 20 in seen["content"]
    assert "slow SQL execution" in seen["content"]


def test_formalizer_rejects_unparsed_output_with_safe_diagnostic():
    adapter = LangChainMechanismFormalizerAdapter(RunnableLambda(
        lambda _: {"parsed": None, "raw": "private model output", "parsing_error": ValueError()},
    ))
    context = MechanismFormalizationContext(
        alert_summary="latency rose",
        candidates=(FormalizationCandidate(
            candidate_track_id="track-" + "a" * 20,
            semantic_digest="b" * 64,
            mechanism_summary="slow SQL execution",
            alternatives=(InvestigationAlternative(
                alternative_id="ialt-" + "c" * 20,
                semantic_digest="d" * 64,
                mechanism_summary="network delay",
            ),),
        ),),
        entity_graph={"nodes": []},
        proposal_ids=("h-runtime-discovery",),
    )
    with pytest.raises(FormalizerBoundaryError) as captured:
        adapter.formalize(context)
    assert captured.value.reason_code == "STRUCTURED_PARSE_FAILED"
    assert captured.value.field_paths == ()
    assert "private model output" not in str(captured.value)


def test_qwen_formalizer_retry_receives_previous_empty_result():
    seen = []

    class FakeModel:
        def with_structured_output(self, schema, *, method, include_raw):
            def respond(messages):
                content = messages.messages[-1].content
                seen.append((messages.messages[0].content, content))
                return {
                    "parsed": {
                        "mechanisms": [],
                        "skipped_candidates": [{
                            "candidate_track_id": "track-" + "a" * 20,
                            "reason_code": "ambiguous_entity",
                            "reason_summary": "The responsible service is not identifiable.",
                        }],
                    },
                    "raw": None,
                }

            return RunnableLambda(respond)

    adapter = LangChainMechanismFormalizerAdapter.from_chat_model(
        FakeModel(), structured_method="json_schema",
    )
    context = MechanismFormalizationContext(
        alert_summary="latency rose",
        candidates=(FormalizationCandidate(
            candidate_track_id="track-" + "a" * 20,
            semantic_digest="b" * 64,
            mechanism_summary="slow SQL execution",
            alternatives=(InvestigationAlternative(
                alternative_id="ialt-" + "c" * 20,
                semantic_digest="d" * 64,
                mechanism_summary="network delay",
            ),),
        ),),
        entity_graph={"nodes": []},
        proposal_ids=("h-runtime-discovery",),
    )
    result = adapter.retry_formalize(context, MechanismFormalizationPayload())
    assert result.skipped_candidates == (SkippedFormalization(
        candidate_track_id="track-" + "a" * 20,
        reason_code=FormalizationSkipReason.AMBIGUOUS_ENTITY,
        reason_summary="The responsible service is not identifiable.",
    ),)
    assert '"previous_response"' in seen[0][1]
    assert "Missing verification evidence is not a reason" in seen[0][0]


def test_discovery_format_formalizer_uses_structured_schema_without_new_cause():
    seen = {}

    class FakeModel:
        def with_structured_output(self, schema, *, method, include_raw):
            seen.update(schema=schema, method=method, include_raw=include_raw)

            def respond(messages):
                seen["prompt"] = messages.messages[-1].content
                return {"parsed": {
                    "mechanisms": [{
                        "source_hypothesis_id": "h-source",
                        "causal_entity_ref": "ent-service",
                        "mechanism": "slow SQL execution",
                        "mechanism_class": "database_execution",
                        "family": "database_dependency",
                        "causal_explanation": "Complex SQL work dominates database response time.",
                        "alternatives": [{
                            "alternative_id": "alt-network",
                            "mechanism": "database network latency",
                            "mechanism_class": "network_latency",
                        }],
                    }],
                }, "raw": None}

            return RunnableLambda(respond)

    context = repair_context_from_raw({
        "mechanisms": [{
            "source_hypothesis_id": "h-source",
            "causal_entity_ref": "ent-service",
            "mechanism": "slow SQL execution",
            "mechanism_class": "database_execution",
            "family": "database_dependency",
            "causal_explanation": "Complex SQL work dominates database response time.",
            "alternatives": [{"mechanism": "database network latency"}],
        }],
    }, proposal_ids={"h-source"}, issued_entities={"ent-service"})
    assert context is not None
    adapter = LangChainDiscoveryFormatFormalizerAdapter.from_chat_model(
        FakeModel(), structured_method="json_schema",
    )
    result = adapter.repair(context)
    assert isinstance(result, MechanismDiscoveryPayload)
    assert result.mechanisms[0].mechanism == "slow SQL execution"
    assert seen["schema"] is MechanismDiscoveryPayload
    assert seen["method"] == "json_schema"
    assert seen["include_raw"] is True
    assert "slow SQL execution" in seen["prompt"]
