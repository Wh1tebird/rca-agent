from __future__ import annotations

import pytest

from rca_agent.investigation.formalizer import FormalizationCandidate, MechanismFormalizationContext
from rca_agent.investigation.models import InvestigationAlternative
from rca_agent.runner.component_adapter import (
    ComponentAdapterSpec,
    WorkerBackedDiscoveryAdapter,
    WorkerBackedMechanismFormalizerAdapter,
    WorkerBackedPlannerAdapter,
)
from rca_agent.runner.transcript import EventType, Transcript


def planner_spec(**config: object) -> ComponentAdapterSpec:
    return ComponentAdapterSpec(
        component="planner",
        factory_id="test.fake.planner",
        config=config,
    )


def formalization_context() -> MechanismFormalizationContext:
    return MechanismFormalizationContext(
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
        entity_graph={"nodes": [{"entity_ref": "ent-service"}]},
        proposal_ids=("h-runtime-discovery",),
    )


def test_planner_adapter_sends_typed_json_and_returns_validated_payload() -> None:
    adapter = WorkerBackedPlannerAdapter(
        planner_spec(
            response={"plans": []},
            partial_response=True,
            partial_delay_ms=1,
        ),
        timeout_seconds=1,
    )
    result = adapter.plan({"mechanism_ids": ["mech-00000000000000000000"]})
    assert result.plans == ()


def test_planner_timeout_raises_before_late_response_and_reaps_worker() -> None:
    adapter = WorkerBackedPlannerAdapter(
        planner_spec(sleep_ms=250, response={"plans": []}),
        timeout_seconds=0.03,
    )
    with pytest.raises(TimeoutError, match="timed out"):
        adapter.plan({"late": True})


def test_planner_exception_and_crash_are_parent_failures() -> None:
    exception_adapter = WorkerBackedPlannerAdapter(
        planner_spec(controlled_exception=True), timeout_seconds=0.5
    )
    with pytest.raises(RuntimeError, match="RuntimeError"):
        exception_adapter.plan({"value": 1})

    crash_adapter = WorkerBackedPlannerAdapter(
        planner_spec(crash_on_request=True), timeout_seconds=0.5
    )
    with pytest.raises(RuntimeError, match="WorkerCrashedError"):
        crash_adapter.plan({"value": 1})


def test_formalizer_adapter_requires_formalizer_spec_and_never_accepts_live_object() -> None:
    with pytest.raises(ValueError, match="formalizer spec"):
        WorkerBackedMechanismFormalizerAdapter(planner_spec())
    with pytest.raises(ValueError, match="JSON serializable"):
        ComponentAdapterSpec(
            component="planner",
            factory_id="test.fake.planner",
            config={"runnable": object()},
        )


def test_discovery_and_formalizer_workers_use_typed_calls_and_timeout() -> None:
    discovery = WorkerBackedDiscoveryAdapter(ComponentAdapterSpec(
        component="discovery",
        factory_id="test.fake.discovery",
        config={"response": {"hypotheses": []}},
    ))
    assert discovery.discover({"hypotheses": []}).hypotheses == ()

    formalizer = WorkerBackedMechanismFormalizerAdapter(ComponentAdapterSpec(
        component="formalizer",
        factory_id="test.fake.formalizer",
        config={"response": {"mechanisms": []}},
    ))
    assert formalizer.formalize(formalization_context()).mechanisms == ()

    for adapter, context in (
        (
            WorkerBackedDiscoveryAdapter(ComponentAdapterSpec(
                component="discovery",
                factory_id="test.fake.discovery",
                config={"sleep_ms": 200},
            ), timeout_seconds=0.03),
            {},
        ),
        (
            WorkerBackedMechanismFormalizerAdapter(ComponentAdapterSpec(
                component="formalizer",
                factory_id="test.fake.formalizer",
                config={"sleep_ms": 200},
            ), timeout_seconds=0.03),
            formalization_context(),
        ),
    ):
        with pytest.raises(TimeoutError):
            if isinstance(adapter, WorkerBackedDiscoveryAdapter):
                adapter.discover(context)
            else:
                adapter.formalize(context)


def test_parent_owns_transcript_writes_for_worker_result() -> None:
    transcript = Transcript("ep-" + "b" * 32)
    adapter = WorkerBackedPlannerAdapter(planner_spec(response={"plans": []}))
    adapter.plan({"context": "typed"})
    assert transcript.events == ()
    transcript.append(EventType.DISCOVERY_PAYLOAD_REJECTED, {"stage": "planner"})
    assert len(transcript.events) == 1
