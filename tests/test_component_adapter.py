from __future__ import annotations

import pytest

from rca_agent.investigation.formalizer import FormalizationCandidate, MechanismFormalizationContext
from rca_agent.investigation.models import InvestigationAlternative
from rca_agent.runner.component_adapter import (
    ComponentAdapterSpec,
    ProductionComponentFactorySpec,
    WorkerBackedDiscoveryAdapter,
    WorkerBackedDiscoveryFormatFormalizerAdapter,
    WorkerBackedMechanismFormalizerAdapter,
    WorkerBackedPlannerAdapter,
)
from rca_agent.runner.component_worker import ComponentWorkerHost, WorkerLifecycle
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


def production_spec(component: str, **config: object) -> ProductionComponentFactorySpec:
    return ProductionComponentFactorySpec(
        component=component,
        provider="offline_stub",
        model="offline-deterministic",
        config=config,
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


def test_child_side_production_factory_constructs_all_three_langchain_adapters() -> None:
    planner = WorkerBackedPlannerAdapter(
        production_spec("planner", response={"plans": []}, partial_response=True), timeout_seconds=1
    )
    assert planner.plan({"context": "planner"}).plans == ()

    discovery = WorkerBackedDiscoveryAdapter(
        production_spec(
            "discovery", response={"mechanisms": []}, partial_response=True,
        ),
        timeout_seconds=1,
    )
    assert discovery.discover({"context": "discovery"}).mechanisms == ()

    formalizer = WorkerBackedMechanismFormalizerAdapter(
        production_spec(
            "formalizer", response={"mechanisms": []}, partial_response=True,
        ),
        timeout_seconds=1,
    )
    assert formalizer.formalize(formalization_context()).mechanisms == ()

    format_formalizer = WorkerBackedDiscoveryFormatFormalizerAdapter(
        production_spec(
            "formalizer",
            formalizer_kind="discovery_format",
            response={"mechanisms": []},
        ),
        timeout_seconds=1,
    )
    from rca_agent.investigation.discovery_format import (
        DiscoveryFormatRepairAnchor,
        DiscoveryFormatRepairContext,
    )

    assert format_formalizer.repair(DiscoveryFormatRepairContext(
        raw_payload={"mechanisms": []},
        anchors=(DiscoveryFormatRepairAnchor(
            source_hypothesis_id="h-runtime-discovery",
            causal_entity_ref="ent-service",
            mechanism="slow SQL execution",
        ),),
    )).mechanisms == ()


def test_production_factory_rejects_live_objects_and_schema_violations() -> None:
    with pytest.raises(ValueError, match="forbidden keys"):
        production_spec("planner", runnable=object())
    adapter = WorkerBackedPlannerAdapter(production_spec("planner", response={"invalid": True}))
    with pytest.raises(RuntimeError, match="ValidationError"):
        adapter.plan({"context": "schema violation"})


def test_production_worker_lifecycle_timeout_partial_cancel_and_reap() -> None:
    timeout_spec = production_spec(
        "planner", sleep_ms=250, response={"plans": []}, partial_response=True
    )
    host = ComponentWorkerHost(
        timeout_spec.worker_spec(),
        ready_timeout_seconds=2,
        request_timeout_seconds=0.03,
        terminate_grace_seconds=0.03,
    )
    host.start()
    with pytest.raises(TimeoutError):
        host.request({"call": {
            "component": "planner",
            "operation": "plan",
            "context": {"late": True},
        }})
    assert host.state is WorkerLifecycle.TIMED_OUT
    assert not host.is_alive
    assert host.wait(0) is not None

    cancel_host = ComponentWorkerHost(timeout_spec.worker_spec(), request_timeout_seconds=0.5)
    cancel_host.start()
    cancel_host.cancel()
    assert cancel_host.state is WorkerLifecycle.CANCELLED
    assert not cancel_host.is_alive

    startup_host = ComponentWorkerHost(
        production_spec("planner", startup_delay_ms=250, response={"plans": []}).worker_spec(),
        ready_timeout_seconds=0.03,
    )
    with pytest.raises(TimeoutError):
        startup_host.start()
    assert startup_host.state is WorkerLifecycle.TIMED_OUT
    assert not startup_host.is_alive


@pytest.mark.parametrize("behavior", ["controlled_exception", "crash_on_request"])
@pytest.mark.parametrize("component", ["discovery", "formalizer", "planner"])
def test_each_production_component_reports_exception_and_crash(
    behavior: str, component: str
) -> None:
    spec = production_spec(component, **{behavior: True})
    if component == "discovery":
        adapter = WorkerBackedDiscoveryAdapter(spec)
        invoke = lambda: adapter.discover({"context": component})
    elif component == "formalizer":
        adapter = WorkerBackedMechanismFormalizerAdapter(spec)
        invoke = lambda: adapter.formalize(formalization_context())
    else:
        adapter = WorkerBackedPlannerAdapter(spec)
        invoke = lambda: adapter.plan({"context": component})
    with pytest.raises(RuntimeError):
        invoke()


@pytest.mark.parametrize("component", ["discovery", "formalizer", "planner"])
def test_each_production_component_rejects_schema_violation(component: str) -> None:
    spec = production_spec(component, response={"invalid": True})
    if component == "discovery":
        adapter = WorkerBackedDiscoveryAdapter(spec)
        invoke = lambda: adapter.discover({"context": component})
    elif component == "formalizer":
        adapter = WorkerBackedMechanismFormalizerAdapter(spec)
        invoke = lambda: adapter.formalize(formalization_context())
    else:
        adapter = WorkerBackedPlannerAdapter(spec)
        invoke = lambda: adapter.plan({"context": component})
    with pytest.raises(RuntimeError) as captured:
        invoke()
    assert str(captured.value) in {"ValidationError", "FormalizerBoundaryError"}


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
