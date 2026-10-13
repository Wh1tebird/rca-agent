from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from rca_agent.investigation import (
    MECHANISM_COVERAGE_POLICY_VERSION,
    CandidateDisposition,
    CandidateStage,
    CoverageCandidateInput,
    CoverageCapabilityInput,
    CoverageDomain,
    CoverageFamily,
    CoverageFormalCandidateInput,
    CoverageObservationInput,
    MechanismCoverageInput,
    MechanismCoveragePolicy,
    MechanismCoverageRecord,
    project_formal_coverage_candidates,
)
from rca_agent.models import (
    CallToolAction,
    CapabilityDescriptor,
    DataStatus,
    EntityRef,
    GetMetricSeriesRequest,
    Modality,
    Observation,
    QueryBudgetView,
    QueryCost,
    RequestStopAction,
    ResetResponse,
    TimeWindow,
    ToolName,
    ToolResponse,
    ToolStatus,
)
from rca_agent.models.contract import MetricDescriptor
from rca_agent.runner import EpisodeRunner, EventType, RecordedReplay, ReplayValidationError
from rca_agent.runner.transcript import Transcript
from rca_agent.tools.fingerprint import query_fingerprint, sha256_digest


def _coverage_input(
    *,
    candidates: tuple[CoverageCandidateInput, ...] = (),
    observations: tuple[CoverageObservationInput, ...] = (),
    capabilities: tuple[CoverageCapabilityInput, ...] = (),
) -> MechanismCoverageInput:
    raw = {
        "policy_version": MECHANISM_COVERAGE_POLICY_VERSION,
        "workspace_snapshot_digest": "1" * 64,
        "alert_entity_type": "service",
        "candidates": [item.model_dump(mode="json") for item in candidates],
        "formal_candidates": [],
        "observations": [item.model_dump(mode="json") for item in observations],
        "capabilities": [item.model_dump(mode="json") for item in capabilities],
    }
    return MechanismCoverageInput(**raw, input_digest=sha256_digest(raw))


def _candidate(mechanism_class: str, suffix: str) -> CoverageCandidateInput:
    return CoverageCandidateInput(
        candidate_track_id=f"track-{suffix * 20}",
        mechanism_class=mechanism_class,
        system_layer="runtime",
        stage=CandidateStage.ACTIVE,
        disposition=CandidateDisposition.OPEN,
    )


def _observation(signal: str, suffix: str) -> CoverageObservationInput:
    return CoverageObservationInput(
        observation_ref=f"obs-{suffix * 16}",
        query_ref=f"qry-{suffix * 16}",
        signal=signal,
        entity_ref="ent-service-main",
        data_status=DataStatus.OBSERVED,
    )


def test_database_gap_reports_only_missing_competing_families():
    coverage_input = _coverage_input(
        candidates=(_candidate("database_execution", "a"),),
        observations=(_observation("db_client_span_duration_ms", "b"),),
    )

    assessment = MechanismCoveragePolicy().evaluate(coverage_input)

    assert assessment.relevant_domains == (CoverageDomain.DATABASE,)
    assert assessment.covered_families == (CoverageFamily.DATABASE_EXECUTION,)
    assert len(assessment.gaps) == 1
    assert assessment.gaps[0].missing_families == (
        CoverageFamily.DATABASE_TRANSPORT,
        CoverageFamily.DATABASE_CONNECTION_ACQUISITION,
    )
    assert CoverageFamily.DATABASE_EXECUTION not in assessment.gaps[0].missing_families
    assert assessment.verify_digest()


def test_jvm_gap_is_relevant_but_does_not_claim_causality():
    coverage_input = _coverage_input(
        observations=(_observation("thread_pool_queue_depth", "c"),),
    )

    assessment = MechanismCoveragePolicy().evaluate(coverage_input)

    assert assessment.relevant_domains == (CoverageDomain.JVM,)
    assert assessment.gaps[0].missing_families == (
        CoverageFamily.JVM_THREAD_QUEUE_PRESSURE,
        CoverageFamily.JVM_GC_PRESSURE,
        CoverageFamily.JVM_MEMORY_PRESSURE,
        CoverageFamily.JVM_CPU_LOAD_PRESSURE,
    )
    assert "relevant" in assessment.gaps[0].rationale
    assert "cause" not in assessment.gaps[0].rationale.casefold()


def test_all_live_jvm_families_covered_produces_no_gap():
    coverage_input = _coverage_input(candidates=(
        _candidate("thread_pool_saturation", "1"),
        _candidate("gc_pause", "2"),
        _candidate("memory_pressure", "3"),
        _candidate("cpu_saturation", "4"),
    ))

    assessment = MechanismCoveragePolicy().evaluate(coverage_input)

    assert assessment.relevant_domains == (CoverageDomain.JVM,)
    assert not assessment.gaps


def test_capability_availability_alone_does_not_trigger_a_domain():
    coverage_input = _coverage_input(capabilities=(
        CoverageCapabilityInput(
            tool_name="get_metric_series",
            available=True,
            advertised_signals=("thread_pool_queue_depth", "sql_duration_share"),
        ),
    ))

    assessment = MechanismCoveragePolicy().evaluate(coverage_input)

    assert not assessment.relevant_domains
    assert not assessment.gaps


class _CoverageEnvironment:
    def __init__(self) -> None:
        now = datetime(2026, 1, 1, tzinfo=UTC)
        self.window = TimeWindow(start=now - timedelta(minutes=10), end=now)
        self.bootstrap = ResetResponse(
            event_time=now,
            alert_summary="Synthetic service latency investigation",
            alert_entity=EntityRef(
                entity_ref="ent-service-main",
                display_name="main service",
                entity_type="service",
            ),
            observation_horizon=self.window,
            capabilities=(CapabilityDescriptor(
                tool_name="get_metric_series",
                modality=Modality.METRIC,
                available=True,
                query_cost=1,
                max_window_seconds=3600,
                max_results=100,
                metrics=(MetricDescriptor(
                    entity_type="service",
                    signal="db_client_span_duration_ms",
                    unit="ms",
                    description="Client-observed database stage duration",
                ),),
            ),),
            query_budget=QueryBudgetView(initial=4, remaining=4),
        )

    def reset(self) -> ResetResponse:
        return self.bootstrap

    def get_metric_series(self, request: GetMetricSeriesRequest) -> ToolResponse:
        query_ref = query_fingerprint(ToolName.GET_METRIC_SERIES, request)
        return ToolResponse(
            tool_name=ToolName.GET_METRIC_SERIES,
            query_ref=query_ref,
            status=ToolStatus.SUCCESS,
            observations=(Observation(
                observation_id="obs-1234567890abcdef",
                modality=Modality.METRIC,
                entity_ref=request.entity_ref,
                signal=request.signal,
                value=[12.0, 15.0],
                unit="ms",
                data_status=DataStatus.OBSERVED,
                observed_at=self.bootstrap.event_time,
            ),),
            cost=QueryCost(units=1, budget_before=4, budget_after=3),
            message="bounded synthetic result",
        )


def _run(mode: str) -> EpisodeRunner:
    environment = _CoverageEnvironment()
    runner = EpisodeRunner(environment, mechanism_coverage_mode=mode)
    bootstrap = runner.start()
    runner.handle(CallToolAction(
        tool_name=ToolName.GET_METRIC_SERIES,
        request=GetMetricSeriesRequest(
            entity_ref=bootstrap.alert_entity.entity_ref,
            signal="db_client_span_duration_ms",
            window=bootstrap.observation_horizon,
            max_points=4,
        ),
    ))
    runner.handle(RequestStopAction(summary="bounded investigation complete"))
    return runner


def test_off_and_shadow_have_identical_inference_behavior():
    off = _run("off")
    shadow = _run("shadow")

    def payloads(runner: EpisodeRunner, event_type: EventType) -> list[dict]:
        return [item.payload for item in runner.transcript.events if item.event_type == event_type]

    assert payloads(off, EventType.TOOL_REQUEST) == payloads(
        shadow, EventType.TOOL_REQUEST
    )
    assert payloads(off, EventType.QUERY_COST) == payloads(shadow, EventType.QUERY_COST)
    assert off.answer is not None and shadow.answer is not None
    assert off.answer.authoritative_decision == shadow.answer.authoritative_decision
    assert off.answer.policy_proposals == shadow.answer.policy_proposals
    assert off.answer.investigation_workspace is not None
    assert shadow.answer.investigation_workspace is not None
    assert (
        off.answer.investigation_workspace.candidates
        == shadow.answer.investigation_workspace.candidates
        == ()
    )
    assert payloads(off, EventType.MECHANISM_FORMALIZATION_COMPLETED) == payloads(
        shadow, EventType.MECHANISM_FORMALIZATION_COMPLETED
    )
    assert not payloads(off, EventType.MECHANISM_COVERAGE_SHADOW_RECORDED)
    records = payloads(shadow, EventType.MECHANISM_COVERAGE_SHADOW_RECORDED)
    assert len(records) == 1
    record = MechanismCoverageRecord.model_validate(records[0])
    assert record.assessment.gaps[0].domain == CoverageDomain.DATABASE
    assert RecordedReplay.verify(off.transcript.events).valid
    assert RecordedReplay.verify(shadow.transcript.events).valid
    shadow.begin_evaluation()
    shadow.complete_evaluation("2" * 64)
    assert RecordedReplay.verify(shadow.transcript.events).valid


def test_replay_rejects_rehashed_coverage_tampering():
    runner = _run("shadow")
    rebuilt = Transcript(runner.episode_id)
    for event in runner.transcript.events:
        payload = event.payload
        if event.event_type == EventType.MECHANISM_COVERAGE_SHADOW_RECORDED:
            record = MechanismCoverageRecord.model_validate(payload)
            gap = record.assessment.gaps[0].model_copy(update={
                "rationale": "A modified but schema-valid coverage rationale.",
            })
            assessment = record.assessment.model_copy(update={"gaps": (gap,)})
            assessment = assessment.model_copy(update={
                "assessment_digest": sha256_digest(
                    assessment.model_dump(mode="json", exclude={"assessment_digest"})
                ),
            })
            payload = record.model_copy(update={"assessment": assessment}).model_dump(
                mode="json"
            )
        rebuilt.append(event.event_type, payload)
        if event.event_type == EventType.MECHANISM_COVERAGE_SHADOW_RECORDED:
            break

    with pytest.raises(ReplayValidationError, match="differs from replayed"):
        RecordedReplay.verify(rebuilt.events)


def test_active_coverage_mode_is_not_available():
    with pytest.raises(ValueError, match="must be off or shadow"):
        EpisodeRunner(_CoverageEnvironment(), mechanism_coverage_mode="active")


def test_phase5_fixed_input_stability_including_gap_order_and_digest():
    coverage_input = _coverage_input(
        candidates=(_candidate("database_execution", "a"),),
        observations=(_observation("db_client_span_duration_ms", "b"),),
        capabilities=(CoverageCapabilityInput(
            tool_name="get_metric_series", available=True,
            advertised_signals=("db_client_span_duration_ms",),
        ),),
    )
    outputs = tuple(MechanismCoveragePolicy().evaluate(coverage_input) for _ in range(3))

    assert outputs[0] == outputs[1] == outputs[2]
    assert outputs[0].assessment_digest == outputs[1].assessment_digest
    assert tuple(gap.missing_families for gap in outputs[0].gaps) == (
        (CoverageFamily.DATABASE_TRANSPORT,
         CoverageFamily.DATABASE_CONNECTION_ACQUISITION),
    )


def test_untyped_existing_candidate_without_explicit_mechanism_remains_unclassified():
    coverage_input = _coverage_input(
        candidates=(CoverageCandidateInput(
            candidate_track_id="track-" + "a" * 20,
            mechanism_class=None,
            system_layer=None,
            stage=CandidateStage.ACTIVE,
            disposition=CandidateDisposition.OPEN,
        ),),
        observations=(_observation("db_client_span_duration_ms", "b"),),
    )

    assessment = MechanismCoveragePolicy().evaluate(coverage_input)

    assert assessment.gaps[0].missing_families == (
        CoverageFamily.DATABASE_EXECUTION,
        CoverageFamily.DATABASE_TRANSPORT,
        CoverageFamily.DATABASE_CONNECTION_ACQUISITION,
    )


def test_runner_inquiry_candidate_prevents_duplicate_family_gap():
    runner = EpisodeRunner(_CoverageEnvironment(), mechanism_coverage_mode="shadow",
                           inquiry_contract_version=1)
    bootstrap = runner.start()
    runner.record_inquiry_round({
        "round": 1,
        "draft": {
            "hypotheses": [{
                "hypothesis_id": "h-db",
                "mechanism": "Slow SQL execution in the database",
                "alternative": "Database network transport delay",
            }],
            "needs": [{
                "need_id": "n-db",
                "hypothesis_id": "h-db",
                "question": "Does SQL execution dominate the client span?",
                "expected_if_true": "SQL execution dominates",
                "expected_if_alternative": "Transport dominates",
            }],
            "choices": [{"need_id": "n-db", "query_ref": "q-fixed"}],
        },
    })
    runner.handle(CallToolAction(
        tool_name=ToolName.GET_METRIC_SERIES,
        request=GetMetricSeriesRequest(
            entity_ref=bootstrap.alert_entity.entity_ref,
            signal="db_client_span_duration_ms",
            window=bootstrap.observation_horizon,
            max_points=4,
        ),
    ))
    runner.handle(RequestStopAction(summary="fixed input complete"))

    assert runner.answer is not None
    assert runner.answer.investigation_workspace is not None
    candidate = runner.answer.investigation_workspace.candidates[0]
    assert candidate.mechanism_class is None
    records = [event for event in runner.transcript.events
               if event.event_type == EventType.MECHANISM_COVERAGE_SHADOW_RECORDED]
    assert len(records) == 1
    assessment = MechanismCoverageRecord.model_validate(records[0].payload).assessment
    assert CoverageFamily.DATABASE_EXECUTION in assessment.covered_families
    assert CoverageFamily.DATABASE_EXECUTION not in assessment.gaps[0].missing_families
    assert RecordedReplay.verify(runner.transcript.events).valid


def test_untyped_explicit_mechanism_is_covered_but_ambiguous_text_is_not():
    explicit = CoverageCandidateInput(
        candidate_track_id="track-" + "a" * 20,
        mechanism_summary="Slow SQL execution in the database",
        stage=CandidateStage.ACTIVE,
        disposition=CandidateDisposition.OPEN,
    )
    ambiguous = explicit.model_copy(update={
        "mechanism_summary": "Slow SQL or database network latency",
    })
    observed = (_observation("db_client_span_duration_ms", "b"),)

    assert CoverageFamily.DATABASE_EXECUTION in MechanismCoveragePolicy().evaluate(
        _coverage_input(candidates=(explicit,), observations=observed)
    ).covered_families
    assert not MechanismCoveragePolicy().evaluate(
        _coverage_input(candidates=(ambiguous,), observations=observed)
    ).covered_families


def test_v1_coverage_digest_and_replay_remain_compatible():
    runner = _run("shadow")
    record = next(event for event in runner.transcript.events
                  if event.event_type == EventType.MECHANISM_COVERAGE_SHADOW_RECORDED)
    assert MechanismCoverageRecord.model_validate(record.payload).input.policy_version == (
        MECHANISM_COVERAGE_POLICY_VERSION
    )
    legacy_raw = {
        "policy_version": "mechanism-coverage-v1",
        "workspace_snapshot_digest": "1" * 64,
        "alert_entity_type": "service",
        "candidates": [{
            key: value for key, value in CoverageCandidateInput(
                candidate_track_id="track-" + "a" * 20,
                mechanism_class=None,
                mechanism_summary="Slow SQL execution",
                stage=CandidateStage.ACTIVE,
                disposition=CandidateDisposition.OPEN,
            ).model_dump(mode="json").items() if key != "mechanism_summary"
        }],
        "observations": [],
        "capabilities": [],
    }
    legacy = MechanismCoverageInput(**legacy_raw, input_digest=sha256_digest(legacy_raw))
    assert legacy.verify_digest()
    assert MechanismCoveragePolicy().evaluate(legacy).covered_families == ()


def test_formal_unresolved_candidate_covers_family_and_failed_candidate_does_not():
    formal = CoverageFormalCandidateInput(
        candidate_id="dyn_slow_sql",
        mechanism_id="mech-" + "a" * 20,
        mechanism_class="database_execution",
        mechanism_summary="SQL query execution time is high",
        family="database_dependency",
        state="unresolved",
    )
    raw = _coverage_input(observations=(_observation("sql_shape", "b"),))
    unresolved_payload = raw.model_dump(mode="json", exclude={"input_digest"})
    unresolved_payload["formal_candidates"] = [formal.model_dump(mode="json")]
    unresolved = MechanismCoverageInput(
        **unresolved_payload, input_digest=sha256_digest(unresolved_payload)
    )
    assessment = MechanismCoveragePolicy().evaluate(unresolved)
    assert CoverageFamily.DATABASE_EXECUTION in assessment.covered_families
    assert CoverageFamily.DATABASE_EXECUTION not in assessment.gaps[0].missing_families

    failed_payload = raw.model_dump(mode="json", exclude={"input_digest"})
    failed_payload["formal_candidates"] = [
        formal.model_copy(update={"state": "failed"}).model_dump(mode="json")
    ]
    failed = MechanismCoverageInput(
        **failed_payload, input_digest=sha256_digest(failed_payload)
    )
    assert CoverageFamily.DATABASE_EXECUTION in (
        MechanismCoveragePolicy().evaluate(failed).gaps[0].missing_families
    )


def test_formal_projection_uses_recorded_core_state_not_just_a_draft():
    from types import SimpleNamespace

    mechanism_id = "mech-" + "a" * 20
    events = [
        SimpleNamespace(event_type="mechanism_draft_frozen", payload={"mechanism": {
            "mechanism_id": mechanism_id,
            "mechanism": "SQL query execution time is high",
            "mechanism_class": "database_execution",
            "family": "database_dependency",
        }}),
        SimpleNamespace(event_type="dynamic_candidate_registered", payload={
            "mechanism_id": mechanism_id,
            "compiled": {"candidate": {"candidate_id": "dyn_slow_sql"}},
        }),
    ]
    assert project_formal_coverage_candidates(events) == ()
    events.append(SimpleNamespace(event_type="dynamic_verification", payload={
        "candidates": {"dyn_slow_sql": {"state": "unresolved"}},
    }))
    assert project_formal_coverage_candidates(events)[0].state == "unresolved"


def test_explicit_formal_database_network_candidate_triggers_db_domain():
    formal = CoverageFormalCandidateInput(
        candidate_id="dyn_db_network",
        mechanism_id="mech-" + "a" * 20,
        mechanism_class="network_latency",
        mechanism_summary="Database network latency between application and DB",
        family="network",
        state="unresolved",
    )
    raw = _coverage_input()
    payload = raw.model_dump(mode="json", exclude={"input_digest"})
    payload["formal_candidates"] = [formal.model_dump(mode="json")]
    coverage_input = MechanismCoverageInput(
        **payload, input_digest=sha256_digest(payload)
    )
    assessment = MechanismCoveragePolicy().evaluate(coverage_input)
    assert assessment.relevant_domains == (CoverageDomain.DATABASE,)
    assert assessment.covered_families == (CoverageFamily.DATABASE_TRANSPORT,)
