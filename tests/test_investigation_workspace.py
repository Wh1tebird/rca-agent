from __future__ import annotations

from copy import deepcopy

import pytest

from rca_agent.environment.rca100 import Rca100Environment
from rca_agent.investigation import (
    AttemptOutcome,
    CandidateCloseReason,
    CandidateDisposition,
    CandidateSourceKind,
    CandidateStage,
    CandidateSuperseded,
    CandidateTransition,
    CandidateTransitionType,
    CandidateVerificationReachability,
    CapabilityRegistrySnapshot,
    DiscriminationAttempt,
    DiscriminationImportance,
    DiscriminationTask,
    DiscriminationTaskCompleted,
    DiscriminationTaskSelected,
    DiscriminationTaskStatus,
    DiscriminationTaskUnverifiable,
    InvestigationAlternative,
    InvestigationCandidate,
    InvestigationWorkspace,
    ResolvedCapability,
    TransitionBasis,
    WorkspaceCandidateAdded,
    WorkspaceValidationError,
    alternative_semantic_digest,
    candidate_semantic_digest,
    candidate_track_id,
    candidate_transition_id,
    discrimination_attempt_id,
    discrimination_task_id,
    investigation_alternative_id,
)
from rca_agent.models import RequestStopAction
from rca_agent.runner import EpisodeRunner, RecordedReplay, ReplayValidationError
from rca_agent.runner.transcript import EventType, Transcript, TranscriptEvent
from rca_agent.tools.fingerprint import sha256_digest

EPISODE_ID = "ep-" + "1" * 32
SOURCE_EVENT = "2" * 64


def _alternative(
    candidate_id: str, text: str = "slow dependency"
) -> InvestigationAlternative:
    digest = alternative_semantic_digest(candidate_id, text)
    return InvestigationAlternative(
        alternative_id=investigation_alternative_id(candidate_id, digest),
        mechanism_summary=text,
        source_event_refs=(SOURCE_EVENT,),
        semantic_digest=digest,
    )


def _candidate(*, sequence: int = 1, mechanism: str = "thread pool exhaustion"):
    semantic_digest = candidate_semantic_digest(mechanism)
    track_id = candidate_track_id(EPISODE_ID, semantic_digest)
    alternative = _alternative(track_id)
    return InvestigationCandidate(
        candidate_track_id=track_id,
        mechanism_summary=mechanism,
        stage=CandidateStage.TENTATIVE,
        disposition=CandidateDisposition.OPEN,
        alternatives=(alternative,),
        source_kind=CandidateSourceKind.INQUIRY,
        source_event_refs=(SOURCE_EVENT,),
        source_hypothesis_ids=("h1",),
        first_seen_round=1,
        last_updated_round=1,
        semantic_digest=semantic_digest,
        created_sequence=sequence,
        last_transition_sequence=sequence,
    )


def _basis(*, observations: tuple[str, ...] = ()) -> TransitionBasis:
    return TransitionBasis(
        observation_refs=observations,
        source_event_refs=(SOURCE_EVENT,),
        source_round=1,
        reason_code="TEST_BASIS",
        rationale="Deterministic workspace unit-test basis.",
    )


def _transition(
    candidate: InvestigationCandidate,
    transition_type: CandidateTransitionType,
    sequence: int,
    *,
    to_stage: CandidateStage | None = None,
    to_disposition: CandidateDisposition | None = None,
    basis: TransitionBasis | None = None,
    mechanism_summary: str | None = None,
) -> CandidateTransition:
    return CandidateTransition(
        transition_id=candidate_transition_id(
            EPISODE_ID, sequence, candidate.candidate_track_id, transition_type
        ),
        candidate_track_id=candidate.candidate_track_id,
        transition_type=transition_type,
        from_stage=candidate.stage,
        to_stage=to_stage or candidate.stage,
        from_disposition=candidate.disposition,
        to_disposition=to_disposition or candidate.disposition,
        previous_semantic_digest=candidate.semantic_digest,
        resulting_semantic_digest=candidate.semantic_digest,
        mechanism_summary=mechanism_summary,
        basis=basis or _basis(),
        sequence=sequence,
    )


def _workspace_with_candidate() -> tuple[InvestigationWorkspace, InvestigationCandidate]:
    workspace = InvestigationWorkspace(EPISODE_ID)
    candidate = _candidate()
    added = _transition(
        candidate,
        CandidateTransitionType.ADD,
        1,
        to_stage=CandidateStage.TENTATIVE,
        to_disposition=CandidateDisposition.OPEN,
    ).model_copy(update={
        "from_stage": None,
        "from_disposition": None,
        "previous_semantic_digest": None,
        "source_hypothesis_ids_added": candidate.source_hypothesis_ids,
    })
    workspace.apply_candidate_added(
        WorkspaceCandidateAdded(candidate=candidate, transition=added), 1,
    )
    return workspace, candidate


def test_clarify_preserves_semantic_digest_and_rejects_semantic_change():
    workspace, candidate = _workspace_with_candidate()
    clarification = _transition(
        candidate,
        CandidateTransitionType.CLARIFY,
        2,
        mechanism_summary="  THREAD pool exhaustion ",
    )
    workspace.apply_candidate_transition(clarification, 2)
    clarified = workspace.candidate(candidate.candidate_track_id)
    assert clarified is not None
    assert clarified.semantic_digest == candidate.semantic_digest

    changed = _transition(
        clarified,
        CandidateTransitionType.CLARIFY,
        3,
        mechanism_summary="database network latency",
    )
    with pytest.raises(WorkspaceValidationError, match="causal semantics"):
        workspace.apply_candidate_transition(changed, 3)


def test_supersede_is_atomic_and_does_not_inherit_formal_or_core_ids():
    workspace, predecessor = _workspace_with_candidate()
    successor = _candidate(sequence=2, mechanism="database network latency").model_copy(update={
        "stage": CandidateStage.ACTIVE,
        "supersedes_candidate_track_id": predecessor.candidate_track_id,
        "source_kind": CandidateSourceKind.DISCOVERY_REVISION,
    })
    event = CandidateSuperseded(
        transition_id=candidate_transition_id(
            EPISODE_ID,
            2,
            predecessor.candidate_track_id,
            CandidateTransitionType.SUPERSEDE,
        ),
        predecessor_track_id=predecessor.candidate_track_id,
        predecessor_semantic_digest=predecessor.semantic_digest,
        successor=successor,
        basis=_basis(),
        sequence=2,
    )
    workspace.apply_candidate_superseded(event, 2)
    assert workspace.candidate(predecessor.candidate_track_id).stage == CandidateStage.CLOSED
    assert workspace.candidate(successor.candidate_track_id).stage == CandidateStage.ACTIVE
    assert workspace.candidate(successor.candidate_track_id).formal_mechanism_id is None
    assert not workspace.candidate(successor.candidate_track_id).compiled_candidate_ids

    invalid_workspace, invalid_predecessor = _workspace_with_candidate()
    invalid = event.model_copy(update={
        "predecessor_track_id": invalid_predecessor.candidate_track_id,
        "predecessor_semantic_digest": invalid_predecessor.semantic_digest,
        "successor": successor.model_copy(update={"formal_mechanism_id": "mech-" + "4" * 20}),
    })
    before = deepcopy(invalid_workspace.candidates)
    with pytest.raises(WorkspaceValidationError):
        invalid_workspace.apply_candidate_superseded(invalid, 2)
    assert invalid_workspace.candidates == before


def test_reactivate_and_reconsider_change_separate_dimensions():
    workspace, candidate = _workspace_with_candidate()
    retain = _transition(
        candidate, CandidateTransitionType.RETAIN, 2, to_stage=CandidateStage.ACTIVE,
    )
    workspace.apply_candidate_transition(retain, 2)
    candidate = workspace.candidate(candidate.candidate_track_id)
    disfavor = _transition(
        candidate,
        CandidateTransitionType.DISFAVOR,
        3,
        to_disposition=CandidateDisposition.DISFAVORED,
        basis=_basis(observations=("obs-a",)),
    )
    workspace.apply_candidate_transition(disfavor, 3)
    candidate = workspace.candidate(candidate.candidate_track_id)
    defer = _transition(
        candidate, CandidateTransitionType.DEFER, 4, to_stage=CandidateStage.DEFERRED,
    )
    workspace.apply_candidate_transition(defer, 4)
    candidate = workspace.candidate(candidate.candidate_track_id)
    reactivate = _transition(
        candidate, CandidateTransitionType.REACTIVATE, 5, to_stage=CandidateStage.ACTIVE,
    )
    workspace.apply_candidate_transition(reactivate, 5)
    candidate = workspace.candidate(candidate.candidate_track_id)
    assert candidate.stage == CandidateStage.ACTIVE
    assert candidate.disposition == CandidateDisposition.DISFAVORED

    reconsider = _transition(
        candidate,
        CandidateTransitionType.RECONSIDER,
        6,
        to_disposition=CandidateDisposition.OPEN,
        basis=_basis(observations=("obs-b",)),
    )
    workspace.apply_candidate_transition(reconsider, 6)
    candidate = workspace.candidate(candidate.candidate_track_id)
    assert candidate.stage == CandidateStage.ACTIVE
    assert candidate.disposition == CandidateDisposition.OPEN


def test_superseded_close_requires_atomic_event():
    workspace, candidate = _workspace_with_candidate()
    close = _transition(
        candidate,
        CandidateTransitionType.CLOSE,
        2,
        to_stage=CandidateStage.CLOSED,
    ).model_copy(update={"close_reason": CandidateCloseReason.SUPERSEDED})
    with pytest.raises(WorkspaceValidationError, match="dedicated atomic event"):
        workspace.apply_candidate_transition(close, 2)


def test_candidate_reachability_is_partial_when_only_some_primary_tasks_resolve():
    workspace, candidate = _workspace_with_candidate()
    capability = ResolvedCapability(
        capability_ref="cap-" + "3" * 20,
        entity_ref="ent-service-a",
        rule_id="queue_growth",
        signal="queue_depth_ratio",
        independent_group="queue_pressure",
        support_eligible=True,
        query_fingerprint="qry-" + "4" * 16,
        tool_name="compare_metric_windows",
        query_cost=1,
        executable=True,
    )
    registry_fields = {
        "registry_digest": "5" * 64,
        "capabilities": (capability,),
    }
    registry = CapabilityRegistrySnapshot(
        **registry_fields,
        snapshot_digest=sha256_digest({
            "registry_digest": registry_fields["registry_digest"],
            "capabilities": [capability.model_dump(mode="json")],
        }),
    )
    workspace.apply_registry_resolved(registry)

    def task(question: str, sequence: int, options=()):
        alternative = candidate.alternatives[0]
        task_id = discrimination_task_id(
            EPISODE_ID,
            candidate.candidate_track_id,
            alternative.alternative_id,
            question,
            "queue rises",
            "downstream time rises",
        )
        return DiscriminationTask(
            task_id=task_id,
            candidate_track_id=candidate.candidate_track_id,
            source_need_id=f"n{sequence}",
            source_round=1,
            question=question,
            alternative_id=alternative.alternative_id,
            observable_kind="queue_depth_ratio",
            candidate_prediction="queue rises",
            comparator_prediction="downstream time rises",
            importance=DiscriminationImportance.PRIMARY,
            status=DiscriminationTaskStatus.OPEN,
            capability_options=options,
            legacy_choice_ref=("aq-choice" if options else None),
            legacy_query_ref=(capability.query_fingerprint if options else None),
            created_registry_digest=registry.registry_digest,
            last_evaluated_registry_digest=registry.registry_digest,
            unresolved_capability_reason=(None if options else "No executable transport data."),
            created_sequence=sequence,
            last_transition_sequence=sequence,
        )

    resolved = task("Does the queue grow?", 2, (capability,))
    workspace.apply_task_created(resolved, 2)
    attempt = DiscriminationAttempt(
        attempt_id=discrimination_attempt_id(
            EPISODE_ID,
            resolved.task_id,
            capability.capability_ref,
            capability.query_fingerprint,
        ),
        task_id=resolved.task_id,
        capability_ref=capability.capability_ref,
        query_ref=capability.query_fingerprint,
        query_fingerprint=capability.query_fingerprint,
        query_cost=capability.query_cost,
        independent_group=capability.independent_group,
        registry_digest=registry.registry_digest,
        selected_sequence=3,
    )
    workspace.apply_task_selected(
        DiscriminationTaskSelected(task_id=resolved.task_id, attempt=attempt, sequence=3), 3
    )
    workspace.apply_task_completed(DiscriminationTaskCompleted(
        task_id=resolved.task_id,
        attempt_id=attempt.attempt_id,
        to_status=DiscriminationTaskStatus.RESOLVED,
        observation_refs=("obs-queue",),
        tool_status="success",
        outcome=AttemptOutcome.DISTINGUISHES_CANDIDATE,
        reason_code="DISTINGUISHING_OBSERVATION",
        sequence=4,
    ), 4)

    unavailable = task("Did transport latency rise?", 5)
    workspace.apply_task_created(unavailable, 5)
    workspace.apply_task_unverifiable(DiscriminationTaskUnverifiable(
        task_id=unavailable.task_id,
        reason_code="NO_EXACT_CAPABILITY_BINDING",
        reason="No executable transport data.",
        registry_digest=registry.registry_digest,
        sequence=6,
    ), 6)

    snapshot = workspace.snapshot(
        transcript_head_sequence=6,
        transcript_head_digest="6" * 64,
    )
    assert snapshot.reachability[0].reachability == (
        CandidateVerificationReachability.PARTIALLY_ACTIONABLE
    )


def test_runner_projects_inquiry_workspace_without_changing_core(cases_root):
    runner = EpisodeRunner(Rca100Environment.from_case(cases_root, "t007"), inquiry_contract_version=1)
    runner.start()
    runner.record_inquiry_round({
        "round": 1,
        "draft": {
            "hypotheses": [{
                "hypothesis_id": "h1",
                "mechanism": "thread pool exhaustion",
                "alternative": "slow downstream service",
            }],
            "needs": [{
                "need_id": "n1",
                "hypothesis_id": "h1",
                "question": "Which mechanism explains the delay?",
                "expected_if_true": "pool pressure",
                "expected_if_alternative": "downstream duration",
            }],
            "choices": [{"need_id": "n1", "query_ref": "q1"}],
        },
    })
    answer = runner.handle(RequestStopAction(summary="workspace shadow projection"))

    assert answer.investigation_workspace is not None
    assert answer.investigation_workspace.verify_digest()
    assert len(answer.investigation_workspace.candidates) == 1
    assert answer.investigation_workspace.candidates[0].candidate_track_id.startswith("track-")
    assert answer.investigation_workspace.candidates[0].stage == CandidateStage.TENTATIVE
    assert answer.authoritative_decision.status.value == "inconclusive"
    assert RecordedReplay.verify(runner.transcript.events).valid


def test_workspace_disambiguates_reused_inquiry_hypothesis_ids(cases_root):
    runner = EpisodeRunner(Rca100Environment.from_case(cases_root, "t007"), inquiry_contract_version=1)
    runner.start()
    for round_number, mechanism, alternative in (
        (1, "thread pool exhaustion", "slow database"),
        (2, "database latency", "thread pool exhaustion"),
    ):
        runner.record_inquiry_round({
            "round": round_number,
            "draft": {
                "hypotheses": [{
                    "hypothesis_id": "h1",
                    "mechanism": mechanism,
                    "alternative": alternative,
                }],
                "needs": [{
                    "need_id": f"n{round_number}",
                    "hypothesis_id": "h1",
                    "question": "Which causal mechanism fits?",
                    "expected_if_true": mechanism,
                    "expected_if_alternative": alternative,
                }],
                "choices": [{"need_id": f"n{round_number}", "query_ref": "q1"}],
            },
        })
    answer = runner.handle(RequestStopAction(summary="stable candidate IDs"))
    tracks = answer.investigation_workspace.candidates
    assert len(tracks) == 2
    assert tracks[0].source_hypothesis_ids == ("h1",)
    assert tracks[1].source_hypothesis_ids[0].startswith("h1-")
    assert RecordedReplay.verify(runner.transcript.events).valid


def test_candidate_track_survives_a_changed_comparison_alternative(cases_root):
    runner = EpisodeRunner(Rca100Environment.from_case(cases_root, "t007"), inquiry_contract_version=1)
    runner.start()
    for round_number, alternative in (
        (1, "slow database"),
        (2, "slow downstream service"),
    ):
        runner.record_inquiry_round({
            "round": round_number,
            "draft": {
                "hypotheses": [{
                    "hypothesis_id": f"h{round_number}",
                    "mechanism": "thread pool exhaustion",
                    "alternative": alternative,
                }],
                "needs": [{
                    "need_id": f"n{round_number}",
                    "hypothesis_id": f"h{round_number}",
                    "question": "Which causal mechanism fits?",
                    "expected_if_true": "pool pressure",
                    "expected_if_alternative": alternative,
                }],
                "choices": [{"need_id": f"n{round_number}", "query_ref": "q1"}],
            },
        })

    answer = runner.handle(RequestStopAction(summary="stable track across alternatives"))
    tracks = answer.investigation_workspace.candidates
    assert len(tracks) == 1
    assert tracks[0].stage == CandidateStage.ACTIVE
    assert len(tracks[0].alternatives) == 2
    assert tracks[0].source_hypothesis_ids == ("h1", "h2")
    assert RecordedReplay.verify(runner.transcript.events).valid


def test_workspace_shadow_mode_can_be_rolled_back(cases_root):
    runner = EpisodeRunner(
        Rca100Environment.from_case(cases_root, "t007"), inquiry_contract_version=1,
        investigation_workspace_mode="off",
    )
    runner.start()
    runner.record_inquiry_round({
        "round": 1,
        "draft": {
            "hypotheses": [{
                "hypothesis_id": "h1",
                "mechanism": "thread pool exhaustion",
                "alternative": "slow downstream service",
            }],
            "needs": [{
                "need_id": "n1",
                "hypothesis_id": "h1",
                "question": "Which mechanism explains the delay?",
                "expected_if_true": "pool pressure",
                "expected_if_alternative": "downstream duration",
            }],
            "choices": [{"need_id": "n1", "query_ref": "q1"}],
        },
    })
    answer = runner.handle(RequestStopAction(summary="workspace rollback"))
    assert answer.investigation_workspace is None
    assert not any(
        event.event_type.value.startswith("investigation_")
        or event.event_type.value.startswith("candidate_")
        for event in runner.transcript.events
    )
    assert RecordedReplay.verify(runner.transcript.events).valid


def test_replay_rejects_semantically_tampered_workspace_event(cases_root):
    runner = EpisodeRunner(Rca100Environment.from_case(cases_root, "t007"), inquiry_contract_version=1)
    runner.start()
    runner.record_inquiry_round({
        "round": 1,
        "draft": {
            "hypotheses": [{
                "hypothesis_id": "h1",
                "mechanism": "thread pool exhaustion",
                "alternative": "slow downstream service",
            }],
            "needs": [{
                "need_id": "n1",
                "hypothesis_id": "h1",
                "question": "Which mechanism explains the delay?",
                "expected_if_true": "pool pressure",
                "expected_if_alternative": "downstream duration",
            }],
            "choices": [{"need_id": "n1", "query_ref": "q1"}],
        },
    })
    runner.handle(RequestStopAction(summary="workspace replay tamper check"))

    rebuilt = Transcript(runner.episode_id)
    for event in runner.transcript.events:
        payload = deepcopy(event.payload)
        if event.event_type == EventType.INVESTIGATION_CANDIDATE_ADDED:
            payload["candidate"]["mechanism_summary"] = "database network latency"
        rebuilt.append(event.event_type, payload)

    with pytest.raises(ReplayValidationError, match="workspace event"):
        RecordedReplay.verify(rebuilt.events)


def test_replay_rejects_workspace_transition_after_answer_freeze(cases_root):
    runner = EpisodeRunner(Rca100Environment.from_case(cases_root, "t007"), inquiry_contract_version=1)
    runner.start()
    runner.record_inquiry_round({
        "round": 1,
        "draft": {
            "hypotheses": [{
                "hypothesis_id": "h1",
                "mechanism": "thread pool exhaustion",
                "alternative": "slow downstream service",
            }],
            "needs": [{
                "need_id": "n1",
                "hypothesis_id": "h1",
                "question": "Which mechanism explains the delay?",
                "expected_if_true": "pool pressure",
                "expected_if_alternative": "downstream duration",
            }],
            "choices": [{"need_id": "n1", "query_ref": "q1"}],
        },
    })
    runner.handle(RequestStopAction(summary="workspace freeze boundary"))
    candidate = runner.investigation_workspace_snapshot.candidates[0]
    sequence = len(runner.transcript.events) + 1
    transition = _transition(
        candidate,
        CandidateTransitionType.RETAIN,
        sequence,
        to_stage=CandidateStage.ACTIVE,
        basis=TransitionBasis(
            source_event_refs=(candidate.source_event_refs[0],),
            source_round=1,
            reason_code="INVALID_POST_FREEZE_TRANSITION",
            rationale="This otherwise valid transition occurs after answer freeze.",
        ),
    ).model_copy(update={
        "transition_id": candidate_transition_id(
            runner.episode_id,
            sequence,
            candidate.candidate_track_id,
            CandidateTransitionType.RETAIN,
        ),
    })
    runner.transcript.append(EventType.CANDIDATE_TRANSITIONED, transition)

    with pytest.raises(ReplayValidationError, match="after answer freeze"):
        RecordedReplay.verify(runner.transcript.events)


def test_phase2_shadow_records_tasks_matrix_and_registry_without_new_query(cases_root):
    from rca_agent.core.dynamic import DiscoveryPayload
    from rca_agent.core.runtime_registry import rca100_evidence_registry
    from rca_agent.langchain_acquisition import (
        InquiryHandoff,
        InquiryHypothesis,
        InquiryNeed,
        InquiryQueryLink,
        inquiry_candidate_records,
    )
    from rca_agent.models import (
        AnswerArtifact,
        CallToolAction,
        CompareMetricWindowsRequest,
        HypothesisProposal,
        SubmitHypothesisAction,
        TimeWindow,
    )

    class EmptyDiscovery:
        def discover(self, context):
            self.context = context
            return DiscoveryPayload()

    discovery = EmptyDiscovery()
    runner = EpisodeRunner(
        Rca100Environment.from_case(cases_root, "t007"), inquiry_contract_version=1,
        max_rounds=20,
        discovery=discovery,
        evidence_registry=rca100_evidence_registry,
    )
    bootstrap = runner.start()
    horizon = bootstrap.observation_horizon
    incident = TimeWindow(
        start=horizon.start + (horizon.end - horizon.start) / 2,
        end=horizon.end,
    )
    response = runner.handle(CallToolAction(
        tool_name="compare_metric_windows",
        request=CompareMetricWindowsRequest(
            entity_ref=bootstrap.alert_entity.entity_ref,
            signal="request_count",
            incident_window=incident,
        ),
    ))
    assert response.status.value == "success"

    round_record = {
        "round": 1,
        "draft": {
            "hypotheses": [{
                "hypothesis_id": "h1",
                "mechanism": "traffic surge",
                "alternative": "slow downstream service",
            }],
            "needs": [
                {
                    "need_id": "n1",
                    "hypothesis_id": "h1",
                    "question": "Did request volume increase?",
                    "expected_if_true": "request volume rises",
                    "expected_if_alternative": "request volume stays near baseline",
                },
                {
                    "need_id": "n2",
                    "hypothesis_id": "h1",
                    "question": "Did downstream transport time increase?",
                    "expected_if_true": "transport time stays normal",
                    "expected_if_alternative": "transport time rises",
                },
            ],
            "choices": [{"need_id": "n1", "query_ref": "aq-legacy-choice"}],
        },
    }
    runner.record_inquiry_round(round_record)
    runner.handle(SubmitHypothesisAction(proposal=HypothesisProposal(
        hypothesis_id="h-runtime-discovery",
        candidate_id="runtimeUnknown",
        mechanism="unexplained operation latency",
        causal_entity_ref=bootstrap.alert_entity.entity_ref,
        rationale="Formalize the retained Inquiry candidate.",
    )))
    hypothesis = InquiryHypothesis.model_validate(round_record["draft"]["hypotheses"][0])
    handoff = InquiryHandoff(
        revision=1,
        hypotheses=(hypothesis,),
        needs=tuple(
            InquiryNeed.model_validate(item) for item in round_record["draft"]["needs"]
        ),
        query_links=(InquiryQueryLink(
            question="Did request volume increase?",
            hypothesis_id="h1",
            signal="request_count",
            entity_ref=bootstrap.alert_entity.entity_ref,
            query_ref=response.query_ref,
            status=response.status.value,
            observation_refs=tuple(
                item.observation_id for item in response.observations
            ),
        ),),
        candidate_history=inquiry_candidate_records([round_record]),
    )
    runner.discover_candidates(inquiry_handoff=handoff.model_dump(mode="json"))
    answer = runner.handle(RequestStopAction(summary="phase 2 shadow acceptance"))

    workspace = answer.investigation_workspace
    assert workspace.registry_digest is not None
    assert workspace.discrimination_matrix is not None
    assert workspace.discrimination_matrix.verify_digest()
    assert len(workspace.tasks) == 2
    by_need = {item.source_need_id: item for item in workspace.tasks}
    assert by_need["n1"].status.value == "ambiguous"
    assert by_need["n1"].attempts[0].observation_refs
    assert by_need["n2"].status.value == "unverifiable"
    assert workspace.shadow_selections[0].reason_codes == (
        "SHADOW_ONLY",
        "LEGACY_QUERY_EXECUTION_UNCHANGED",
    )
    assert sum(
        event.event_type == EventType.TOOL_REQUEST for event in runner.transcript.events
    ) == 1
    assert RecordedReplay.verify(runner.transcript.events).valid

    legacy_workspace = workspace.model_dump(mode="json", exclude={"snapshot_digest"})
    legacy_workspace.pop("discrimination_matrix")
    legacy_workspace.pop("shadow_selections")
    legacy_workspace["snapshot_digest"] = sha256_digest(legacy_workspace)
    legacy_fields = answer.model_dump(mode="json", exclude={"artifact_digest"})
    legacy_fields["investigation_workspace"] = legacy_workspace
    legacy_answer = AnswerArtifact(
        **legacy_fields,
        artifact_digest=sha256_digest(legacy_fields),
    )
    assert legacy_answer.investigation_workspace.verify_digest()
    assert legacy_answer.verify_digest()
    original_events = runner.transcript.events
    frozen_event = original_events[-1]
    assert frozen_event.event_type == EventType.ANSWER_FROZEN
    legacy_payload = legacy_answer.model_dump(mode="json")
    legacy_payload["investigation_workspace"] = legacy_workspace
    replacement = TranscriptEvent(
        **frozen_event.model_dump(exclude={"payload", "digest"}),
        payload=legacy_payload,
        digest="0" * 64,
    )
    replacement = replacement.model_copy(update={
        "digest": sha256_digest(replacement.digest_payload()),
    })
    assert RecordedReplay.verify((*original_events[:-1], replacement)).valid

    rebuilt = Transcript(runner.episode_id)
    changed = False
    for event in runner.transcript.events:
        payload = deepcopy(event.payload)
        if (
            not changed
            and event.event_type == EventType.DISCRIMINATION_TASK_COMPLETED
            and payload["observation_refs"]
        ):
            payload["observation_refs"] = []
            changed = True
        rebuilt.append(event.event_type, payload)
    assert changed
    with pytest.raises(ReplayValidationError, match="workspace event"):
        RecordedReplay.verify(rebuilt.events)


def test_phase2_active_selects_one_model_offered_capability_before_query(cases_root):
    from rca_agent.core.runtime_registry import rca100_evidence_registry
    from rca_agent.langchain_acquisition import query_options
    from rca_agent.models import CallToolAction
    from rca_agent.tools.fingerprint import query_fingerprint

    runner = EpisodeRunner(
        Rca100Environment.from_case(cases_root, "t007"), inquiry_contract_version=1,
        evidence_registry=rca100_evidence_registry,
        investigation_workspace_mode="active",
    )
    bootstrap = runner.start()
    entity = bootstrap.alert_entity.model_dump(mode="json")
    options = query_options(
        {"bootstrap": bootstrap.model_dump(mode="json")}, [entity]
    )
    offered_ref, offered = next(
        (ref, item)
        for ref, item in options.items()
        if getattr(item["action"].request, "signal", None) == "request_count"
        and item["action"].tool_name.value == "compare_metric_windows"
    )
    action = CallToolAction.model_validate(offered["action"])
    runner.record_inquiry_round({
        "round": 1,
        "draft": {
            "hypotheses": [{
                "hypothesis_id": "h1",
                "mechanism": "traffic surge",
                "alternative": "slow downstream service",
            }],
            "needs": [{
                "need_id": "n1",
                "hypothesis_id": "h1",
                "question": "Did request volume increase?",
                "expected_if_true": "request volume rises",
                "expected_if_alternative": "request volume stays near baseline",
            }],
            "choices": [{"need_id": "n1", "query_ref": offered_ref}],
        },
        "selected_actions": [{
            "need_id": "n1",
            "query_ref": offered_ref,
            "signal": "request_count",
            "entity_ref": entity["entity_ref"],
            "action": action.model_dump(mode="json"),
        }],
    })
    selection_seq = next(
        event.sequence for event in runner.transcript.events
        if event.event_type == EventType.DISCRIMINATION_ACTIVE_SELECTION_RECORDED
    )
    selected = runner.route_active_inquiry_action(action)
    assert selected == action
    runner.handle(selected)
    request_seq = next(
        event.sequence for event in runner.transcript.events
        if event.event_type == EventType.TOOL_REQUEST
    )
    assert selection_seq < request_seq
    task = runner.investigation_workspace_snapshot.tasks[0]
    assert task.status == DiscriminationTaskStatus.AMBIGUOUS
    assert task.attempts[0].query_ref == query_fingerprint(action.tool_name, action.request)
    answer = runner.handle(RequestStopAction(summary="active discrimination acceptance"))
    assert answer.investigation_workspace.active_selections
    assert RecordedReplay.verify(runner.transcript.events).valid


def test_active_repeated_need_can_select_new_query_in_later_round(cases_root):
    from rca_agent.core.runtime_registry import rca100_evidence_registry
    from rca_agent.langchain_acquisition import query_options
    from rca_agent.models import CallToolAction

    runner = EpisodeRunner(
        Rca100Environment.from_case(cases_root, "t007"), inquiry_contract_version=1,
        evidence_registry=rca100_evidence_registry,
        investigation_workspace_mode="active",
    )
    bootstrap = runner.start()
    entity = bootstrap.alert_entity.model_dump(mode="json")
    options = query_options({"bootstrap": bootstrap.model_dump(mode="json")}, [entity])

    def record_round(round_number: int, signal: str) -> CallToolAction:
        offered_ref, offered = next(
            (ref, item) for ref, item in options.items()
            if getattr(item["action"].request, "signal", None) == signal
            and item["action"].tool_name.value == "compare_metric_windows"
        )
        action = CallToolAction.model_validate(offered["action"])
        runner.record_inquiry_round({
            "round": round_number,
            "draft": {
                "hypotheses": [{
                    "hypothesis_id": "h1",
                    "mechanism": "traffic surge",
                    "alternative": "slow downstream service",
                }],
                "needs": [{
                    "need_id": "n1",
                    "hypothesis_id": "h1",
                    "question": "Did traffic or latency increase?",
                    "expected_if_true": "traffic or latency rises",
                    "expected_if_alternative": "traffic and latency stay near baseline",
                }],
                "choices": [{"need_id": "n1", "query_ref": offered_ref}],
            },
            "selected_actions": [{
                "need_id": "n1",
                "query_ref": offered_ref,
                "signal": signal,
                "entity_ref": entity["entity_ref"],
                "action": action.model_dump(mode="json"),
            }],
        })
        return action

    first = record_round(1, "request_count")
    assert runner.route_active_inquiry_action(first) == first
    runner.handle(first)
    second = record_round(2, "request_latency_seconds")
    assert runner.route_active_inquiry_action(second) == second
    runner.handle(second)

    tasks = runner.investigation_workspace_snapshot.tasks
    assert len(tasks) == 2
    assert tasks[1].supersedes_task_id == tasks[0].task_id
    assert tasks[1].source_round == 2
    assert tasks[1].status == DiscriminationTaskStatus.AMBIGUOUS
    runner.handle(RequestStopAction(summary="repeated discriminator acceptance"))
    assert RecordedReplay.verify(runner.transcript.events).valid


@pytest.mark.parametrize("source_id", ["h-runtime-discovery", "h-inquiry-draft"])
def test_phase3_formalizer_preserves_candidate_semantics_and_replays(cases_root, source_id):
    from rca_agent.core.dynamic import (
        AlternativeDraft,
        MechanismClass,
        MechanismDraft,
        VerificationPlanPayload,
    )
    from rca_agent.core.runtime_registry import rca100_evidence_registry
    from rca_agent.core.v3.models import MechanismFamily
    from rca_agent.investigation import (
        FormalizedMechanism,
        MechanismFormalizationPayload,
    )
    from rca_agent.models import (
        CallToolAction,
        GetServiceTraceSummaryRequest,
        HypothesisProposal,
        SubmitHypothesisAction,
    )

    class Formalizer:
        def formalize(self, context):
            candidate = context.candidates[0]
            return MechanismFormalizationPayload(mechanisms=(FormalizedMechanism(
                candidate_track_id=candidate.candidate_track_id,
                source_semantic_digest=candidate.semantic_digest,
                mechanism=MechanismDraft(
                    source_hypothesis_id=source_id,
                    causal_entity_ref=context.entity_graph["nodes"][0]["entity_ref"],
                    mechanism=candidate.mechanism_summary,
                    mechanism_class=MechanismClass.TRAFFIC_SURGE,
                    family=MechanismFamily.APPLICATION,
                    causal_explanation=(
                        "The retained traffic mechanism is formalized without changing meaning."
                    ),
                    alternatives=(AlternativeDraft(
                        alternative_id="alt-downstream",
                        mechanism="slow downstream service",
                        mechanism_class=MechanismClass.DOWNSTREAM_DEPENDENCY,
                    ),),
                ),
            ),))

    class EmptyPlanner:
        def plan(self, context):
            assert context["frozen_mechanisms"]
            return VerificationPlanPayload()

    runner = EpisodeRunner(
        Rca100Environment.from_case(cases_root, "t007"), inquiry_contract_version=1,
        mechanism_formalizer=Formalizer(),
        verification_planner=EmptyPlanner(),
        evidence_registry=rca100_evidence_registry,
    )
    bootstrap = runner.start()
    runner.handle(CallToolAction(
        tool_name="get_service_trace_summary",
        request=GetServiceTraceSummaryRequest(
            entity_ref=bootstrap.alert_entity.entity_ref,
            window=bootstrap.observation_horizon,
            min_duration_ms=100,
            max_traces=2,
        ),
    ))
    runner.record_inquiry_round({
        "round": 1,
        "draft": {
            "hypotheses": [{
                "hypothesis_id": "h1",
                "mechanism": "traffic surge",
                "alternative": "slow downstream service",
            }],
            "needs": [{
                "need_id": "n1",
                "hypothesis_id": "h1",
                "question": "Did traffic rise?",
                "expected_if_true": "traffic rises",
                "expected_if_alternative": "traffic stays stable",
            }],
            "choices": [{"need_id": "n1", "query_ref": "aq-unused"}],
        },
    })
    runner.handle(SubmitHypothesisAction(proposal=HypothesisProposal(
        hypothesis_id="h-runtime-discovery",
        candidate_id="runtimeUnknown",
        mechanism="unexplained operation latency",
        causal_entity_ref=bootstrap.alert_entity.entity_ref,
        rationale="Formalize the retained Inquiry candidate.",
    )))
    result = runner.discover_candidates()
    assert result["status"] == "completed"
    completed = next(
        event for event in runner.transcript.events
        if event.event_type == EventType.MECHANISM_FORMALIZATION_COMPLETED
    )
    assert completed.payload["source_id_rebindings"] == (source_id != "h-runtime-discovery")
    assert completed.payload["payload"]["mechanisms"][0]["mechanism"]["source_hypothesis_id"] == (
        "h-runtime-discovery"
    )
    runner.handle(RequestStopAction(summary="formalizer acceptance"))
    assert RecordedReplay.verify(runner.transcript.events).valid
    if source_id == "h-inquiry-draft":
        from rca_agent.core.dynamic import MechanismDiscoveryPayload

        rebuilt = list(runner.transcript.events)
        changed = False
        for index, event in enumerate(rebuilt):
            if event.event_type == EventType.MECHANISM_FORMALIZATION_COMPLETED:
                changed = True
            if not changed:
                continue
            event_payload = deepcopy(event.payload)
            if event.event_type == EventType.MECHANISM_FORMALIZATION_COMPLETED:
                event_payload["payload"]["mechanisms"][0]["mechanism"][
                    "source_hypothesis_id"
                ] = "h-unissued"
                mechanism_payload = MechanismDiscoveryPayload.model_validate({
                    "mechanisms": [
                        item["mechanism"] for item in event_payload["payload"]["mechanisms"]
                    ],
                })
                event_payload["mechanism_payload_digest"] = sha256_digest(
                    mechanism_payload.model_dump(mode="json")
                )
            replacement = event.model_copy(update={
                "payload": event_payload,
                "previous_digest": rebuilt[index - 1].digest,
                "digest": "0" * 64,
            })
            rebuilt[index] = replacement.model_copy(update={
                "digest": sha256_digest(replacement.digest_payload()),
            })
        with pytest.raises(ReplayValidationError, match="formalizer changed candidate semantics"):
            RecordedReplay.verify(rebuilt)


def test_phase3_formalizer_rejects_semantic_drift():
    from rca_agent.core.dynamic import (
        AlternativeDraft,
        MechanismClass,
        MechanismDraft,
    )
    from rca_agent.core.v3.models import MechanismFamily
    from rca_agent.investigation import (
        FormalizationCandidate,
        FormalizedMechanism,
        MechanismFormalizationContext,
        MechanismFormalizationPayload,
        validate_formalization_semantics,
    )

    candidate = _candidate()
    context = MechanismFormalizationContext(
        alert_summary={"title": "latency"},
        candidates=(FormalizationCandidate(
            candidate_track_id=candidate.candidate_track_id,
            semantic_digest=candidate.semantic_digest,
            mechanism_summary=candidate.mechanism_summary,
            alternatives=candidate.alternatives,
            source_hypothesis_ids=candidate.source_hypothesis_ids,
        ),),
        entity_graph={"nodes": [], "edges": []},
        proposal_ids=("h-runtime",),
    )
    payload = MechanismFormalizationPayload(mechanisms=(FormalizedMechanism(
        candidate_track_id=candidate.candidate_track_id,
        source_semantic_digest=candidate.semantic_digest,
        mechanism=MechanismDraft(
            source_hypothesis_id="h-runtime",
            causal_entity_ref="ent-service",
            mechanism="database query latency",
            mechanism_class=MechanismClass.DATABASE_EXECUTION,
            family=MechanismFamily.DATABASE_DEPENDENCY,
            causal_explanation="A different causal mechanism was substituted by the formalizer.",
            alternatives=(AlternativeDraft(
                alternative_id="alt-network",
                mechanism="network latency",
                mechanism_class=MechanismClass.NETWORK_LATENCY,
            ),),
        ),
    ),))
    with pytest.raises(ValueError, match="changed or detached") as captured:
        validate_formalization_semantics(payload, context)
    assert captured.value.reason_code == "MECHANISM_TEXT_CHANGED"

    unissued = payload.model_copy(update={"mechanisms": (
        payload.mechanisms[0].model_copy(update={"mechanism": (
            payload.mechanisms[0].mechanism.model_copy(update={
                "mechanism": candidate.mechanism_summary,
                "source_hypothesis_id": "h-unknown",
            })
        )}),
    )})
    ambiguous_context = context.model_copy(update={
        "proposal_ids": ("h-runtime", "h-other"),
    })
    with pytest.raises(ValueError, match="changed or detached") as captured:
        validate_formalization_semantics(unissued, ambiguous_context)
    assert captured.value.reason_code == "UNISSUED_SOURCE_HYPOTHESIS"


@pytest.mark.parametrize("outcome", ["skipped", "mechanism", "incomplete"])
def test_formalizer_empty_result_has_one_audited_retry(cases_root, outcome):
    from rca_agent.core.dynamic import (
        AlternativeDraft,
        MechanismClass,
        MechanismDraft,
        VerificationPlanPayload,
    )
    from rca_agent.core.runtime_registry import rca100_evidence_registry
    from rca_agent.core.v3.models import MechanismFamily
    from rca_agent.investigation import (
        FormalizationSkipReason,
        FormalizedMechanism,
        MechanismFormalizationPayload,
        SkippedFormalization,
    )
    from rca_agent.models import (
        CallToolAction,
        GetServiceTraceSummaryRequest,
        HypothesisProposal,
        SubmitHypothesisAction,
    )

    class Formalizer:
        supports_candidate_accounting = True

        def __init__(self):
            self.calls = 0

        def formalize(self, context):
            self.calls += 1
            return MechanismFormalizationPayload()

        def retry_formalize(self, context, previous):
            self.calls += 1
            assert previous == MechanismFormalizationPayload()
            if outcome == "incomplete":
                return MechanismFormalizationPayload()
            if outcome == "mechanism":
                item = context.candidates[0]
                return MechanismFormalizationPayload(mechanisms=(FormalizedMechanism(
                    candidate_track_id=item.candidate_track_id,
                    source_semantic_digest=item.semantic_digest,
                    mechanism=MechanismDraft(
                        source_hypothesis_id=context.proposal_ids[0],
                        causal_entity_ref=context.entity_graph["nodes"][0]["entity_ref"],
                        mechanism=item.mechanism_summary,
                        mechanism_class=MechanismClass.THREAD_POOL_SATURATION,
                        family=MechanismFamily.APPLICATION,
                        causal_explanation="Inventory thread saturation delays requests.",
                        alternatives=(AlternativeDraft(
                            alternative_id="alt-downstream",
                            mechanism="slow downstream service",
                            mechanism_class=MechanismClass.DOWNSTREAM_DEPENDENCY,
                        ),),
                    ),
                ),))
            return MechanismFormalizationPayload(skipped_candidates=tuple(
                SkippedFormalization(
                    candidate_track_id=item.candidate_track_id,
                    reason_code=FormalizationSkipReason.AMBIGUOUS_ENTITY,
                    reason_summary="The causal service cannot be identified from the record.",
                )
                for item in context.candidates
            ))

    formalizer = Formalizer()

    class EmptyPlanner:
        def plan(self, context):
            return VerificationPlanPayload()

    runner = EpisodeRunner(
        Rca100Environment.from_case(cases_root, "t007"), inquiry_contract_version=1,
        mechanism_formalizer=formalizer,
        verification_planner=EmptyPlanner(),
        evidence_registry=rca100_evidence_registry,
    )
    bootstrap = runner.start()
    runner.handle(CallToolAction(
        tool_name="get_service_trace_summary",
        request=GetServiceTraceSummaryRequest(
            entity_ref=bootstrap.alert_entity.entity_ref,
            window=bootstrap.observation_horizon,
            min_duration_ms=100,
            max_traces=2,
        ),
    ))
    runner.record_inquiry_round({
        "round": 1,
        "draft": {
            "hypotheses": [{
                "hypothesis_id": "h1",
                "mechanism": "inventory thread pool exhaustion",
                "alternative": "slow downstream service",
            }],
            "needs": [{
                "need_id": "n1",
                "hypothesis_id": "h1",
                "question": "Did inventory threads saturate?",
                "expected_if_true": "thread demand rises",
                "expected_if_alternative": "downstream time rises",
            }],
            "choices": [{"need_id": "n1", "query_ref": "aq-unused"}],
        },
    })
    runner.handle(SubmitHypothesisAction(proposal=HypothesisProposal(
        hypothesis_id="h-runtime-discovery",
        candidate_id="runtimeUnknown",
        mechanism="unexplained operation latency",
        causal_entity_ref=bootstrap.alert_entity.entity_ref,
        rationale="Formalize the retained Inquiry candidate.",
    )))
    result = runner.discover_candidates()
    assert formalizer.calls == 2
    retries = [event for event in runner.transcript.events
               if event.event_type == EventType.MECHANISM_FORMALIZATION_RETRIED]
    assert len(retries) == 1
    assert retries[0].payload["reason_code"] == "INCOMPLETE_CANDIDATE_ACCOUNTING"
    if outcome != "incomplete":
        assert result["status"] == "completed"
        completed = next(event for event in runner.transcript.events
                         if event.event_type == EventType.MECHANISM_FORMALIZATION_COMPLETED)
        if outcome == "skipped":
            assert len(completed.payload["payload"]["skipped_candidates"]) == 1
        else:
            assert len(completed.payload["payload"]["mechanisms"]) == 1
            assert any(event.event_type == EventType.MECHANISM_DRAFT_FROZEN
                       for event in runner.transcript.events)
    else:
        assert result["status"] != "completed"
        rejected = next(event for event in runner.transcript.events
                        if event.event_type == EventType.MECHANISM_FORMALIZATION_REJECTED)
        assert rejected.payload["reason_code"] == "INCOMPLETE_CANDIDATE_ACCOUNTING"
    runner.handle(RequestStopAction(summary="formalizer retry acceptance"))
    assert RecordedReplay.verify(runner.transcript.events).valid
    if outcome == "skipped":
        rebuilt = list(runner.transcript.events)
        changed = False
        for index, event in enumerate(rebuilt):
            payload = deepcopy(event.payload)
            if event.event_type == EventType.MECHANISM_FORMALIZATION_COMPLETED:
                payload["payload"]["skipped_candidates"][0]["candidate_track_id"] = (
                    "track-" + "f" * 20
                )
                changed = True
            if not changed:
                continue
            replacement = event.model_copy(update={
                "payload": payload,
                "previous_digest": rebuilt[index - 1].digest,
                "digest": "0" * 64,
            })
            rebuilt[index] = replacement.model_copy(update={
                "digest": sha256_digest(replacement.digest_payload()),
            })
        with pytest.raises(ReplayValidationError, match="accounting is incomplete"):
            RecordedReplay.verify(rebuilt)


def test_worker_backed_formalizer_failure_freezes_and_replays(cases_root):
    from rca_agent.core.dynamic import VerificationPlanPayload
    from rca_agent.core.runtime_registry import rca100_evidence_registry
    from rca_agent.models import (
        CallToolAction,
        GetServiceTraceSummaryRequest,
        HypothesisProposal,
        RequestStopAction,
        SubmitHypothesisAction,
    )
    from rca_agent.runner.component_adapter import (
        ProductionComponentFactorySpec,
        WorkerBackedMechanismFormalizerAdapter,
    )

    class EmptyPlanner:
        def plan(self, context):
            return VerificationPlanPayload()

    formalizer = WorkerBackedMechanismFormalizerAdapter(ProductionComponentFactorySpec(
        component="formalizer",
        provider="offline_stub",
        model="offline-deterministic",
        config={"response": {"mechanisms": []}},
    ))
    runner = EpisodeRunner(
        Rca100Environment.from_case(cases_root, "t007"),
        inquiry_contract_version=1,
        mechanism_formalizer=formalizer,
        verification_planner=EmptyPlanner(),
        evidence_registry=rca100_evidence_registry,
    )
    bootstrap = runner.start()
    runner.handle(CallToolAction(
        tool_name="get_service_trace_summary",
        request=GetServiceTraceSummaryRequest(
            entity_ref=bootstrap.alert_entity.entity_ref,
            window=bootstrap.observation_horizon,
            min_duration_ms=100,
            max_traces=2,
        ),
    ))
    runner.record_inquiry_round({
        "round": 1,
        "draft": {
            "hypotheses": [{
                "hypothesis_id": "h1",
                "mechanism": "inventory thread pool exhaustion",
                "alternative": "slow downstream service",
            }],
            "needs": [{
                "need_id": "n1",
                "hypothesis_id": "h1",
                "question": "Did inventory threads saturate?",
                "expected_if_true": "thread demand rises",
                "expected_if_alternative": "downstream time rises",
            }],
            "choices": [{"need_id": "n1", "query_ref": "aq-unused"}],
        },
    })
    runner.handle(SubmitHypothesisAction(proposal=HypothesisProposal(
        hypothesis_id="h-runtime-discovery",
        candidate_id="runtimeUnknown",
        mechanism="unexplained operation latency",
        causal_entity_ref=bootstrap.alert_entity.entity_ref,
        rationale="Formalize the retained Inquiry candidate.",
    )))
    result = runner.discover_candidates()
    assert result["status"] != "completed"
    answer = runner.handle(RequestStopAction(summary="worker formalizer failure"))
    assert answer is not None
    assert RecordedReplay.verify(runner.transcript.events).valid
