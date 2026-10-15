"""Deterministic reducer for the transcript-backed investigation workspace."""

from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy
from typing import Any

from pydantic import TypeAdapter

from rca_agent.investigation.models import (
    WORKSPACE_REDUCER_VERSION,
    WORKSPACE_REDUCER_VERSION_V2,
    WORKSPACE_SCHEMA_VERSION,
    AttemptOutcome,
    CandidateCloseReason,
    CandidateDisposition,
    CandidateReachabilityView,
    CandidateStage,
    CandidateSuperseded,
    CandidateTransition,
    CandidateTransitionType,
    CandidateVerificationReachability,
    CapabilityRegistrySnapshot,
    DiscriminationActiveSelection,
    DiscriminationImportance,
    DiscriminationMatrix,
    DiscriminationMatrixEntry,
    DiscriminationShadowSelection,
    DiscriminationTask,
    DiscriminationTaskCompleted,
    DiscriminationTaskSelected,
    DiscriminationTaskStatus,
    DiscriminationTaskUnverifiable,
    InvestigationAlternative,
    InvestigationCandidate,
    InvestigationFrontier,
    InvestigationWorkspaceSnapshot,
    WorkspaceCandidateAdded,
    alternative_semantic_digest,
    candidate_semantic_digest,
    candidate_track_id,
    candidate_transition_id,
    discrimination_attempt_id,
    discrimination_task_id,
    investigation_alternative_id,
)
from rca_agent.tools.fingerprint import sha256_digest


class WorkspaceValidationError(ValueError):
    pass


def _normalized_task_text(value: str) -> str:
    return " ".join(value.casefold().split())


class InvestigationWorkspace:
    """Runtime aggregate rebuilt from Transcript events; never a Core authority."""

    def __init__(self, episode_id: str, *, reducer_version: str = WORKSPACE_REDUCER_VERSION) -> None:
        self.episode_id = episode_id
        self.schema_version = WORKSPACE_SCHEMA_VERSION
        if reducer_version not in {WORKSPACE_REDUCER_VERSION, WORKSPACE_REDUCER_VERSION_V2}:
            raise ValueError("unsupported workspace reducer version")
        self.reducer_version = reducer_version
        self._candidates: dict[str, InvestigationCandidate] = {}
        self._tasks: dict[str, DiscriminationTask] = {}
        self._registries: dict[str, CapabilityRegistrySnapshot] = {}
        self._latest_registry_digest: str | None = None
        self._shadow_selections: list[DiscriminationShadowSelection] = []
        self._active_selections: list[DiscriminationActiveSelection] = []
        self._inquiry_loop_state: Any | None = None

    @property
    def candidates(self) -> tuple[InvestigationCandidate, ...]:
        return tuple(sorted(self._candidates.values(), key=lambda item: item.created_sequence))

    @property
    def tasks(self) -> tuple[DiscriminationTask, ...]:
        return tuple(sorted(self._tasks.values(), key=lambda item: item.created_sequence))

    def candidate(self, candidate_track_id: str) -> InvestigationCandidate | None:
        return self._candidates.get(candidate_track_id)

    def task(self, task_id: str) -> DiscriminationTask | None:
        return self._tasks.get(task_id)

    def candidate_by_semantic_digest(self, digest: str) -> InvestigationCandidate | None:
        return next((item for item in self._candidates.values()
                     if item.semantic_digest == digest and item.stage != CandidateStage.CLOSED), None)

    def apply_registry_resolved(self, snapshot: CapabilityRegistrySnapshot) -> None:
        if not snapshot.verify_digest():
            raise WorkspaceValidationError("capability Registry snapshot digest is invalid")
        previous = self._registries.get(snapshot.registry_digest)
        if previous is not None and previous != snapshot:
            raise WorkspaceValidationError("capability Registry digest was reused")
        refs = [item.capability_ref for item in snapshot.capabilities]
        if len(refs) != len(set(refs)):
            raise WorkspaceValidationError("capability Registry has duplicate handles")
        self._registries[snapshot.registry_digest] = snapshot
        self._latest_registry_digest = snapshot.registry_digest

    def apply_task_created(self, task: DiscriminationTask, sequence: int) -> None:
        if task.task_id in self._tasks:
            raise WorkspaceValidationError("discrimination task already exists")
        candidate = self._candidates.get(task.candidate_track_id)
        if candidate is None:
            raise WorkspaceValidationError("discrimination task references an unknown candidate")
        if task.created_registry_digest not in self._registries:
            raise WorkspaceValidationError("discrimination task Registry is unknown")
        if task.created_sequence != sequence or task.last_transition_sequence != sequence:
            raise WorkspaceValidationError("discrimination task sequence differs from event")
        if task.status != DiscriminationTaskStatus.OPEN or task.attempts:
            raise WorkspaceValidationError("new discrimination task must be open")
        alternative = next(
            (item for item in candidate.alternatives if item.alternative_id == task.alternative_id),
            None,
        )
        if alternative is None or task.compared_candidate_track_id is not None:
            raise WorkspaceValidationError("discrimination task comparator is not bound")
        predecessor = self._tasks.get(task.supersedes_task_id or "")
        if task.supersedes_task_id is not None and (
            predecessor is None
            or predecessor.candidate_track_id != task.candidate_track_id
            or predecessor.alternative_id != task.alternative_id
            or _normalized_task_text(predecessor.question) != _normalized_task_text(task.question)
            or _normalized_task_text(predecessor.candidate_prediction)
            != _normalized_task_text(task.candidate_prediction)
            or _normalized_task_text(predecessor.comparator_prediction)
            != _normalized_task_text(task.comparator_prediction)
            or predecessor.source_round >= task.source_round
        ):
            raise WorkspaceValidationError("recurring discrimination task predecessor is invalid")
        if task.task_id != discrimination_task_id(
            self.episode_id,
            candidate.candidate_track_id,
            alternative.alternative_id,
            task.question,
            task.candidate_prediction,
            task.comparator_prediction,
            recurrence_round=(task.source_round if predecessor is not None else None),
            query_fingerprint=(task.legacy_query_ref if task.query_strategy_version == 2 else None),
            binding_id=task.source_binding_id,
        ):
            raise WorkspaceValidationError("discrimination task ID is not deterministic")
        registry = self._registries[task.created_registry_digest]
        known = {item.capability_ref: item for item in registry.capabilities}
        if any(known.get(item.capability_ref) != item for item in task.capability_options):
            raise WorkspaceValidationError("discrimination task capability is not in Registry")
        if task.capability_options and task.unresolved_capability_reason is not None:
            raise WorkspaceValidationError("resolved task carries an unavailable reason")
        self._tasks[task.task_id] = task

    def apply_task_selected(self, payload: DiscriminationTaskSelected, sequence: int) -> None:
        task = self._tasks.get(payload.task_id)
        if task is None or task.status != DiscriminationTaskStatus.OPEN:
            raise WorkspaceValidationError("only an open discrimination task can be selected")
        attempt = payload.attempt
        if payload.sequence != sequence or attempt.selected_sequence != sequence:
            raise WorkspaceValidationError("task selection sequence differs from event")
        if attempt.task_id != task.task_id or attempt.completed_sequence is not None:
            raise WorkspaceValidationError("task selection attempt is invalid")
        option = next(
            (item for item in task.capability_options
             if item.capability_ref == attempt.capability_ref),
            None,
        )
        if option is None or not option.executable:
            raise WorkspaceValidationError("task selection capability is not executable")
        if (
            attempt.query_fingerprint != option.query_fingerprint
            or attempt.query_ref != task.legacy_query_ref
            or attempt.query_cost != option.query_cost
            or attempt.independent_group != option.independent_group
            or attempt.registry_digest != task.last_evaluated_registry_digest
            or attempt.observation_refs
            or attempt.tool_status is not None
            or attempt.outcome is not None
            or attempt.attempt_id != discrimination_attempt_id(
                self.episode_id,
                task.task_id,
                attempt.capability_ref,
                attempt.query_fingerprint,
            )
        ):
            raise WorkspaceValidationError("task selection attempt differs from Matrix option")
        self._tasks[task.task_id] = task.model_copy(update={
            "status": DiscriminationTaskStatus.SELECTED,
            "attempts": (*task.attempts, attempt),
            "selected_attempt_id": attempt.attempt_id,
            "last_transition_sequence": sequence,
        })

    def apply_active_selection(
        self, selection: DiscriminationActiveSelection, sequence: int,
    ) -> None:
        matrix = self.discrimination_matrix()
        entry = next((
            item for item in matrix.entries
            if item.task_id == selection.selected_task_id
            and item.capability_ref == selection.selected_capability_ref
        ), None)
        expected_id = "active-selection-" + sha256_digest({
            "episode_id": self.episode_id,
            "sequence": sequence,
            "matrix_digest": matrix.matrix_digest,
        })[:20]
        if (
            selection.sequence != sequence
            or selection.selection_id != expected_id
            or selection.matrix_digest != matrix.matrix_digest
            or entry is None
            or entry.query_fingerprint != selection.selected_query_fingerprint
            or selection.selected_query_fingerprint
            not in selection.offered_query_fingerprints
            or not set(selection.suppressed_query_fingerprints)
            <= set(selection.offered_query_fingerprints)
            or selection.selected_query_fingerprint
            in selection.suppressed_query_fingerprints
        ):
            raise WorkspaceValidationError("active discrimination selection is invalid")
        if selection.strategy_version == 2:
            pending = {
                option.query_fingerprint for task in self.tasks
                if task.scheduling is not None and task.status == DiscriminationTaskStatus.OPEN
                for option in task.capability_options
            }
            eligible = [item for item in matrix.entries if item.query_fingerprint in pending]
            if (not eligible or entry != eligible[0]
                    or selection.suppressed_query_fingerprints
                    or set(selection.pending_query_fingerprints or ())
                    != pending - {selection.selected_query_fingerprint}):
                raise WorkspaceValidationError("active query queue or utility ranking differs")
        self._active_selections.append(selection)

    def apply_query_cancelled(self, payload: dict, sequence: int) -> None:
        task = self.task(payload.get("task_id", ""))
        if (task is None or task.scheduling is None or task.status != DiscriminationTaskStatus.OPEN
                or payload.get("sequence") != sequence
                or payload.get("query_fingerprint") != task.legacy_query_ref
                or payload.get("reason_code") not in {
                    "DUPLICATE_EXECUTED_QUERY", "QUERY_BUDGET_UNAVAILABLE", "EPISODE_ENDED",
                }):
            raise WorkspaceValidationError("invalid query cancellation")
        self._tasks[task.task_id] = task.model_copy(update={
            "status": DiscriminationTaskStatus.CANCELLED,
            "unresolved_capability_reason": payload["reason_code"],
            "last_transition_sequence": sequence,
        })

    def apply_task_completed(self, payload: DiscriminationTaskCompleted, sequence: int) -> None:
        task = self._tasks.get(payload.task_id)
        if task is None or task.status != DiscriminationTaskStatus.SELECTED:
            raise WorkspaceValidationError("only a selected discrimination task can complete")
        if payload.sequence != sequence or task.selected_attempt_id != payload.attempt_id:
            raise WorkspaceValidationError("task completion sequence or attempt differs")
        attempt = next(
            (item for item in task.attempts if item.attempt_id == payload.attempt_id), None
        )
        if attempt is None or attempt.completed_sequence is not None:
            raise WorkspaceValidationError("task completion attempt is invalid")
        if payload.to_status == DiscriminationTaskStatus.RESOLVED and payload.outcome not in {
            AttemptOutcome.DISTINGUISHES_CANDIDATE,
            AttemptOutcome.DISTINGUISHES_COMPARATOR,
        }:
            raise WorkspaceValidationError("resolved task lacks a distinguishing outcome")
        if (
            payload.to_status == DiscriminationTaskStatus.AMBIGUOUS
            and payload.outcome != AttemptOutcome.AMBIGUOUS
        ):
            raise WorkspaceValidationError("ambiguous task has a non-ambiguous outcome")
        if (
            payload.to_status == DiscriminationTaskStatus.UNVERIFIABLE
            and payload.outcome not in {
                AttemptOutcome.NO_DATA,
                AttemptOutcome.QUERY_FAILED,
            }
        ):
            raise WorkspaceValidationError("unverifiable task has an invalid outcome")
        completed = attempt.model_copy(update={
            "observation_refs": payload.observation_refs,
            "tool_status": payload.tool_status,
            "outcome": payload.outcome,
            "completed_sequence": sequence,
        })
        self._tasks[task.task_id] = task.model_copy(update={
            "status": payload.to_status,
            "attempts": tuple(
                completed if item.attempt_id == completed.attempt_id else item
                for item in task.attempts
            ),
            "unresolved_capability_reason": (
                payload.reason_code
                if payload.to_status == DiscriminationTaskStatus.UNVERIFIABLE
                else task.unresolved_capability_reason
            ),
            "last_transition_sequence": sequence,
        })

    def apply_task_unverifiable(
        self, payload: DiscriminationTaskUnverifiable, sequence: int,
    ) -> None:
        task = self._tasks.get(payload.task_id)
        if task is None or task.status != DiscriminationTaskStatus.OPEN:
            raise WorkspaceValidationError("only an open task can be directly unverifiable")
        if (
            payload.sequence != sequence
            or payload.registry_digest != task.last_evaluated_registry_digest
            or task.capability_options
        ):
            raise WorkspaceValidationError("unverifiable task still has executable capabilities")
        self._tasks[task.task_id] = task.model_copy(update={
            "status": DiscriminationTaskStatus.UNVERIFIABLE,
            "unresolved_capability_reason": payload.reason,
            "last_transition_sequence": sequence,
        })

    def discrimination_matrix(self) -> DiscriminationMatrix:
        raw = []
        for task in self.tasks:
            if task.scheduling is not None and task.status != DiscriminationTaskStatus.OPEN:
                continue
            for option in task.capability_options:
                if not option.executable:
                    continue
                score = 100 - 5 * option.query_cost
                reasons = (
                    "EXPLICIT_CONTRASTING_PREDICTIONS",
                    "EXACT_EXECUTABLE_CAPABILITY_BINDING",
                    "LOWER_QUERY_COST_PREFERRED",
                )
                if task.scheduling is not None:
                    from rca_agent.investigation.query_strategy import task_score

                    score, reasons = task_score(task, option, self.tasks)
                raw.append((
                    -score,
                    task.importance != DiscriminationImportance.PRIMARY,
                    task.task_id,
                    option.capability_ref,
                    task,
                    option,
                    score,
                    reasons,
                ))
        raw.sort(key=lambda item: item[:4])
        entries = []
        for index, item in enumerate(raw, start=1):
            task, option, score, reasons = item[4:]
            entries.append(DiscriminationMatrixEntry(
                rank=index,
                task_id=task.task_id,
                candidate_track_id=task.candidate_track_id,
                capability_ref=option.capability_ref,
                query_fingerprint=option.query_fingerprint,
                query_cost=option.query_cost,
                independent_group=option.independent_group,
                score=score,
                reason_codes=reasons,
                legacy_selected=task.legacy_choice_ref is not None,
            ))
        entries = tuple(entries)
        fields = {
            "registry_digest": self._latest_registry_digest,
            "entries": entries,
        }
        return DiscriminationMatrix(
            **fields,
            matrix_digest=sha256_digest(
                TypeAdapter(dict[str, Any]).dump_python(fields, mode="json")
            ),
        )

    def apply_shadow_selection(
        self, selection: DiscriminationShadowSelection, sequence: int,
    ) -> None:
        matrix = self.discrimination_matrix()
        top = matrix.entries[0] if matrix.entries else None
        if (
            selection.sequence != sequence
            or selection.matrix_digest != matrix.matrix_digest
            or selection.evaluated_task_ids != tuple(item.task_id for item in self.tasks)
            or selection.selected_task_id != (top.task_id if top else None)
            or selection.selected_capability_ref != (top.capability_ref if top else None)
            or selection.matches_legacy_first != (
                None
                if top is None or selection.legacy_first_task_id is None
                else top.task_id == selection.legacy_first_task_id
            )
            or selection.selection_id != "selection-" + sha256_digest({
                "episode_id": self.episode_id,
                "sequence": sequence,
                "matrix_digest": matrix.matrix_digest,
            })[:20]
        ):
            raise WorkspaceValidationError("shadow discrimination selection is invalid")
        self._shadow_selections.append(selection)

    def apply_candidate_added(self, payload: WorkspaceCandidateAdded, sequence: int) -> None:
        candidate = payload.candidate
        transition = payload.transition
        if candidate.candidate_track_id in self._candidates:
            raise WorkspaceValidationError("candidate track already exists")
        if candidate.semantic_digest != candidate_semantic_digest(
            candidate.mechanism_summary,
            mechanism_class=candidate.mechanism_class,
            causal_entity_ref=candidate.causal_entity_ref,
        ):
            raise WorkspaceValidationError("candidate semantic digest is invalid")
        if candidate.candidate_track_id != candidate_track_id(
            self.episode_id, candidate.semantic_digest
        ):
            raise WorkspaceValidationError("candidate track ID is not deterministic")
        if candidate.created_sequence != sequence or candidate.last_transition_sequence != sequence:
            raise WorkspaceValidationError("candidate creation sequence differs from event")
        if not set(candidate.source_event_refs) <= set(transition.basis.source_event_refs):
            raise WorkspaceValidationError("candidate ADD has unbound source events")
        self._validate_alternatives(
            candidate,
            candidate.alternatives,
            allowed_source_refs=set(candidate.source_event_refs),
        )
        if (
            transition.sequence != sequence
            or transition.candidate_track_id != candidate.candidate_track_id
            or transition.transition_type != CandidateTransitionType.ADD
            or transition.from_stage is not None
            or transition.from_disposition is not None
            or transition.to_stage != CandidateStage.TENTATIVE
            or transition.to_disposition != CandidateDisposition.OPEN
            or transition.previous_semantic_digest is not None
            or transition.resulting_semantic_digest != candidate.semantic_digest
            or candidate.stage != CandidateStage.TENTATIVE
            or candidate.disposition != CandidateDisposition.OPEN
            or transition.source_hypothesis_ids_added != candidate.source_hypothesis_ids
        ):
            raise WorkspaceValidationError("invalid candidate ADD transition")
        self._validate_transition_id(transition)
        self._candidates[candidate.candidate_track_id] = candidate

    def apply_candidate_transition(self, transition: CandidateTransition, sequence: int) -> None:
        candidate = self._candidates.get(transition.candidate_track_id)
        if candidate is None:
            raise WorkspaceValidationError("candidate transition references unknown track")
        if candidate.stage == CandidateStage.CLOSED:
            raise WorkspaceValidationError("closed candidate cannot transition")
        if transition.sequence != sequence:
            raise WorkspaceValidationError("candidate transition sequence differs from event")
        self._validate_transition_id(transition)
        if (
            transition.from_stage != candidate.stage
            or transition.from_disposition != candidate.disposition
            or transition.previous_semantic_digest != candidate.semantic_digest
        ):
            raise WorkspaceValidationError("candidate transition does not match current state")

        expected_stage = candidate.stage
        expected_disposition = candidate.disposition
        transition_type = transition.transition_type
        updates: dict[str, Any] = {}

        if transition_type == CandidateTransitionType.RETAIN:
            if candidate.stage not in {CandidateStage.TENTATIVE, CandidateStage.ACTIVE}:
                raise WorkspaceValidationError("RETAIN requires tentative or active candidate")
            expected_stage = CandidateStage.ACTIVE
        elif transition_type == CandidateTransitionType.CLARIFY:
            if transition.resulting_semantic_digest != candidate.semantic_digest:
                raise WorkspaceValidationError("CLARIFY changed semantic digest")
            if transition.mechanism_summary is not None:
                if candidate_semantic_digest(
                    transition.mechanism_summary,
                    mechanism_class=candidate.mechanism_class,
                    causal_entity_ref=candidate.causal_entity_ref,
                ) != candidate.semantic_digest:
                    raise WorkspaceValidationError("CLARIFY changed causal semantics")
                updates["mechanism_summary"] = transition.mechanism_summary
        elif transition_type == CandidateTransitionType.DEFER:
            if candidate.stage not in {CandidateStage.TENTATIVE, CandidateStage.ACTIVE}:
                raise WorkspaceValidationError("DEFER requires tentative or active candidate")
            expected_stage = CandidateStage.DEFERRED
        elif transition_type == CandidateTransitionType.REACTIVATE:
            if candidate.stage != CandidateStage.DEFERRED:
                raise WorkspaceValidationError("REACTIVATE requires deferred candidate")
            expected_stage = CandidateStage.ACTIVE
        elif transition_type == CandidateTransitionType.FORMALIZE:
            if candidate.stage != CandidateStage.ACTIVE or transition.formal_mechanism_id is None:
                raise WorkspaceValidationError("FORMALIZE requires active candidate and mechanism")
            if candidate.formal_mechanism_id not in {None, transition.formal_mechanism_id}:
                raise WorkspaceValidationError("candidate already links another formal mechanism")
            updates["formal_mechanism_id"] = transition.formal_mechanism_id
            updates["compiled_candidate_ids"] = tuple(dict.fromkeys((
                *candidate.compiled_candidate_ids,
                *transition.compiled_candidate_ids_added,
            )))
        elif transition_type == CandidateTransitionType.DISFAVOR:
            if not transition.basis.observation_refs:
                raise WorkspaceValidationError("DISFAVOR requires issued observations")
            expected_disposition = CandidateDisposition.DISFAVORED
        elif transition_type == CandidateTransitionType.RECONSIDER:
            if candidate.disposition != CandidateDisposition.DISFAVORED:
                raise WorkspaceValidationError("RECONSIDER requires disfavored candidate")
            if not transition.basis.observation_refs:
                raise WorkspaceValidationError("RECONSIDER requires new observations")
            expected_disposition = CandidateDisposition.OPEN
        elif transition_type == CandidateTransitionType.CLOSE:
            if transition.close_reason is None:
                raise WorkspaceValidationError("CLOSE requires a reason code")
            if transition.close_reason == CandidateCloseReason.SUPERSEDED:
                raise WorkspaceValidationError(
                    "SUPERSEDED closure requires the dedicated atomic event"
                )
            if (
                transition.close_reason == CandidateCloseReason.DISFAVORED_BY_EVIDENCE
                and candidate.disposition != CandidateDisposition.DISFAVORED
            ):
                raise WorkspaceValidationError("evidence closure requires prior DISFAVOR")
            expected_stage = CandidateStage.CLOSED
        elif transition_type in {
            CandidateTransitionType.ADD,
            CandidateTransitionType.SUPERSEDE,
        }:
            raise WorkspaceValidationError("transition requires its dedicated atomic event")

        if transition.alternatives_added:
            if transition_type not in {
                CandidateTransitionType.RETAIN,
                CandidateTransitionType.CLARIFY,
            }:
                raise WorkspaceValidationError(
                    "alternatives can only be added while retaining or clarifying a candidate"
                )
            self._validate_alternatives(
                candidate,
                transition.alternatives_added,
                allowed_source_refs=set(transition.basis.source_event_refs),
            )
            existing_digests = {item.semantic_digest for item in candidate.alternatives}
            if any(
                item.semantic_digest in existing_digests
                for item in transition.alternatives_added
            ):
                raise WorkspaceValidationError("candidate transition repeats an alternative")
            if len(candidate.alternatives) + len(transition.alternatives_added) > 12:
                raise WorkspaceValidationError("candidate exceeds the alternative limit")
            updates["alternatives"] = (
                *candidate.alternatives,
                *transition.alternatives_added,
            )

        if (
            transition.to_stage != expected_stage
            or transition.to_disposition != expected_disposition
            or transition.resulting_semantic_digest != candidate.semantic_digest
        ):
            raise WorkspaceValidationError("candidate transition has invalid result state")

        updates.update({
            "stage": expected_stage,
            "disposition": expected_disposition,
            "source_event_refs": tuple(dict.fromkeys((
                *candidate.source_event_refs,
                *transition.basis.source_event_refs,
            ))),
            "source_hypothesis_ids": tuple(dict.fromkeys((
                *candidate.source_hypothesis_ids,
                *transition.source_hypothesis_ids_added,
            ))),
            "last_updated_round": transition.basis.source_round or candidate.last_updated_round,
            "last_transition_sequence": sequence,
        })
        self._candidates[candidate.candidate_track_id] = candidate.model_copy(update=updates)

    def apply_candidate_superseded(self, payload: CandidateSuperseded, sequence: int) -> None:
        predecessor = self._candidates.get(payload.predecessor_track_id)
        successor = payload.successor
        if predecessor is None or predecessor.stage == CandidateStage.CLOSED:
            raise WorkspaceValidationError("SUPERSEDE requires an open predecessor")
        if successor.candidate_track_id in self._candidates:
            raise WorkspaceValidationError("SUPERSEDE successor already exists")
        if payload.sequence != sequence or successor.created_sequence != sequence:
            raise WorkspaceValidationError("SUPERSEDE sequence differs from event")
        if predecessor.semantic_digest != payload.predecessor_semantic_digest:
            raise WorkspaceValidationError("SUPERSEDE predecessor digest mismatch")
        if successor.semantic_digest == predecessor.semantic_digest:
            raise WorkspaceValidationError("SUPERSEDE requires changed causal semantics")
        if successor.semantic_digest != candidate_semantic_digest(
            successor.mechanism_summary,
            mechanism_class=successor.mechanism_class,
            causal_entity_ref=successor.causal_entity_ref,
        ):
            raise WorkspaceValidationError("SUPERSEDE successor digest is invalid")
        if successor.candidate_track_id != candidate_track_id(
            self.episode_id, successor.semantic_digest
        ):
            raise WorkspaceValidationError("SUPERSEDE successor track ID is not deterministic")
        if payload.transition_id != candidate_transition_id(
            self.episode_id,
            sequence,
            predecessor.candidate_track_id,
            CandidateTransitionType.SUPERSEDE,
        ):
            raise WorkspaceValidationError("SUPERSEDE transition ID is not deterministic")
        if (
            successor.stage != CandidateStage.ACTIVE
            or successor.disposition != CandidateDisposition.OPEN
            or successor.supersedes_candidate_track_id != predecessor.candidate_track_id
            or successor.superseded_by_candidate_track_id is not None
            or successor.formal_mechanism_id is not None
            or successor.compiled_candidate_ids
            or successor.last_transition_sequence != sequence
        ):
            raise WorkspaceValidationError("invalid atomic SUPERSEDE successor")
        if set(successor.source_event_refs) != {
            *predecessor.source_event_refs,
            *payload.basis.source_event_refs,
        }:
            raise WorkspaceValidationError("SUPERSEDE successor source events are invalid")
        self._validate_alternatives(
            successor,
            successor.alternatives,
            allowed_source_refs=set(successor.source_event_refs),
        )
        self._candidates[predecessor.candidate_track_id] = predecessor.model_copy(update={
            "stage": CandidateStage.CLOSED,
            "superseded_by_candidate_track_id": successor.candidate_track_id,
            "last_transition_sequence": sequence,
        })
        self._candidates[successor.candidate_track_id] = successor

    def _validate_transition_id(self, transition: CandidateTransition) -> None:
        if transition.transition_id != candidate_transition_id(
            self.episode_id,
            transition.sequence,
            transition.candidate_track_id,
            transition.transition_type,
        ):
            raise WorkspaceValidationError("candidate transition ID is not deterministic")

    @staticmethod
    def _validate_alternatives(
        candidate: InvestigationCandidate,
        alternatives: tuple[InvestigationAlternative, ...],
        *,
        allowed_source_refs: set[str],
    ) -> None:
        if len({item.semantic_digest for item in alternatives}) != len(alternatives):
            raise WorkspaceValidationError("candidate has duplicate alternatives")
        if len({item.alternative_id for item in alternatives}) != len(alternatives):
            raise WorkspaceValidationError("candidate has duplicate alternative IDs")
        for alternative in alternatives:
            digest = alternative_semantic_digest(
                candidate.candidate_track_id, alternative.mechanism_summary
            )
            if (
                alternative.semantic_digest != digest
                or alternative.alternative_id
                != investigation_alternative_id(candidate.candidate_track_id, digest)
                or alternative.related_candidate_track_id
                not in {None, candidate.candidate_track_id}
                or not set(alternative.source_event_refs) <= allowed_source_refs
            ):
                raise WorkspaceValidationError("candidate alternative identity is invalid")

    def snapshot(
        self,
        *,
        transcript_head_sequence: int,
        transcript_head_digest: str,
        registry_digest: str | None = None,
    ) -> InvestigationWorkspaceSnapshot:
        candidates = self.candidates
        tasks = self.tasks
        frontier = InvestigationFrontier(
            tentative_candidate_track_ids=_ids(candidates, stage=CandidateStage.TENTATIVE),
            active_candidate_track_ids=_ids(candidates, stage=CandidateStage.ACTIVE),
            deferred_candidate_track_ids=_ids(candidates, stage=CandidateStage.DEFERRED),
            closed_candidate_track_ids=_ids(candidates, stage=CandidateStage.CLOSED),
            open_candidate_track_ids=_ids(candidates, disposition=CandidateDisposition.OPEN),
            disfavored_candidate_track_ids=_ids(
                candidates, disposition=CandidateDisposition.DISFAVORED
            ),
            formalized_candidate_track_ids=tuple(
                item.candidate_track_id for item in candidates
                if item.formal_mechanism_id is not None
            ),
        )
        reachability = tuple(
            CandidateReachabilityView(
                candidate_track_id=item.candidate_track_id,
                reachability=_candidate_reachability(item, tasks),
                primary_task_ids=tuple(
                    task.task_id for task in tasks
                    if task.candidate_track_id == item.candidate_track_id
                    and task.importance == DiscriminationImportance.PRIMARY
                ),
            )
            for item in candidates
        )
        resolved_registry_digest = registry_digest or self._latest_registry_digest
        if (
            registry_digest is not None
            and self._latest_registry_digest is not None
            and registry_digest != self._latest_registry_digest
        ):
            raise WorkspaceValidationError("snapshot Registry digest differs from Workspace")
        fields = {
            "episode_id": self.episode_id,
            "schema_version": self.schema_version,
            "reducer_version": self.reducer_version,
            "candidates": candidates,
            "tasks": tasks,
            "frontier": frontier,
            "reachability": reachability,
            "discrimination_matrix": self.discrimination_matrix(),
            "shadow_selections": tuple(self._shadow_selections),
            "active_selections": tuple(self._active_selections),
            "transcript_head_sequence": transcript_head_sequence,
            "transcript_head_digest": transcript_head_digest,
            "registry_digest": resolved_registry_digest,
        }
        if self._inquiry_loop_state is not None:
            fields["inquiry_loop"] = self._inquiry_loop_state.model_dump(mode="json")
        return InvestigationWorkspaceSnapshot(
            **fields,
            snapshot_digest=sha256_digest(
                TypeAdapter(dict[str, Any]).dump_python(fields, mode="json")
            ),
        )

    def copy(self) -> InvestigationWorkspace:
        return deepcopy(self)


class InvestigationWorkspaceReducer:
    """Fold Transcript events into a Workspace and verify replayable transition bases."""

    def __init__(self, episode_id: str) -> None:
        self.workspace = InvestigationWorkspace(episode_id)
        self._event_digests: set[str] = set()
        self._observation_refs: set[str] = set()
        self._query_refs: set[str] = set()
        self._query_observation_refs: dict[str, tuple[str, ...]] = {}
        self._query_status: dict[str, str] = {}
        self._loop_config: dict[str, Any] | None = None
        self._loop_view: dict[str, Any] | None = None
        self._loop_last_round: dict[str, Any] | None = None
        self._loop_bootstrap: Any | None = None
        self._loop_budget: int | None = None
        self._loop_pending_request: tuple[int, str] | None = None
        self._prior_events: list[Any] = []

    def apply(self, event: Any) -> None:
        event_type = str(getattr(event.event_type, "value", event.event_type))
        payload = event.payload
        if event_type == "episode_created":
            mode = payload.get("inquiry_loop_mode", "legacy")
            if mode not in {"legacy", "batched"}:
                raise WorkspaceValidationError("unsupported Inquiry loop mode")
            if mode == "batched":
                from rca_agent.investigation.inquiry_loop import LoopState

                if (type(payload.get("inquiry_loop_version")) is not int
                        or payload.get("inquiry_loop_version") not in {1, 2, 3, 4}
                        or payload.get("inquiry_contract_version") != 2
                        or payload.get("investigation_workspace_mode") not in {"shadow", "active"}
                        or type(payload.get("inquiry_batch_limit")) is not int
                        or payload["inquiry_batch_limit"] not in {1, 2}
                        or type(payload.get("inquiry_query_limit")) is not int
                        or not 1 <= payload["inquiry_query_limit"] <= 10):
                    raise WorkspaceValidationError("invalid batched Inquiry host configuration")
                self._loop_config = payload
                if payload["inquiry_loop_version"] == 4:
                    self.workspace.reducer_version = WORKSPACE_REDUCER_VERSION_V2
                self.workspace._inquiry_loop_state = LoopState.initial(self.workspace.episode_id)
        elif event_type == "environment_reset" and self._loop_config is not None:
            from rca_agent.models import ResetResponse

            self._loop_bootstrap = ResetResponse.model_validate(payload["bootstrap"])
            self._loop_budget = self._loop_bootstrap.query_budget.remaining
        elif event_type == "query_cost" and self._loop_config is not None:
            self._loop_budget = payload["budget_after"]
        elif event_type == "inquiry_loop_view_recorded":
            from rca_agent.investigation.inquiry_loop import make_loop_view

            if self._loop_config is None or self._loop_bootstrap is None or self._loop_view is not None:
                raise WorkspaceValidationError("Inquiry loop view is out of order")
            expected_view, expected_menu = make_loop_view(
                self.workspace._inquiry_loop_state, tuple(self._prior_events),
                self._loop_bootstrap, budget_remaining=self._loop_budget,
                query_limit=self._loop_config["inquiry_query_limit"],
                batch_limit=self._loop_config["inquiry_batch_limit"],
                workspace_mode=self._loop_config["investigation_workspace_mode"],
                host_loop_version=self._loop_config["inquiry_loop_version"])
            if payload != {"view": expected_view, "menu": expected_menu}:
                changed = [key for key in ("view", "menu")
                           if payload.get(key) != {"view": expected_view,
                                                   "menu": expected_menu}[key]]
                if "view" in changed:
                    changed.extend(key for key in expected_view
                                   if payload["view"].get(key) != expected_view[key])
                raise WorkspaceValidationError(
                    f"Inquiry loop input differs from issued facts: {','.join(changed)}")
            self._loop_view = payload
        elif event_type == "inquiry_round_recorded" and self._loop_config is not None:
            self._loop_last_round = payload
        elif event_type == "inquiry_loop_attempt_recorded":
            from rca_agent.investigation.inquiry_loop import LoopAttempt, loop_attempt_digest

            if self._loop_config is None:
                raise WorkspaceValidationError("loop diagnostic in a legacy episode")
            attempt = LoopAttempt.model_validate(payload)
            if attempt.status == "parsed" and attempt.code != "parsed":
                raise WorkspaceValidationError("inconsistent loop attempt")
            if self._loop_view is not None and attempt.input_digest != loop_attempt_digest(
                    self._loop_view, repair_feedback=attempt.repair_feedback):
                raise WorkspaceValidationError("loop attempt input digest changed")
        elif event_type == "inquiry_loop_step_accepted":
            from rca_agent.investigation.inquiry_loop import (
                accept_loop_step,
                normalize_loop_step,
            )

            if self._loop_config is None or self._loop_view is None:
                raise WorkspaceValidationError("Inquiry step has no recorded state view")
            menu = self._loop_view["menu"]
            step = normalize_loop_step(
                payload["step"], menu,
                menu_aliases=self._loop_view["view"].get("menu_aliases"),
            )
            if step.model_dump(mode="json") != payload["step"]:
                raise WorkspaceValidationError("Inquiry step is not canonical")
            visible = {item["observation_id"] for item in self._loop_view["view"]["observations"]}
            state, round_record = accept_loop_step(
                self.workspace._inquiry_loop_state, step, menu=menu,
                batch_limit=self._loop_config["inquiry_batch_limit"],
                query_limit=self._loop_config["inquiry_query_limit"], visible_refs=visible,
                budget_remaining=self._loop_view["view"]["budget_remaining"],
                offered_refs=set(self._loop_view["view"].get("offered_menu_refs", ())) or None)
            if (payload.get("view_digest") != sha256_digest(self._loop_view)
                    or payload.get("round_record") != round_record
                    or payload.get("state") != state.model_dump(mode="json")
                    or (round_record is not None and self._loop_last_round != round_record)):
                changed = []
                if payload.get("view_digest") != sha256_digest(self._loop_view):
                    changed.append("view")
                if payload.get("round_record") != round_record:
                    changed.append("round")
                if payload.get("state") != state.model_dump(mode="json"):
                    changed.append("state")
                if round_record is not None and self._loop_last_round != round_record:
                    changed.append("recorded_round")
                raise WorkspaceValidationError("Inquiry step, round or state binding changed: "
                                               + ",".join(changed))
            self.workspace._inquiry_loop_state = state
            self._loop_view = None
            self._loop_last_round = None
        elif event_type == "startup_action_selected":
            from rca_agent.investigation.startup_authorization import (
                selected_startup_action,
                startup_grant,
            )
            if (self._loop_config is None or self._loop_config["inquiry_loop_version"] not in {3, 4}
                    or not self._prior_events or startup_grant(tuple(self._prior_events)) is not None):
                raise WorkspaceValidationError("startup selection is out of order")
            action = selected_startup_action(self.workspace._inquiry_loop_state,
                                             self._loop_bootstrap, tuple(self._prior_events))
            if (action is None or payload != {
                    "based_on_digest": self._prior_events[-1].digest,
                    "action": action.model_dump(mode="json")}):
                raise WorkspaceValidationError("startup selection differs from host policy")
        elif event_type == "tool_request" and self._loop_config is not None:
            from rca_agent.investigation.inquiry_loop import (
                admitted_batch_member,
            )
            from rca_agent.investigation.startup_authorization import (
                selected_startup_action,
                startup_grant,
            )
            from rca_agent.models import CallToolAction

            state = self.workspace._inquiry_loop_state
            action = CallToolAction.model_validate({"action_type": "call_tool",
                "tool_name": payload["tool_name"], "request": payload["request"]})
            source = payload.get("execution_source")
            if self._loop_config["inquiry_loop_version"] >= 2 and source not in {
                    "inquiry_batch", "startup", "verification"}:
                raise WorkspaceValidationError("batched request has no host execution source")
            if source in {None, "inquiry_batch"}:
                try:
                    admitted_batch_member(state, action)
                except ValueError as exc:
                    if source == "inquiry_batch" or selected_startup_action(
                            state, self._loop_bootstrap, tuple(self._prior_events)) != action:
                        raise WorkspaceValidationError(str(exc)) from exc
            elif source == "startup":
                if selected_startup_action(state, self._loop_bootstrap,
                                           tuple(self._prior_events)) != action:
                    raise WorkspaceValidationError("startup request differs from host selection")
                if self._loop_config["inquiry_loop_version"] in {3, 4}:
                    grant = startup_grant(tuple(self._prior_events))
                    if grant is None or grant["action"] != action.model_dump(mode="json"):
                        raise WorkspaceValidationError("startup request has no unconsumed host selection")
            elif source == "verification":
                if state.phase != "handoff" or not any(
                        str(getattr(e.event_type, "value", e.event_type)) == "discovery_completed"
                        for e in self._prior_events):
                    raise WorkspaceValidationError("verification request precedes Discovery handoff")
                authorized = set()
                capability_refs = set()
                resolved_caps = ()
                for prior in self._prior_events:
                    kind = str(getattr(prior.event_type, "value", prior.event_type))
                    if kind in {"dynamic_candidate_registered", "verification_plan_extended"}:
                        for signal in prior.payload["compiled"]["signals"]:
                            query = CallToolAction.model_validate(signal["query"])
                            authorized.add(query.model_dump_json())
                    elif kind == "verification_plan_compiled":
                        capability_refs.update(p["capability_ref"] for p in
                                               prior.payload["plan"]["predictions"])
                    elif kind == "capability_registry_resolved":
                        resolved_caps = prior.payload["capabilities"]
                authorized_refs = {c["query_fingerprint"] for c in resolved_caps
                                   if c["capability_ref"] in capability_refs
                                   and c["executable"]}
                if (action.model_dump_json() not in authorized
                        and payload["query_ref"] not in authorized_refs):
                    raise WorkspaceValidationError("verification query has no compiled host plan")
            else:
                raise WorkspaceValidationError("unknown tool execution source")
            self._loop_pending_request = (event.sequence, payload["query_ref"])
        elif event_type == "tool_response" and self._loop_config is not None:
            from rca_agent.investigation.inquiry_loop import apply_loop_response
            from rca_agent.models import ToolResponse

            response = ToolResponse.model_validate(payload)
            if self._loop_pending_request is None or self._loop_pending_request[1] != response.query_ref:
                raise WorkspaceValidationError("Inquiry response has no matching request")
            self.workspace._inquiry_loop_state = apply_loop_response(
                self.workspace._inquiry_loop_state, response, event.sequence)
            self._loop_pending_request = None
        elif event_type == "stop_requested" and self._loop_config is not None:
            state = self.workspace._inquiry_loop_state
            if state.phase in {"awaiting_results", "awaiting_assessment"} and state.batches:
                from rca_agent.investigation.inquiry_loop import _state

                batches = list(state.batches)
                batches[-1] = batches[-1].model_copy(update={"unassessed_reason": str(payload["reason"])})
                self.workspace._inquiry_loop_state = _state(state,
                    batches=tuple(b.model_dump(mode="json") for b in batches))
        if event_type == "investigation_candidate_added":
            added = WorkspaceCandidateAdded.model_validate(payload)
            self._validate_basis(added.transition.basis)
            self.workspace.apply_candidate_added(added, event.sequence)
        elif event_type == "candidate_transitioned":
            transition = CandidateTransition.model_validate(payload)
            self._validate_basis(transition.basis)
            self.workspace.apply_candidate_transition(transition, event.sequence)
        elif event_type == "candidate_superseded":
            superseded = CandidateSuperseded.model_validate(payload)
            self._validate_basis(superseded.basis)
            self.workspace.apply_candidate_superseded(superseded, event.sequence)
        elif event_type == "capability_registry_resolved":
            self.workspace.apply_registry_resolved(
                CapabilityRegistrySnapshot.model_validate(payload)
            )
        elif event_type == "discrimination_task_created":
            self.workspace.apply_task_created(
                DiscriminationTask.model_validate(payload), event.sequence
            )
        elif event_type == "discrimination_task_selected":
            selected = DiscriminationTaskSelected.model_validate(payload)
            if selected.selection_mode == "shadow":
                self._validate_task_query(selected.attempt.query_ref)
            self.workspace.apply_task_selected(selected, event.sequence)
        elif event_type == "discrimination_task_completed":
            completed = DiscriminationTaskCompleted.model_validate(payload)
            task = self.workspace.task(completed.task_id)
            attempt = next(
                (
                    item for item in (task.attempts if task is not None else ())
                    if item.attempt_id == completed.attempt_id
                ),
                None,
            )
            if (
                attempt is None
                or completed.observation_refs
                != self._query_observation_refs.get(attempt.query_ref, ())
                or completed.tool_status != self._query_status.get(attempt.query_ref)
            ):
                raise WorkspaceValidationError(
                    "task completion differs from its issued query response"
                )
            self.workspace.apply_task_completed(completed, event.sequence)
        elif event_type == "discrimination_task_unverifiable":
            self.workspace.apply_task_unverifiable(
                DiscriminationTaskUnverifiable.model_validate(payload), event.sequence
            )
        elif event_type == "discrimination_shadow_selection_recorded":
            self.workspace.apply_shadow_selection(
                DiscriminationShadowSelection.model_validate(payload), event.sequence
            )
        elif event_type == "discrimination_active_selection_recorded":
            self.workspace.apply_active_selection(
                DiscriminationActiveSelection.model_validate(payload), event.sequence
            )
        elif event_type == "inquiry_query_cancelled":
            self.workspace.apply_query_cancelled(payload, event.sequence)

        if event_type == "tool_response":
            query_ref = payload.get("query_ref")
            if query_ref:
                self._query_refs.add(query_ref)
                self._query_observation_refs[query_ref] = tuple(
                    item["observation_id"] for item in payload.get("observations", ())
                    if item.get("observation_id")
                )
                self._query_status[query_ref] = payload.get("status", "")
            self._observation_refs.update(
                item["observation_id"] for item in payload.get("observations", ())
                if item.get("observation_id")
            )
        self._event_digests.add(event.digest)
        self._prior_events.append(event)

    def _validate_basis(self, basis: Any) -> None:
        if not set(basis.source_event_refs) <= self._event_digests:
            raise WorkspaceValidationError("transition references a future or unknown event")
        if not set(basis.observation_refs) <= self._observation_refs:
            raise WorkspaceValidationError("transition references an unissued observation")
        if not set(basis.query_refs) <= self._query_refs:
            raise WorkspaceValidationError("transition references an unissued query")

    def _validate_task_query(self, query_ref: str) -> None:
        if query_ref not in self._query_refs:
            raise WorkspaceValidationError("task selection references an unissued query")


def reduce_investigation_workspace(
    events: Iterable[Any], episode_id: str | None = None,
) -> InvestigationWorkspace:
    items = tuple(events)
    if not items and episode_id is None:
        raise WorkspaceValidationError("workspace reduction requires an episode ID")
    resolved_episode_id = episode_id or items[0].episode_id
    reducer = InvestigationWorkspaceReducer(resolved_episode_id)
    for event in items:
        if event.episode_id != resolved_episode_id:
            raise WorkspaceValidationError("workspace event changed episode ID")
        reducer.apply(event)
    return reducer.workspace


def _ids(
    candidates: tuple[InvestigationCandidate, ...],
    *,
    stage: CandidateStage | None = None,
    disposition: CandidateDisposition | None = None,
) -> tuple[str, ...]:
    return tuple(
        item.candidate_track_id for item in candidates
        if (stage is None or item.stage == stage)
        and (disposition is None or item.disposition == disposition)
    )


def _candidate_reachability(
    candidate: InvestigationCandidate,
    tasks: tuple[DiscriminationTask, ...],
) -> CandidateVerificationReachability:
    primary = tuple(
        item for item in tasks
        if item.candidate_track_id == candidate.candidate_track_id
        and item.importance == DiscriminationImportance.PRIMARY
        and item.status != DiscriminationTaskStatus.CANCELLED
    )
    if not primary:
        return CandidateVerificationReachability.NOT_ASSESSED
    if all(item.status == DiscriminationTaskStatus.RESOLVED for item in primary):
        return CandidateVerificationReachability.DISCRIMINATION_COMPLETE
    unavailable = sum(
        item.status == DiscriminationTaskStatus.UNVERIFIABLE for item in primary
    )
    if unavailable == len(primary):
        return CandidateVerificationReachability.UNVERIFIABLE
    if unavailable:
        return CandidateVerificationReachability.PARTIALLY_ACTIONABLE
    return CandidateVerificationReachability.ACTIONABLE
