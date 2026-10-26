from __future__ import annotations

import re
import uuid
from collections.abc import Callable
from copy import deepcopy
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal, Protocol

from rca_agent.core.dynamic import (
    DiscoveryPayload,
    EvidenceRegistry,
    FrozenMechanism,
    IdentifiedMechanismHint,
    MechanismDiscoveryPayload,
    VerificationPlanDraft,
    VerificationPlanExtension,
    VerificationPlanPayload,
    freeze_mechanism,
    observation_fact_digest,
    repair_mechanical_contract,
)
from rca_agent.environment.contract import RcaEnvironment
from rca_agent.investigation import (
    AttemptOutcome,
    CandidateDisposition,
    CandidateSourceKind,
    CandidateStage,
    CandidateSuperseded,
    CandidateTransition,
    CandidateTransitionType,
    CapabilityRegistrySnapshot,
    DiscriminationActiveSelection,
    DiscriminationAttempt,
    DiscriminationImportance,
    DiscriminationShadowSelection,
    DiscriminationTask,
    DiscriminationTaskCompleted,
    DiscriminationTaskSelected,
    DiscriminationTaskStatus,
    DiscriminationTaskUnverifiable,
    FormalizationCandidate,
    FormalizationSemanticError,
    InvestigationAlternative,
    InvestigationCandidate,
    InvestigationWorkspace,
    InvestigationWorkspaceSnapshot,
    MechanismCoveragePolicy,
    MechanismCoverageRecord,
    MechanismFormalizationContext,
    MechanismFormalizationPayload,
    ResolvedCapability,
    TransitionBasis,
    WorkspaceCandidateAdded,
    alternative_semantic_digest,
    build_mechanism_coverage_input,
    candidate_semantic_digest,
    candidate_track_id,
    candidate_transition_id,
    discrimination_attempt_id,
    discrimination_task_id,
    investigation_alternative_id,
    project_formal_coverage_candidates,
    reduce_investigation_workspace,
    validate_formalization_accounting,
    validate_formalization_semantics,
)
from rca_agent.investigation.discovery_format import (
    DiscoveryFormatRepairContext,
    repair_context_from_raw,
    validate_discovery_format_repair,
)
from rca_agent.models import (
    AnswerArtifact,
    AuthoritativeDecision,
    CallToolAction,
    CoreAssessmentPreview,
    DecisionSource,
    DecisionStatus,
    EpisodeStatus,
    HypothesisProposal,
    PolicyAction,
    RequestStopAction,
    ResetResponse,
    StopReason,
    SubmitHypothesisAction,
    ToolResponse,
    ToolStatus,
    UpdateCandidateEvidenceAction,
)
from rca_agent.runner.core_decision import CoreDecisionProvider, IssuedObservation
from rca_agent.runner.inquiry_outcomes import inquiry_outcomes
from rca_agent.runner.plan_extension import added_predictions
from rca_agent.runner.transcript import EventType, JsonlTranscriptStore, Transcript
from rca_agent.tools.fingerprint import query_fingerprint, sha256_digest

ENTITY_REF_RE = re.compile(r"^ent-[a-z0-9-]{3,40}$")


def _contract_error_code(message: str) -> str:
    mappings = (
        ("unknown capability reference", "UNKNOWN_CAPABILITY_REF"),
        ("unknown alternative reference", "UNKNOWN_ALTERNATIVE_REF"),
        ("unknown frozen mechanism", "UNKNOWN_FROZEN_MECHANISM"),
        ("refutation prediction", "MISSING_REFUTATION"),
        ("dedicated refutation prediction", "ALTERNATIVES_NOT_REFUTED"),
        ("at most one alternative", "REFUTATION_NOT_DEDICATED"),
        ("support prediction cannot distinguish", "SUPPORT_CANNOT_REFUTE"),
        ("cannot refute the alternative mechanism class", "REFUTATION_SCOPE_MISMATCH"),
        ("not eligible as root-cause support", "SUPPORT_RULE_NOT_ELIGIBLE"),
        ("independent support groups", "INSUFFICIENT_SUPPORT_GROUPS"),
        ("generic GC observations", "FULL_GC_SIGNAL_UNAVAILABLE"),
        ("duplicate predictions", "DUPLICATE_PREDICTIONS"),
        ("not executable", "CAPABILITY_NOT_EXECUTABLE"),
        ("removed or changed a frozen prediction", "NON_ADDITIVE_PLAN"),
        ("requires one or two new predictions", "NO_NEW_PREDICTION"),
    )
    return next((code for fragment, code in mappings if fragment in message), "CONTRACT_REJECTED")


class EpisodeStateError(RuntimeError):
    pass


class ActionRejectedError(ValueError):
    pass


class DecisionProvider(Protocol):
    def decide(
        self,
        proposals: tuple[HypothesisProposal, ...],
        observations: tuple[IssuedObservation, ...],
        termination_reason: StopReason,
    ) -> AuthoritativeDecision: ...


class LifecycleDefaultDecisionProvider:
    """Phase 2 lifecycle decision only; it has no root-cause confirmation authority."""

    def decide(
        self,
        proposals: tuple[HypothesisProposal, ...],
        observations: tuple[IssuedObservation, ...],
        termination_reason: StopReason,
    ) -> AuthoritativeDecision:
        del proposals, observations, termination_reason
        return AuthoritativeDecision(
            status=DecisionStatus.INCONCLUSIVE,
            source=DecisionSource.PHASE2_LIFECYCLE_DEFAULT,
            reason="phase 2 validates lifecycle and isolation without a diagnosis policy",
        )


class EpisodeRunner:
    def __init__(
        self,
        environment: RcaEnvironment,
        *,
        max_rounds: int = 20,
        bound_handoff_version: int = 7,
        engineering_contract_revision: int = 2,
        inquiry_contract_version: Literal[1, 2] | None = None,
        inquiry_loop_mode: Literal["legacy", "batched"] = "legacy",
        inquiry_loop_version: Literal[3, 4] | None = None,
        inquiry_batch_limit: Literal[1, 2] = 2,
        inquiry_query_limit: int = 10,
        transcript_path: Path | None = None,
        decision_provider: DecisionProvider | None = None,
        discovery: Any | None = None,
        mechanism_formalizer: Any | None = None,
        discovery_format_formalizer: Any | None = None,
        verification_planner: Any | None = None,
        evidence_registry: EvidenceRegistry | Callable[..., EvidenceRegistry] | None = None,
        discovery_timeout_seconds: float = 30,
        investigation_workspace_mode: Literal["off", "shadow", "active"] = "shadow",
        mechanism_coverage_mode: Literal["off", "shadow"] = "off",
    ) -> None:
        if max_rounds < 1:
            raise ValueError("max_rounds must be positive")
        if discovery_timeout_seconds <= 0 or discovery_timeout_seconds > 600:
            raise ValueError("discovery timeout must be in (0, 600]")
        if investigation_workspace_mode not in {"off", "shadow", "active"}:
            raise ValueError("investigation workspace mode must be off, shadow or active")
        if inquiry_loop_mode not in {"legacy", "batched"}:
            raise ValueError("unsupported Inquiry loop mode")
        if inquiry_loop_version is None:
            inquiry_loop_version = 4 if inquiry_loop_mode == "batched" else 3
        if type(inquiry_loop_version) is not int or inquiry_loop_version not in {3, 4}:
            raise ValueError("unsupported Inquiry loop version")
        if inquiry_loop_mode != "batched" and inquiry_loop_version != 3:
            raise ValueError("legacy Inquiry cannot select a batched loop version")
        if inquiry_loop_mode == "batched" and investigation_workspace_mode == "off":
            raise ValueError("batched Inquiry requires shadow or active Workspace")
        if type(inquiry_batch_limit) is not int or inquiry_batch_limit not in {1, 2}:
            raise ValueError("Inquiry batch limit must be one or two")
        if type(inquiry_query_limit) is not int or not 1 <= inquiry_query_limit <= 10:
            raise ValueError("Inquiry query limit must be between one and ten")
        if mechanism_coverage_mode not in {"off", "shadow"}:
            raise ValueError("mechanism coverage mode must be off or shadow")
        if discovery is not None and mechanism_formalizer is not None:
            raise ValueError("configure Discovery or MechanismFormalizer, not both")
        if discovery_format_formalizer is not None and (
            discovery is None or verification_planner is None
        ):
            raise ValueError("Discovery format repair requires Discovery and planner")
        if (discovery is not None or mechanism_formalizer is not None) and evidence_registry is None:
            raise ValueError("discovery requires host-owned signal and rule contracts")
        if verification_planner is not None and discovery is None and mechanism_formalizer is None:
            raise ValueError("verification planner requires mechanism formalization")
        if mechanism_formalizer is not None and verification_planner is None:
            raise ValueError("MechanismFormalizer requires a verification planner")
        if isinstance(decision_provider, CoreDecisionProvider) and decision_provider.has_dynamic_candidates:
            raise ValueError("a runner requires a fresh Core decision provider")
        self.episode_id = f"ep-{uuid.uuid4().hex}"
        store = JsonlTranscriptStore(transcript_path) if transcript_path is not None else None
        self.transcript = Transcript(self.episode_id, store)
        self.status = EpisodeStatus.CREATED
        self.__environment = environment
        self.__max_rounds = max_rounds
        self.__rounds = 0
        self.__decision_provider = (deepcopy(decision_provider)
                                   if isinstance(decision_provider, CoreDecisionProvider)
                                   else decision_provider or CoreDecisionProvider())
        self.__discovery = discovery
        self.__mechanism_formalizer = mechanism_formalizer
        self.__discovery_format_formalizer = discovery_format_formalizer
        self.__verification_planner = verification_planner
        if bound_handoff_version not in {6, 7}:
            raise ValueError("unsupported bound handoff version")
        self.__bound_handoff_version = bound_handoff_version
        from rca_agent.investigation.contract_enforcement import validate_revision
        self.__engineering_revision = validate_revision(engineering_contract_revision)
        from rca_agent.investigation.contract_enforcement import inquiry_host_contract
        config = {"engineering_contract_revision": engineering_contract_revision,
                  "bound_handoff_version": bound_handoff_version}
        if inquiry_contract_version is not None:
            config["inquiry_contract_version"] = inquiry_contract_version
        self.__inquiry_contract = inquiry_host_contract(config)
        if inquiry_loop_mode == "batched" and (self.__inquiry_contract != 2
                or self.__engineering_revision != 2):
            raise ValueError("batched Inquiry requires the host's strict v2 contract")
        self.__inquiry_loop_mode = inquiry_loop_mode
        self.__inquiry_loop_version = inquiry_loop_version
        self.__inquiry_batch_limit = inquiry_batch_limit
        self.__inquiry_query_limit = inquiry_query_limit
        self.__registry = evidence_registry
        self.__discovery_timeout = discovery_timeout_seconds
        self.__discovery_attempted = False
        self.__dynamic_options: list[dict[str, Any]] = []
        self.__verification_queries: list[tuple[str, CallToolAction]] = []
        self.__tool_grant: tuple[str, CallToolAction] | None = None
        self.__tool_grant_validated = False
        self.__startup_selection: tuple[CallToolAction, str] | None = None
        self.__resolved_registry: EvidenceRegistry | None = None
        self.__frozen_mechanisms: dict[str, FrozenMechanism] = {}
        self.__compiled_plans: dict[str, VerificationPlanDraft] = {}
        self.__candidate_for_mechanism: dict[str, str] = {}
        self.__feedback_planned = False
        self.__inquiry_rounds: list[dict[str, Any]] = []
        self.__workspace_mode = investigation_workspace_mode
        self.__coverage_mode = mechanism_coverage_mode
        self.__investigation_workspace = InvestigationWorkspace(self.episode_id)
        self.__workspace_finalized = False
        self.__active_selected_action: CallToolAction | None = None
        self.__active_offered_query_refs: set[str] = set()
        self.__active_selected_attempt_id: str | None = None
        self.__active_selected_task_id: str | None = None
        self.__pending_active_actions: dict[str, CallToolAction] = {}
        self.__bootstrap: ResetResponse | None = None
        self.__budget_remaining = 0
        self.__proposals: list[HypothesisProposal] = []
        self.__observation_refs: set[str] = set()
        self.__observation_ledger: dict[str, IssuedObservation] = {}
        self.__issued_entities: set[str] = set()
        self.__answer: AnswerArtifact | None = None
        self.transcript.append(
            EventType.EPISODE_CREATED,
            {
                "status": self.status,
                "max_rounds": max_rounds,
                "bound_handoff_version": bound_handoff_version,
                "engineering_contract_revision": engineering_contract_revision,
                "inquiry_contract_version": self.__inquiry_contract,
                **({"inquiry_loop_mode": "batched", "inquiry_loop_version": self.__inquiry_loop_version,
                    "inquiry_batch_limit": inquiry_batch_limit,
                    "inquiry_query_limit": inquiry_query_limit}
                   if inquiry_loop_mode == "batched" else {}),
                "investigation_workspace_mode": investigation_workspace_mode,
                "mechanism_coverage_mode": mechanism_coverage_mode,
                "excluded_catalog_candidate_ids": sorted(
                    self.excluded_catalog_candidate_ids
                ),
            },
        )

    @property
    def answer(self) -> AnswerArtifact | None:
        return self.__answer

    @property
    def rounds(self) -> int:
        return self.__rounds

    @property
    def bootstrap(self) -> ResetResponse | None:
        return self.__bootstrap

    @property
    def dynamic_candidate_options(self) -> tuple[dict[str, Any], ...]:
        return tuple(deepcopy(self.__dynamic_options))

    @property
    def investigation_workspace_snapshot(self) -> InvestigationWorkspaceSnapshot:
        """Read-only projection of transcript-backed investigation state."""

        if self.__inquiry_loop_mode == "batched":
            self.__investigation_workspace = reduce_investigation_workspace(
                self.transcript.events, self.episode_id)
        return self.__investigation_workspace.snapshot(
            transcript_head_sequence=len(self.transcript.events),
            transcript_head_digest=self.transcript.head_digest,
        )

    @property
    def excluded_catalog_candidate_ids(self) -> frozenset[str]:
        provider = self.__decision_provider
        return (
            provider.excluded_catalog_candidate_ids
            if isinstance(provider, CoreDecisionProvider)
            else frozenset()
        )

    @property
    def inquiry_loop_mode(self) -> str:
        return self.__inquiry_loop_mode

    @property
    def inquiry_query_limit(self) -> int:
        return self.__inquiry_query_limit

    @property
    def inquiry_loop_phase(self) -> str | None:
        if self.__inquiry_loop_mode != "batched":
            return None
        return self.investigation_workspace_snapshot.inquiry_loop["phase"]

    def record_inquiry_loop_attempt(self, record: dict[str, Any]) -> None:
        from rca_agent.investigation.inquiry_loop import LoopAttempt, loop_attempt_digest

        self._require_status(EpisodeStatus.ACTIVE)
        if self.__inquiry_loop_mode != "batched":
            raise ValueError("loop diagnostic in legacy episode")
        payload = LoopAttempt.model_validate(record)
        feedback = record.get("repair_feedback") if isinstance(record, dict) else None
        expected = loop_attempt_digest(self.inquiry_loop_view(), repair_feedback=feedback)
        if payload.input_digest != expected:
            raise ValueError("loop diagnostic does not bind the current input")
        self._append_workspace_event(EventType.INQUIRY_LOOP_ATTEMPT_RECORDED, payload)

    def inquiry_loop_handoff(self) -> dict[str, Any]:
        from rca_agent.investigation.inquiry_loop import LoopState, loop_handoff

        if self.__inquiry_loop_mode != "batched":
            raise ValueError("legacy Inquiry has no batched handoff")
        state = LoopState.model_validate(self.investigation_workspace_snapshot.inquiry_loop)
        return loop_handoff(state, self.__inquiry_rounds, self.transcript.events)

    def inquiry_loop_view(self) -> dict[str, Any]:
        """Read-only current state and visible results; no model or tool call."""
        from rca_agent.investigation.inquiry_loop import make_loop_view

        self._require_status(EpisodeStatus.ACTIVE)
        if self.__inquiry_loop_mode != "batched" or self.__bootstrap is None:
            raise ValueError("batched Inquiry is not enabled")
        workspace = self.investigation_workspace_snapshot
        from rca_agent.investigation.inquiry_loop import LoopState

        state = LoopState.model_validate(workspace.inquiry_loop)
        view, menu = make_loop_view(state, self.transcript.events, self.__bootstrap,
            budget_remaining=self.__budget_remaining,
            query_limit=self.__inquiry_query_limit,
            batch_limit=self.__inquiry_batch_limit,
            workspace_mode=self.__workspace_mode,
            host_loop_version=self.__inquiry_loop_version)
        return {"view": view, "menu": menu}

    def record_inquiry_loop_step(self, raw: dict[str, Any], *,
                                 input_view: dict[str, Any]) -> dict[str, Any] | None:
        """Validate one complete step before changing the authoritative Workspace."""
        from rca_agent.investigation.inquiry_loop import (
            LoopState,
            accept_loop_step,
            normalize_loop_step,
        )
        from rca_agent.langchain_acquisition import parse_inquiry_draft

        self._require_status(EpisodeStatus.ACTIVE)
        if self.__inquiry_loop_mode != "batched" or input_view != self.inquiry_loop_view():
            raise ValueError("Inquiry step did not use the current host view")
        state = LoopState.model_validate(self.investigation_workspace_snapshot.inquiry_loop)
        step = normalize_loop_step(
            raw, input_view["menu"],
            menu_aliases=input_view["view"].get("menu_aliases"),
        )
        visible = {item["observation_id"] for item in input_view["view"]["observations"]}
        next_state, round_record = accept_loop_step(state, step,
            menu=input_view["menu"], batch_limit=self.__inquiry_batch_limit,
            query_limit=self.__inquiry_query_limit, visible_refs=visible,
            budget_remaining=input_view["view"]["budget_remaining"],
            offered_refs=set(input_view["view"].get("offered_menu_refs", ())) or None)
        if round_record is not None:
            # The normal Runner binding gate must pass before even the view is recorded.
            parsed = parse_inquiry_draft(round_record["draft"], live_version=2, strict=True)
            self._validate_selected_actions(parsed, round_record["selected_actions"],
                                            round_record["round"])
        self._append_workspace_event(EventType.INQUIRY_LOOP_VIEW_RECORDED, input_view)
        if round_record is not None:
            self.record_inquiry_round(round_record)
        self._append_workspace_event(EventType.INQUIRY_LOOP_STEP_ACCEPTED, {
            "view_digest": sha256_digest(input_view), "step": step.model_dump(mode="json"),
            "round_record": round_record, "state": next_state.model_dump(mode="json"),
        })
        return round_record

    def record_inquiry_attempt(self, record: dict[str, Any]) -> None:
        """Persist only bounded structured diagnostics, never raw model reasoning."""
        self._require_status(EpisodeStatus.ACTIVE)
        from rca_agent.inquiry_policy import validate_inquiry_attempt

        payload = validate_inquiry_attempt(record)
        self.transcript.append(EventType.INQUIRY_ATTEMPT_RECORDED, payload)

    def record_inquiry_round(self, round_record: dict[str, Any]) -> None:
        """Bind each model-authored exploration draft before its selected query executes."""
        from rca_agent.langchain_acquisition import parse_inquiry_draft

        self._require_status(EpisodeStatus.ACTIVE)
        if self.__discovery_attempted or len(self.__inquiry_rounds) >= 5:
            raise ValueError("inquiry round is out of bounds")
        round_number = round_record["round"]
        if (not 1 <= round_number <= 5
                or self.__inquiry_rounds and round_number <= self.__inquiry_rounds[-1]["round"]):
            raise ValueError("inquiry rounds must increase")
        draft = parse_inquiry_draft(round_record["draft"], live_version=self.__inquiry_contract,
                                    strict=self.__engineering_revision == 2)
        payload = {"round": round_number, "draft": draft.model_dump(mode="json")}
        if self.__workspace_mode == "active" or draft.contract_version == 2:
            payload["selected_actions"] = self._validate_selected_actions(
                draft, round_record.get("selected_actions"), round_number
            )
        self.__inquiry_rounds.append(payload)
        source_event = self.transcript.append(EventType.INQUIRY_ROUND_RECORDED, payload)
        if self.__workspace_mode in {"shadow", "active"}:
            self._record_investigation_candidates(draft, round_number, source_event.digest)
        if self.__workspace_mode == "active" or (self.__workspace_mode == "shadow" and draft.contract_version == 2):
            self._record_active_discrimination_round(payload)

    def _validate_selected_actions(self, draft: Any, raw: Any, round_number: int = 1) -> list[dict[str, Any]]:
        if draft.contract_version == 2:
            from rca_agent.investigation.inquiry_contract import validate_bindings
            from rca_agent.investigation.query_strategy import issued_entities
            from rca_agent.langchain_acquisition import query_options
            observations = tuple(r.observation for r in self.__observation_ledger.values())
            menu = query_options({"bootstrap": self.__bootstrap.model_dump(mode="json")},
                                 issued_entities(self.__bootstrap, observations))
            return validate_bindings(draft, raw, round_number, menu,
                max_bindings=16 if self.__inquiry_loop_mode == "batched" else 2)
        if not isinstance(raw, list) or not raw:
            raise ValueError("active Inquiry round requires selected_actions")
        choices = {(item.need_id, item.query_ref) for item in draft.choices}
        normalized = []
        seen: set[tuple[str, str]] = set()
        for item in raw:
            if not isinstance(item, dict) or (item.get("need_id"), item.get("query_ref")) not in choices:
                raise ValueError("selected action is not bound to an Inquiry need")
            action = CallToolAction.model_validate(item.get("action"))
            offered_ref = item.get("query_ref")
            expected_ref = "aq-" + sha256_digest(action.model_dump(mode="json"))[:20]
            entity_ref = getattr(action.request, "entity_ref", None)
            signal = getattr(action.request, "signal", "related_entity")
            if (
                offered_ref != expected_ref
                or item.get("entity_ref") != entity_ref
                or item.get("signal") != signal
                or (item["need_id"], expected_ref) in seen
            ):
                raise ValueError("selected action differs from its offered query")
            seen.add((item["need_id"], expected_ref))
            normalized.append({
                "need_id": item["need_id"],
                "query_ref": offered_ref,
                "signal": signal,
                "entity_ref": entity_ref,
                "action": action.model_dump(mode="json"),
            })
        return normalized

    def route_active_inquiry_action(self, action: PolicyAction) -> PolicyAction | None:
        """Apply the pre-query Matrix choice only to actions offered in the current round."""

        if self.__workspace_mode != "active":
            return action
        if isinstance(action, RequestStopAction) and self.__pending_active_actions:
            self._select_pending_active_query()
            if self.__active_selected_action is not None:
                selected = self.__active_selected_action
                self.__active_selected_action = None
                return selected
        if not isinstance(action, CallToolAction):
            return action
        query_ref = query_fingerprint(action.tool_name, action.request)
        if query_ref not in self.__active_offered_query_refs:
            return action
        if self.__active_selected_action is None:
            self._select_pending_active_query()
        if self.__active_selected_action is None:
            return None
        selected = self.__active_selected_action
        self.__active_selected_action = None
        return selected

    def _record_investigation_candidates(
        self, draft: Any, round_number: int, source_event_digest: str,
    ) -> None:
        from rca_agent.langchain_acquisition import inquiry_candidate_records

        stable_records = inquiry_candidate_records(self.__inquiry_rounds)
        stable_ids = {
            (
                _normalized_semantic_text(item.hypothesis.mechanism),
                _normalized_semantic_text(item.hypothesis.alternative),
            ): item.hypothesis.hypothesis_id
            for item in stable_records
        }
        seen: set[str] = set()
        for hypothesis in draft.hypotheses:
            stable_hypothesis_id = stable_ids[(
                _normalized_semantic_text(hypothesis.mechanism),
                _normalized_semantic_text(hypothesis.alternative),
            )]
            semantic_digest = candidate_semantic_digest(
                hypothesis.mechanism,
            )
            existing = self.__investigation_workspace.candidate_by_semantic_digest(
                semantic_digest
            )
            track_id = candidate_track_id(self.episode_id, semantic_digest)
            alternative_digest = alternative_semantic_digest(
                track_id, hypothesis.alternative
            )
            alternative = InvestigationAlternative(
                source_alternative_id=hypothesis.alternative_id,
                alternative_id=investigation_alternative_id(
                    track_id, alternative_digest
                ),
                mechanism_summary=hypothesis.alternative,
                source_event_refs=(source_event_digest,),
                semantic_digest=alternative_digest,
            )
            if existing is None:
                sequence = len(self.transcript.events) + 1
                candidate = InvestigationCandidate(
                    candidate_track_id=track_id,
                    mechanism_summary=hypothesis.mechanism,
                    stage=CandidateStage.TENTATIVE,
                    disposition=CandidateDisposition.OPEN,
                    alternatives=(alternative,),
                    source_kind=CandidateSourceKind.INQUIRY,
                    source_event_refs=(source_event_digest,),
                    source_hypothesis_ids=(stable_hypothesis_id,),
                    first_seen_round=round_number,
                    last_updated_round=round_number,
                    semantic_digest=semantic_digest,
                    created_sequence=sequence,
                    last_transition_sequence=sequence,
                )
                basis = TransitionBasis(
                    source_event_refs=(source_event_digest,),
                    source_round=round_number,
                    reason_code="INQUIRY_CANDIDATE_ADDED",
                    rationale="A schema-valid Inquiry round introduced this tentative idea.",
                )
                transition = CandidateTransition(
                    transition_id=self._workspace_transition_id(
                        sequence, track_id, CandidateTransitionType.ADD
                    ),
                    candidate_track_id=track_id,
                    transition_type=CandidateTransitionType.ADD,
                    from_stage=None,
                    to_stage=CandidateStage.TENTATIVE,
                    from_disposition=None,
                    to_disposition=CandidateDisposition.OPEN,
                    previous_semantic_digest=None,
                    resulting_semantic_digest=semantic_digest,
                    source_hypothesis_ids_added=(stable_hypothesis_id,),
                    basis=basis,
                    sequence=sequence,
                )
                self._append_workspace_event(
                    EventType.INVESTIGATION_CANDIDATE_ADDED,
                    WorkspaceCandidateAdded(candidate=candidate, transition=transition),
                )
                existing = candidate
            alternatives_added = (
                ()
                if any(
                    item.semantic_digest == alternative.semantic_digest
                    for item in existing.alternatives
                )
                else (alternative,)
            )
            if existing.candidate_track_id in seen:
                if alternatives_added:
                    self._append_candidate_transition(
                        existing,
                        CandidateTransitionType.RETAIN,
                        to_stage=CandidateStage.ACTIVE,
                        to_disposition=existing.disposition,
                        basis=TransitionBasis(
                            source_event_refs=(source_event_digest,),
                            source_round=round_number,
                            reason_code="INQUIRY_ALTERNATIVE_ADDED",
                            rationale=(
                                "The same causal candidate was paired with another "
                                "competing explanation."
                            ),
                        ),
                        alternatives_added=alternatives_added,
                        source_hypothesis_ids_added=(stable_hypothesis_id,),
                    )
                continue
            seen.add(existing.candidate_track_id)
            if existing.first_seen_round == round_number:
                continue
            self._append_candidate_transition(
                existing,
                CandidateTransitionType.RETAIN,
                to_stage=CandidateStage.ACTIVE,
                to_disposition=existing.disposition,
                basis=TransitionBasis(
                    source_event_refs=(source_event_digest,),
                    source_round=round_number,
                    reason_code="INQUIRY_CANDIDATE_RETAINED",
                    rationale="A later Inquiry round repeated the same normalized causal idea.",
                ),
                alternatives_added=alternatives_added,
                source_hypothesis_ids_added=(stable_hypothesis_id,),
            )

    def _append_candidate_transition(
        self,
        candidate: InvestigationCandidate,
        transition_type: CandidateTransitionType,
        *,
        to_stage: CandidateStage,
        to_disposition: CandidateDisposition,
        basis: TransitionBasis,
        mechanism_summary: str | None = None,
        alternatives_added: tuple[InvestigationAlternative, ...] = (),
        source_hypothesis_ids_added: tuple[str, ...] = (),
        formal_mechanism_id: str | None = None,
        compiled_candidate_ids_added: tuple[str, ...] = (),
    ) -> None:
        sequence = len(self.transcript.events) + 1
        transition = CandidateTransition(
            transition_id=self._workspace_transition_id(
                sequence, candidate.candidate_track_id, transition_type
            ),
            candidate_track_id=candidate.candidate_track_id,
            transition_type=transition_type,
            from_stage=candidate.stage,
            to_stage=to_stage,
            from_disposition=candidate.disposition,
            to_disposition=to_disposition,
            previous_semantic_digest=candidate.semantic_digest,
            resulting_semantic_digest=candidate.semantic_digest,
            mechanism_summary=mechanism_summary,
            alternatives_added=alternatives_added,
            source_hypothesis_ids_added=source_hypothesis_ids_added,
            formal_mechanism_id=formal_mechanism_id,
            compiled_candidate_ids_added=compiled_candidate_ids_added,
            basis=basis,
            sequence=sequence,
        )
        self._append_workspace_event(EventType.CANDIDATE_TRANSITIONED, transition)

    def _append_candidate_superseded(
        self,
        candidate: InvestigationCandidate,
        mechanism_summary: str,
        basis: TransitionBasis,
    ) -> str:
        sequence = len(self.transcript.events) + 1
        semantic_digest = candidate_semantic_digest(
            mechanism_summary,
            mechanism_class=candidate.mechanism_class,
            causal_entity_ref=candidate.causal_entity_ref,
        )
        successor_id = candidate_track_id(self.episode_id, semantic_digest)
        successor = InvestigationCandidate(
            candidate_track_id=successor_id,
            mechanism_summary=mechanism_summary,
            mechanism_class=candidate.mechanism_class,
            causal_entity_ref=candidate.causal_entity_ref,
            causal_explanation=candidate.causal_explanation,
            system_layer=candidate.system_layer,
            stage=CandidateStage.ACTIVE,
            disposition=CandidateDisposition.OPEN,
            alternatives=candidate.alternatives,
            source_kind=CandidateSourceKind.DISCOVERY_REVISION,
            source_event_refs=tuple(dict.fromkeys((
                *candidate.source_event_refs, *basis.source_event_refs,
            ))),
            source_hypothesis_ids=candidate.source_hypothesis_ids,
            first_seen_round=candidate.first_seen_round,
            last_updated_round=basis.source_round or candidate.last_updated_round,
            semantic_digest=semantic_digest,
            supersedes_candidate_track_id=candidate.candidate_track_id,
            created_sequence=sequence,
            last_transition_sequence=sequence,
        )
        payload = CandidateSuperseded(
            transition_id=self._workspace_transition_id(
                sequence, candidate.candidate_track_id, CandidateTransitionType.SUPERSEDE
            ),
            predecessor_track_id=candidate.candidate_track_id,
            predecessor_semantic_digest=candidate.semantic_digest,
            successor=successor,
            basis=basis,
            sequence=sequence,
        )
        self._append_workspace_event(EventType.CANDIDATE_SUPERSEDED, payload)
        return successor_id

    def _append_workspace_event(self, event_type: EventType, payload: Any) -> None:
        self.transcript.append(event_type, payload)
        self.__investigation_workspace = reduce_investigation_workspace(
            self.transcript.events, self.episode_id,
        )

    def _record_capability_registry_snapshot(
        self, registry: EvidenceRegistry, query_catalog: tuple[ResolvedCapability, ...] = (),
    ) -> CapabilityRegistrySnapshot:
        assert self.__bootstrap is not None
        costs = {
            item.tool_name: item.query_cost for item in self.__bootstrap.capabilities
        }
        signals = {(item.entity_ref, item.signal): item for item in registry.signals}
        capabilities = []
        for handle in registry.capability_handles():
            signal = signals[(handle.entity_ref, handle.signal)]
            tool_name = signal.query.tool_name.value
            capabilities.append(ResolvedCapability(
                capability_ref=handle.capability_ref,
                entity_ref=handle.entity_ref,
                rule_id=handle.rule_id,
                signal=handle.signal,
                independent_group=handle.independent_group,
                support_eligible=handle.support_eligible,
                refutes=tuple(item.value for item in handle.refutes),
                query_fingerprint=query_fingerprint(
                    signal.query.tool_name, signal.query.request
                ),
                tool_name=tool_name,
                query_cost=costs[tool_name],
                executable=self._signal_executable(signal),
            ))
        capabilities.extend(query_catalog)
        digest_input = registry.model_dump(mode="json")
        if query_catalog:
            digest_input = {"evidence_registry": digest_input,
                            "query_catalog_v2": [c.model_dump(mode="json") for c in query_catalog]}
        fields = {
            "registry_digest": sha256_digest(digest_input),
            "capabilities": tuple(sorted(
                capabilities, key=lambda item: item.capability_ref
            )),
        }
        snapshot = CapabilityRegistrySnapshot(
            **fields,
            snapshot_digest=sha256_digest(_jsonable(fields)),
        )
        self._append_workspace_event(EventType.CAPABILITY_REGISTRY_RESOLVED, snapshot)
        return snapshot

    def _record_discrimination_shadow(
        self,
        handoff: Any | None,
        registry_snapshot: CapabilityRegistrySnapshot,
    ) -> None:
        """Retrospective diagnostics only; never changes which query executes."""

        if handoff is None or not self.__inquiry_rounds:
            return
        from rca_agent.langchain_acquisition import inquiry_candidate_records

        stable_records = inquiry_candidate_records(self.__inquiry_rounds)
        stable_by_key = {
            (
                _normalized_semantic_text(item.hypothesis.mechanism),
                _normalized_semantic_text(item.hypothesis.alternative),
            ): item.hypothesis.hypothesis_id
            for item in stable_records
        }
        link_queues: dict[tuple[str, str], list[Any]] = {}
        for link in handoff.query_links:
            if link.hypothesis_id is not None:
                link_queues.setdefault((
                    link.hypothesis_id,
                    _normalized_semantic_text(link.question),
                ), []).append(link)
        responses = {
            event.payload["query_ref"]: event.payload
            for event in self.transcript.events
            if event.event_type == EventType.TOOL_RESPONSE
        }
        created_task_ids: list[str] = []
        legacy_task_order: list[str] = []
        known_task_ids = {item.task_id for item in self.__investigation_workspace.tasks}

        for round_record in self.__inquiry_rounds:
            if round_record["draft"].get("contract_version") == 2:
                continue
            round_number = round_record["round"]
            draft = round_record["draft"]
            hypotheses = {item["hypothesis_id"]: item for item in draft["hypotheses"]}
            choices = {item["need_id"]: item["query_ref"] for item in draft["choices"]}
            for need in draft["needs"]:
                hypothesis = hypotheses[need["hypothesis_id"]]
                stable_id = stable_by_key[(
                    _normalized_semantic_text(hypothesis["mechanism"]),
                    _normalized_semantic_text(hypothesis["alternative"]),
                )]
                candidate = next((
                    item for item in self.__investigation_workspace.candidates
                    if stable_id in item.source_hypothesis_ids
                ), None)
                if candidate is None:
                    continue
                alternative = next((
                    item for item in candidate.alternatives
                    if _normalized_semantic_text(item.mechanism_summary)
                    == _normalized_semantic_text(hypothesis["alternative"])
                ), None)
                if alternative is None:
                    continue
                link_key = (stable_id, _normalized_semantic_text(need["question"]))
                links = link_queues.get(link_key, [])
                link = links.pop(0) if links else None
                options = tuple(
                    item for item in registry_snapshot.capabilities
                    if link is not None
                    and item.executable
                    and item.entity_ref == link.entity_ref
                    and item.query_fingerprint == link.query_ref
                )
                task_id = discrimination_task_id(
                    self.episode_id,
                    candidate.candidate_track_id,
                    alternative.alternative_id,
                    need["question"],
                    need["expected_if_true"],
                    need["expected_if_alternative"],
                )
                if task_id in known_task_ids:
                    continue
                known_task_ids.add(task_id)
                sequence = len(self.transcript.events) + 1
                legacy_choice_ref = choices.get(need["need_id"])
                task = DiscriminationTask(
                    task_id=task_id,
                    candidate_track_id=candidate.candidate_track_id,
                    source_need_id=need["need_id"],
                    source_round=round_number,
                    question=need["question"],
                    alternative_id=alternative.alternative_id,
                    observable_kind=(link.signal if link is not None else "unresolved_need"),
                    candidate_prediction=need["expected_if_true"],
                    comparator_prediction=need["expected_if_alternative"],
                    importance=(
                        DiscriminationImportance.PRIMARY
                        if legacy_choice_ref is not None
                        else DiscriminationImportance.SUPPLEMENTAL
                    ),
                    status=DiscriminationTaskStatus.OPEN,
                    capability_options=options,
                    legacy_choice_ref=legacy_choice_ref,
                    legacy_query_ref=(link.query_ref if link is not None else None),
                    created_registry_digest=registry_snapshot.registry_digest,
                    last_evaluated_registry_digest=registry_snapshot.registry_digest,
                    unresolved_capability_reason=(
                        None
                        if options
                        else (
                            "No exact executable Registry capability was bound to this "
                            "natural-language discriminator in the current Episode."
                        )
                    ),
                    created_sequence=sequence,
                    last_transition_sequence=sequence,
                )
                self._append_workspace_event(EventType.DISCRIMINATION_TASK_CREATED, task)
                created_task_ids.append(task_id)
                if legacy_choice_ref is not None:
                    legacy_task_order.append(task_id)

        matrix = self.__investigation_workspace.discrimination_matrix()
        top = matrix.entries[0] if matrix.entries else None
        sequence = len(self.transcript.events) + 1
        selection = DiscriminationShadowSelection(
            selection_id="selection-" + sha256_digest({
                "episode_id": self.episode_id,
                "sequence": sequence,
                "matrix_digest": matrix.matrix_digest,
            })[:20],
            matrix_digest=matrix.matrix_digest,
            evaluated_task_ids=tuple(
                item.task_id for item in self.__investigation_workspace.tasks
            ),
            selected_task_id=(top.task_id if top else None),
            selected_capability_ref=(top.capability_ref if top else None),
            legacy_first_task_id=(legacy_task_order[0] if legacy_task_order else None),
            matches_legacy_first=(
                None
                if top is None or not legacy_task_order
                else top.task_id == legacy_task_order[0]
            ),
            reason_codes=(
                "SHADOW_ONLY",
                "LEGACY_QUERY_EXECUTION_UNCHANGED",
            ),
            sequence=sequence,
        )
        self._append_workspace_event(
            EventType.DISCRIMINATION_SHADOW_SELECTION_RECORDED, selection
        )

        ranked_by_task = {}
        for entry in matrix.entries:
            ranked_by_task.setdefault(entry.task_id, entry)
        for task_id in created_task_ids:
            task = next(
                item for item in self.__investigation_workspace.tasks
                if item.task_id == task_id
            )
            entry = ranked_by_task.get(task_id)
            response = responses.get(task.legacy_query_ref or "")
            if entry is None:
                sequence = len(self.transcript.events) + 1
                self._append_workspace_event(
                    EventType.DISCRIMINATION_TASK_UNVERIFIABLE,
                    DiscriminationTaskUnverifiable(
                        task_id=task.task_id,
                        reason_code=(
                            "NO_EXACT_CAPABILITY_BINDING"
                        ),
                        reason=(
                            task.unresolved_capability_reason
                            or "No exact executable Registry capability was bound."
                        ),
                        registry_digest=registry_snapshot.registry_digest,
                        sequence=sequence,
                    ),
                )
                continue
            if response is None:
                # Handoff validation normally makes this unreachable. Keep the task open
                # rather than letting a shadow diagnostic alter Discovery behavior.
                continue
            sequence = len(self.transcript.events) + 1
            attempt = DiscriminationAttempt(
                attempt_id=discrimination_attempt_id(
                    self.episode_id,
                    task.task_id,
                    entry.capability_ref,
                    entry.query_fingerprint,
                ),
                task_id=task.task_id,
                capability_ref=entry.capability_ref,
                query_ref=task.legacy_query_ref,
                query_fingerprint=entry.query_fingerprint,
                query_cost=entry.query_cost,
                independent_group=entry.independent_group,
                registry_digest=registry_snapshot.registry_digest,
                selection_reason_codes=entry.reason_codes,
                selected_sequence=sequence,
            )
            self._append_workspace_event(
                EventType.DISCRIMINATION_TASK_SELECTED,
                DiscriminationTaskSelected(
                    task_id=task.task_id, attempt=attempt, sequence=sequence
                ),
            )
            observations = tuple(
                item["observation_id"] for item in response.get("observations", ())
            )
            success = response["status"] == ToolStatus.SUCCESS.value
            to_status = (
                DiscriminationTaskStatus.AMBIGUOUS
                if success and observations
                else DiscriminationTaskStatus.UNVERIFIABLE
            )
            outcome = (
                AttemptOutcome.AMBIGUOUS
                if success and observations
                else (AttemptOutcome.NO_DATA if success else AttemptOutcome.QUERY_FAILED)
            )
            sequence = len(self.transcript.events) + 1
            self._append_workspace_event(
                EventType.DISCRIMINATION_TASK_COMPLETED,
                DiscriminationTaskCompleted(
                    task_id=task.task_id,
                    attempt_id=attempt.attempt_id,
                    to_status=to_status,
                    observation_refs=observations,
                    tool_status=response["status"],
                    outcome=outcome,
                    reason_code=(
                        "SHADOW_OBSERVATION_NOT_SEMANTICALLY_ADJUDICATED"
                        if to_status == DiscriminationTaskStatus.AMBIGUOUS
                        else "QUERY_PRODUCED_NO_USABLE_DISCRIMINATOR"
                    ),
                    sequence=sequence,
                ),
            )

    def _record_active_discrimination_round(self, round_record: dict[str, Any]) -> None:
        """Enqueue legal exploration queries; select by utility, never delete the runners-up."""
        from rca_agent.investigation.query_strategy import (
            query_capabilities,
            scheduling_metadata,
        )

        assert self.__bootstrap is not None
        registry = self.__registry
        if registry is None:
            registry = EvidenceRegistry(signals=(), rules=())
        if callable(registry):
            registry = registry(
                self.__bootstrap,
                tuple(item.observation for item in self.__observation_ledger.values()),
            )
        registry = EvidenceRegistry.model_validate(registry)
        observations = tuple(item.observation for item in self.__observation_ledger.values())
        query_catalog = query_capabilities(
            self.__bootstrap, observations, registry, round_record["selected_actions"],
        )
        snapshot = self._record_capability_registry_snapshot(registry, query_catalog)
        draft = round_record["draft"]
        selected_bindings = round_record["selected_actions"]
        hypotheses = {item["hypothesis_id"]: item for item in draft["hypotheses"]}
        known_task_ids = {item.task_id for item in self.__investigation_workspace.tasks}
        created: list[DiscriminationTask] = []
        offered_actions: dict[str, CallToolAction] = {}

        for selected in selected_bindings:
            action = CallToolAction.model_validate(selected["action"])
            fingerprint = query_fingerprint(action.tool_name, action.request)
            offered_actions[fingerprint] = action

        need_bindings = [(need, selected) for need in draft["needs"]
                         for selected in ([s for s in selected_bindings
                                           if s["need_id"] == need["need_id"]] or [None])]
        for need, selected in need_bindings:
            hypothesis = hypotheses[need["hypothesis_id"]]
            candidate = self.__investigation_workspace.candidate_by_semantic_digest(
                candidate_semantic_digest(hypothesis["mechanism"])
            )
            if candidate is None:
                continue
            alternative = next((
                item for item in candidate.alternatives
                if _normalized_semantic_text(item.mechanism_summary)
                == _normalized_semantic_text(hypothesis["alternative"])
            ), None)
            if alternative is None:
                continue
            action = (
                CallToolAction.model_validate(selected["action"])
                if selected is not None else None
            )
            fingerprint = (
                query_fingerprint(action.tool_name, action.request)
                if action is not None else None
            )
            options = tuple(
                item for item in snapshot.capabilities
                if fingerprint is not None
                and item.query_action is not None
                and item.executable
                and item.entity_ref == selected["entity_ref"]
                and item.query_fingerprint == fingerprint
            )
            base_task_id = discrimination_task_id(
                self.episode_id,
                candidate.candidate_track_id,
                alternative.alternative_id,
                need["question"],
                need["expected_if_true"],
                need["expected_if_alternative"],
                query_fingerprint=fingerprint,
                binding_id=("binding-" + sha256_digest({"round": round_record["round"],
                            "need_id": need["need_id"], "query_ref": selected["query_ref"]})[:20]
                            if selected else None),
            )
            predecessor = next((
                item for item in reversed(self.__investigation_workspace.tasks)
                if item.source_round < round_record["round"] and (item.task_id == base_task_id or (
                    item.candidate_track_id == candidate.candidate_track_id
                    and item.alternative_id == alternative.alternative_id
                    and _normalized_semantic_text(item.question)
                    == _normalized_semantic_text(need["question"])
                    and _normalized_semantic_text(item.candidate_prediction)
                    == _normalized_semantic_text(need["expected_if_true"])
                    and _normalized_semantic_text(item.comparator_prediction)
                    == _normalized_semantic_text(need["expected_if_alternative"])
                ))
            ), None)
            task_id = (
                discrimination_task_id(
                    self.episode_id,
                    candidate.candidate_track_id,
                    alternative.alternative_id,
                    need["question"],
                    need["expected_if_true"],
                    need["expected_if_alternative"],
                    recurrence_round=round_record["round"],
                    query_fingerprint=fingerprint,
                    binding_id=("binding-" + sha256_digest({"round": round_record["round"],
                                "need_id": need["need_id"], "query_ref": selected["query_ref"]})[:20]
                                if selected else None),
                ) if predecessor is not None else base_task_id
            )
            if task_id in known_task_ids:
                continue
            known_task_ids.add(task_id)
            sequence = len(self.transcript.events) + 1
            task = DiscriminationTask(
                task_id=task_id,
                candidate_track_id=candidate.candidate_track_id,
                source_need_id=need["need_id"],
                source_round=round_record["round"],
                source_binding_id=("binding-" + sha256_digest({"round": round_record["round"],
                                   "need_id": need["need_id"], "query_ref": selected["query_ref"]})[:20]
                                   if selected else None),
                question=need["question"],
                alternative_id=alternative.alternative_id,
                observable_kind=(selected["signal"] if selected else "unresolved_need"),
                candidate_prediction=need["expected_if_true"],
                comparator_prediction=need["expected_if_alternative"],
                importance=(
                    DiscriminationImportance.PRIMARY
                    if selected is not None else DiscriminationImportance.SUPPLEMENTAL
                ),
                status=DiscriminationTaskStatus.OPEN,
                capability_options=options,
                legacy_choice_ref=(selected["query_ref"] if selected else None),
                legacy_query_ref=fingerprint,
                created_registry_digest=snapshot.registry_digest,
                last_evaluated_registry_digest=snapshot.registry_digest,
                unresolved_capability_reason=(
                    None if options else
                    "No exact executable Registry capability resolves this discriminator."
                ),
                supersedes_task_id=(predecessor.task_id if predecessor else None),
                created_sequence=sequence,
                last_transition_sequence=sequence,
                query_strategy_version=2,
                scheduling=(scheduling_metadata(
                    need, options[0], self.__bootstrap, observations,
                    SimpleNamespace(signals=snapshot.capabilities),
                    binding=selected if draft.get("contract_version") == 2 else None,
                ) if options else None),
            )
            self._append_workspace_event(EventType.DISCRIMINATION_TASK_CREATED, task)
            created.append(task)

        for task in created:
            if not task.capability_options:
                sequence = len(self.transcript.events) + 1
                self._append_workspace_event(
                    EventType.DISCRIMINATION_TASK_UNVERIFIABLE,
                    DiscriminationTaskUnverifiable(
                        task_id=task.task_id,
                        reason_code="NO_EXACT_CAPABILITY_BINDING",
                        reason=task.unresolved_capability_reason or "Capability unavailable.",
                        registry_digest=snapshot.registry_digest,
                        sequence=sequence,
                    ),
                )

        if self.__workspace_mode != "active":
            return
        self.__active_offered_query_refs.update(offered_actions)
        for task in created:
            if task.capability_options and task.legacy_query_ref is not None:
                self.__pending_active_actions[task.legacy_query_ref] = offered_actions[
                    task.legacy_query_ref
                ]
        self._select_pending_active_query()

    def _cancel_active_query(self, task: DiscriminationTask, reason: str) -> None:
        self.__pending_active_actions.pop(task.legacy_query_ref, None)
        self._append_workspace_event(EventType.INQUIRY_QUERY_CANCELLED, {
            "task_id": task.task_id, "query_fingerprint": task.legacy_query_ref,
            "reason_code": reason, "sequence": len(self.transcript.events) + 1,
        })

    def _select_pending_active_query(self) -> None:
        if self.__active_selected_action is not None or self.__active_selected_task_id is not None:
            return
        attempted = {event.payload["query_ref"] for event in self.transcript.events
                     if event.event_type == EventType.TOOL_REQUEST}
        for task in self.__investigation_workspace.tasks:
            if task.status == DiscriminationTaskStatus.OPEN and task.scheduling is not None:
                if task.legacy_query_ref in attempted:
                    self._cancel_active_query(task, "DUPLICATE_EXECUTED_QUERY")
                elif task.capability_options[0].query_cost > self.__budget_remaining:
                    self._cancel_active_query(task, "QUERY_BUDGET_UNAVAILABLE")
        matrix = self.__investigation_workspace.discrimination_matrix()
        entry = next((
            item for item in matrix.entries
            if item.query_fingerprint in self.__pending_active_actions
        ), None)
        if entry is None:
            return
        task = self.__investigation_workspace.task(entry.task_id)
        assert task is not None
        round_record = next(item for item in self.__inquiry_rounds
                            if item["round"] == task.source_round)
        offered_refs = tuple(sorted(query_fingerprint(
            CallToolAction.model_validate(item["action"]).tool_name,
            CallToolAction.model_validate(item["action"]).request,
        ) for item in round_record["selected_actions"]))
        sequence = len(self.transcript.events) + 1
        selection = DiscriminationActiveSelection(
            selection_id="active-selection-" + sha256_digest({
                "episode_id": self.episode_id,
                "sequence": sequence,
                "matrix_digest": matrix.matrix_digest,
            })[:20],
            source_round=round_record["round"],
            matrix_digest=matrix.matrix_digest,
            offered_query_fingerprints=offered_refs,
            selected_task_id=entry.task_id,
            selected_capability_ref=entry.capability_ref,
            selected_query_fingerprint=entry.query_fingerprint,
            suppressed_query_fingerprints=(),
            strategy_version=2,
            pending_query_fingerprints=tuple(sorted(
                ref for ref in self.__pending_active_actions if ref != entry.query_fingerprint
            )),
            reason_codes=(
                "ACTIVE_MATRIX_SELECTION",
                "MODEL_OFFERED_QUERY_ONLY",
                "PENDING_QUERIES_RETAINED",
            ),
            sequence=sequence,
        )
        self._append_workspace_event(
            EventType.DISCRIMINATION_ACTIVE_SELECTION_RECORDED, selection
        )
        sequence = len(self.transcript.events) + 1
        attempt = DiscriminationAttempt(
            attempt_id=discrimination_attempt_id(
                self.episode_id, task.task_id, entry.capability_ref, entry.query_fingerprint
            ),
            task_id=task.task_id,
            capability_ref=entry.capability_ref,
            query_ref=entry.query_fingerprint,
            query_fingerprint=entry.query_fingerprint,
            query_cost=entry.query_cost,
            independent_group=entry.independent_group,
            registry_digest=task.last_evaluated_registry_digest,
            selection_reason_codes=entry.reason_codes,
            selected_sequence=sequence,
        )
        self._append_workspace_event(
            EventType.DISCRIMINATION_TASK_SELECTED,
            DiscriminationTaskSelected(
                task_id=task.task_id,
                selection_mode="active",
                attempt=attempt,
                sequence=sequence,
            ),
        )
        self.__active_selected_action = self.__pending_active_actions.pop(entry.query_fingerprint)
        # All declared bindings share this physical execution, selected before any result.
        for shared in self.__investigation_workspace.tasks:
            if (shared.task_id == task.task_id or shared.status != DiscriminationTaskStatus.OPEN
                    or shared.legacy_query_ref != entry.query_fingerprint
                    or not shared.capability_options):
                continue
            option = shared.capability_options[0]
            sequence = len(self.transcript.events) + 1
            self._append_workspace_event(EventType.DISCRIMINATION_TASK_SELECTED,
                DiscriminationTaskSelected(task_id=shared.task_id, selection_mode="active",
                    attempt=DiscriminationAttempt(
                        attempt_id=discrimination_attempt_id(self.episode_id, shared.task_id,
                                                           option.capability_ref, entry.query_fingerprint),
                        task_id=shared.task_id, capability_ref=option.capability_ref,
                        query_ref=entry.query_fingerprint, query_fingerprint=entry.query_fingerprint,
                        query_cost=option.query_cost, independent_group=option.independent_group,
                        registry_digest=shared.last_evaluated_registry_digest,
                        selection_reason_codes=("SHARED_DECLARED_QUERY_BINDING",),
                        selected_sequence=sequence), sequence=sequence))
        self.__active_selected_attempt_id = attempt.attempt_id
        self.__active_selected_task_id = task.task_id

    def _workspace_transition_id(
        self, sequence: int, track_id: str, transition_type: CandidateTransitionType,
    ) -> str:
        return candidate_transition_id(
            self.episode_id, sequence, track_id, transition_type
        )

    def should_discover(self, *, stopping: bool = False) -> bool:
        if (self.status != EpisodeStatus.ACTIVE
                or (self.__discovery is None and self.__mechanism_formalizer is None)
                or self.__discovery_attempted or self.__rounds >= self.__max_rounds
                or self.__budget_remaining <= 0 or not self.__observation_ledger
                or not isinstance(self.__decision_provider, CoreDecisionProvider)):
            return False
        preview = self.preview_core_assessment()
        if preview.overall_decision.status in {
            DecisionStatus.CONFIRMED_ROOT_CAUSE, DecisionStatus.DISCOVERED_VERIFIED,
        }:
            return False
        resolved = {h for c in preview.candidate_assessments for h in c.hypothesis_ids}
        if any(p.hypothesis_id not in resolved for p in self.__proposals):
            return True
        return bool(self.__proposals) and (stopping or all(
            c.evidence_status.value == "contradicted" for c in preview.candidate_assessments
        ))

    def discover_candidates(self, *, inquiry_handoff: dict[str, Any] | None = None) -> dict[str, Any]:
        """One bounded model call; only the Runner registers and binds compiled candidates."""
        from queue import Empty, Queue
        from threading import Thread

        self._require_status(EpisodeStatus.ACTIVE)
        if not self.should_discover(stopping=True):
            return {"status": "skipped", "registered": 0}
        self.__discovery_attempted = True
        self.__rounds += 1
        registry = self.__registry
        assert registry is not None and self.__bootstrap is not None
        if callable(registry):
            try:
                registry = registry(self.__bootstrap, tuple(
                    r.observation for r in self.__observation_ledger.values()
                ))
                registry = EvidenceRegistry.model_validate(registry)
            except (ValueError, TypeError):
                self.transcript.append(EventType.DISCOVERY_STARTED, {"registry_unavailable": True})
                result = {"status": "registry_unavailable", "registered": 0}
                self.transcript.append(EventType.DISCOVERY_COMPLETED, result)
                return result
        self.__resolved_registry = registry
        proposals = {p.hypothesis_id: p for p in self.__proposals}
        context = {
            "alert_summary": self.__bootstrap.alert_summary,
            "hypotheses": [p.model_dump(mode="json") for p in proposals.values()],
            "signals": [{"signal": s.signal, "entity_ref": s.entity_ref,
                         "data_type": s.data_type, "unit": s.unit, "description": s.description,
                         "available": self._signal_executable(s)} for s in registry.signals],
            "rules": [{"rule_id": r.rule_id, "signal": r.signal,
                       "phenomenon": r.description, "aggregation": r.aggregation,
                       "support_eligible": r.support_eligible}
                      for r in registry.rules],
            "entity_graph": self._discovery_entity_graph(registry),
            "observations": [r.observation.model_dump(mode="json")
                             for r in self.__observation_ledger.values()
                             if any(s.accepts(r.observation) for s in registry.signals)],
            "existing_candidates": [
                {"candidate_id": c.candidate_id, "mechanism": c.fault_type}
                for c in self.__decision_provider.catalog(self.__bootstrap.alert_entity.entity_ref)
            ],
        }
        validated_handoff = None
        if inquiry_handoff is not None:
            from rca_agent.langchain_acquisition import InquiryHandoff

            handoff = InquiryHandoff.model_validate(inquiry_handoff)
            if self.__inquiry_rounds:
                from rca_agent.langchain_acquisition import inquiry_candidate_records

                if self.__inquiry_loop_mode == "batched":
                    if handoff.model_dump(mode="json") != self.inquiry_loop_handoff():
                        raise ValueError("batched handoff differs from the reviewed Workspace")
                elif handoff.candidate_history != inquiry_candidate_records(self.__inquiry_rounds):
                    raise ValueError("inquiry handoff omitted a recorded candidate")
            completed = {
                event.payload["query_ref"]: event.payload
                for event in self.transcript.events
                if event.event_type == EventType.TOOL_RESPONSE
            }
            from rca_agent.investigation.contract_enforcement import execution_binding_responses
            from rca_agent.investigation.inquiry_contract import validate_handoff_links
            validate_handoff_links(handoff, self.__inquiry_rounds, completed,
                binding_responses=(execution_binding_responses(self.transcript.events)
                                   if self.__engineering_revision == 2 else None))
            for link in handoff.query_links:
                response = completed.get(link.query_ref)
                if (response is None or response["status"] != link.status
                        or set(link.observation_refs) != {
                            item["observation_id"] for item in response["observations"]
                        }):
                    raise ValueError("inquiry handoff references an unissued query result")
            context["inquiry_handoff"] = handoff.model_dump(mode="json")
            validated_handoff = handoff
        default_dispositions = [
            {"hypothesis_id": item["hypothesis_id"], "disposition": "unreported",
             "reason": "This discovery interface does not report inquiry dispositions."}
            for item in context.get("inquiry_handoff", {}).get("hypotheses", ())
        ]
        if self.__workspace_mode == "shadow":
            registry_snapshot = self._record_capability_registry_snapshot(registry)
            self._record_discrimination_shadow(validated_handoff, registry_snapshot)
        handoff_contract = (
            self.__verification_planner is not None
            and self.__mechanism_formalizer is None
            and bool(context.get("inquiry_handoff", {}).get("hypotheses"))
        )
        if handoff_contract:
            context["handoff_contract"] = {
                "version": self.__bound_handoff_version if getattr(self.__discovery, "supports_bound_candidates", False) else 4,
                "candidate_semantics_are_immutable": True,
            }
        self.transcript.append(EventType.DISCOVERY_STARTED, {
            "context_digest": sha256_digest(context),
            "inquiry_handoff": context.get("inquiry_handoff"),
            **({"handoff_contract_version": context["handoff_contract"]["version"]} if handoff_contract else {}),
        })
        if self.__verification_planner is not None:
            return self._discover_two_stage(context, registry, proposals)
        mailbox: Queue = Queue(maxsize=1)
        generator = self.__discovery
        assert generator is not None

        def invoke():
            try:
                mailbox.put((True, generator.discover(deepcopy(context))))
            except Exception as exc:  # noqa: BLE001 - failure is sanitized and bounded below
                failure = {"error_type": type(exc).__name__}
                raw_payload = getattr(exc, "raw_payload", None)
                if isinstance(raw_payload, (dict, list)):
                    failure["raw_payload"] = raw_payload
                mailbox.put((False, failure))

        Thread(target=invoke, daemon=True).start()
        registered = 0
        aligned = 0
        rejections = []
        try:
            ok, raw = mailbox.get(timeout=self.__discovery_timeout)
            if not ok:
                self.transcript.append(EventType.DISCOVERY_PAYLOAD_REJECTED, raw)
                result = {
                    "status": "failed", "registered": 0,
                    "error_type": raw["error_type"],
                }
                self.transcript.append(EventType.DISCOVERY_COMPLETED, result)
                return result
            payload = DiscoveryPayload.model_validate(raw)
            seen = set()
            for draft_index, draft in enumerate(payload.hypotheses):
                evaluated_draft = draft
                initial_error = None
                repair_actions: tuple[str, ...] = ()
                try:
                    original = proposals.get(draft.source_hypothesis_id)
                    discovery_key = (
                        draft.source_hypothesis_id,
                        draft.mechanism.strip().casefold(),
                    )
                    if original is None or discovery_key in seen:
                        raise ValueError("unknown or duplicate source hypothesis")
                    seen.add(discovery_key)
                    referenced_entities = {
                        draft.causal_entity_ref or original.causal_entity_ref,
                        *(prediction.target_entity_ref
                          for prediction in draft.predictions
                          if prediction.target_entity_ref is not None),
                    }
                    if not referenced_entities <= self.__issued_entities:
                        raise ValueError("dynamic hypothesis references an unissued entity")
                    compile_kwargs = {
                        "discovery_query_refs": tuple(sorted({
                            r.query_ref for r in self.__observation_ledger.values()
                        })),
                        "discovery_observation_refs": tuple(sorted(self.__observation_ledger)),
                        "discovery_fact_digests": tuple(sorted({
                            observation_fact_digest(r.observation)
                            for r in self.__observation_ledger.values()
                        })),
                    }
                    try:
                        compiled = registry.compile(
                            evaluated_draft, original.causal_entity_ref, **compile_kwargs
                        )
                    except ValueError as exc:
                        initial_error = str(exc)
                        evaluated_draft, repair_actions = repair_mechanical_contract(draft)
                        if not repair_actions:
                            raise
                        compiled = registry.compile(
                            evaluated_draft, original.causal_entity_ref, **compile_kwargs
                        )
                    if not all(self._signal_executable(s) for s in compiled.signals):
                        raise ValueError("query recipe is not executable in this episode")
                    existing = next((c for c in self.__decision_provider.catalog(
                                     compiled.candidate.causal_entity)
                                     if evaluated_draft.mechanism.casefold() in {
                                         c.candidate_id.casefold(), c.fault_type.casefold()
                                     } and c.family == draft.family), None)
                    if existing is not None:
                        required = {c.signal for c in existing.conditions}
                        if not required <= {s.signal for s in compiled.signals}:
                            raise ValueError("catalog alignment lacks executable signal recipes")
                        hypothesis_id = "h-aligned-" + compiled.version[:20]
                        self.transcript.append(EventType.CATALOG_CANDIDATE_ALIGNED, {
                            "candidate_id": existing.candidate_id,
                            "hypothesis_id": hypothesis_id,
                            "source_hypothesis_id": original.hypothesis_id,
                        })
                        self._submit_hypothesis(HypothesisProposal(
                            hypothesis_id=hypothesis_id, candidate_id=existing.candidate_id,
                            mechanism=existing.fault_type,
                            causal_entity_ref=compiled.candidate.causal_entity,
                            rationale=evaluated_draft.causal_explanation,
                        ))
                        queries = {query_fingerprint(s.query.tool_name, s.query.request): s.query
                                   for s in compiled.signals}
                        self.__verification_queries.extend((hypothesis_id, q) for q in queries.values())
                        aligned += 1
                        self._record_discovery_draft(
                            draft_index, draft, evaluated_draft, initial_error,
                            repair_actions, "accepted", None, "catalog_aligned",
                        )
                        continue
                    # Preserve the original hypothesis and its evidence; discovery gets its own ID.
                    hypothesis_id = "h-discovered-" + compiled.version[:20]
                    rebound = compiled.model_copy(update={"source_hypothesis_id": hypothesis_id})
                    self.__decision_provider.register_dynamic(rebound)
                    self.transcript.append(EventType.DYNAMIC_CANDIDATE_REGISTERED, {
                        "compiled": rebound.model_dump(mode="json"),
                        "source_hypothesis_id": original.hypothesis_id,
                    })
                    self._submit_hypothesis(HypothesisProposal(
                        hypothesis_id=hypothesis_id, candidate_id=compiled.candidate.candidate_id,
                        mechanism=evaluated_draft.mechanism,
                        causal_entity_ref=compiled.candidate.causal_entity,
                        rationale=evaluated_draft.causal_explanation,
                    ))
                    candidate = compiled.candidate
                    self.__dynamic_options.append({
                        "candidate_id": candidate.candidate_id, "fault_type": candidate.fault_type,
                        "family": candidate.family.value,
                        "required_signals": tuple(s.signal for s in compiled.signals),
                        "required_capabilities": tuple(candidate.required_capabilities),
                    })
                    queries = {query_fingerprint(s.query.tool_name, s.query.request): s.query
                               for s in compiled.signals}
                    self.__verification_queries.extend((hypothesis_id, q) for q in queries.values())
                    registered += 1
                    self._record_discovery_draft(
                        draft_index, draft, evaluated_draft, initial_error,
                        repair_actions, "accepted", None, "dynamic_registered",
                    )
                except ValueError as exc:
                    final_error = (str(exc) if type(exc) is ValueError
                                   else "invalid compiled candidate contract")
                    rejections.append(final_error)
                    self._record_discovery_draft(
                        draft_index, draft, evaluated_draft, initial_error,
                        repair_actions, "rejected", final_error, "none",
                    )
            result = {"status": "completed", "registered": registered,
                      "aligned": aligned, "rejections": rejections,
                      "inquiry_dispositions": default_dispositions}
        except Empty:
            result = {"status": "timeout", "registered": 0}
        except (ValueError, TypeError):
            result = {"status": "failed", "registered": 0}
        self.transcript.append(EventType.DISCOVERY_COMPLETED, result)
        return result

    def _discover_two_stage(
        self,
        context: dict[str, Any],
        registry: EvidenceRegistry,
        proposals: dict[str, HypothesisProposal],
    ) -> dict[str, Any]:
        mechanism_context = {
            key: deepcopy(value)
            for key, value in context.items()
            if key not in {"signals", "rules"}
        }
        for node in mechanism_context["entity_graph"]["nodes"]:
            node.pop("available_rule_ids", None)
        bound = mechanism_context.get("handoff_contract", {}).get("version") in {6, 7}
        if bound:
            ok, raw = True, self._discover_bound_candidates(mechanism_context, proposals)
        elif self.__mechanism_formalizer is not None:
            ok, raw = self._invoke_formalizer(mechanism_context, proposals)
        else:
            assert self.__discovery is not None
            ok, raw = self._invoke_component_method(
                self.__discovery, "discover", mechanism_context
            )
        if not ok and raw.get("error_type") == "DiscoveryPayloadError":
            repaired = self._repair_discovery_format(raw.get("raw_payload"), proposals)
            if repaired is not None:
                ok, raw = True, repaired
        if not ok:
            self.transcript.append(EventType.DISCOVERY_PAYLOAD_REJECTED, {
                "stage": "mechanism_discovery", **raw,
            })
            result = {"status": "failed", "registered": 0, "error_type": raw["error_type"]}
            self.transcript.append(EventType.DISCOVERY_COMPLETED, result)
            return result
        try:
            payload = MechanismDiscoveryPayload.model_validate(raw)
        except (ValueError, TypeError) as exc:
            repaired = self._repair_discovery_format(raw, proposals)
            if repaired is None:
                failure = {"stage": "mechanism_discovery", "error_type": type(exc).__name__}
                self.transcript.append(EventType.DISCOVERY_PAYLOAD_REJECTED, failure)
                result = {"status": "failed", "registered": 0, "error_type": type(exc).__name__}
                self.transcript.append(EventType.DISCOVERY_COMPLETED, result)
                return result
            payload = repaired

        if not bound:
            payload = self._complete_identified_mechanisms(payload, mechanism_context, proposals)
        if mechanism_context.get("handoff_contract") and not bound:
            payload = self._validate_discovery_handoff(payload, mechanism_context)

        inquiry_ids = {
            item["hypothesis_id"]
            for item in context.get("inquiry_handoff", {}).get("hypotheses", ())
        }
        model_dispositions = {
            item.hypothesis_id: item for item in payload.inquiry_dispositions
            if item.hypothesis_id in inquiry_ids
        }
        inquiry_dispositions = [
            (model_dispositions[item].model_dump(mode="json")
             if item in model_dispositions else {
                 "hypothesis_id": item, "disposition": "unreported",
                 "reason": "The discovery model did not provide a disposition.",
             })
            for item in sorted(inquiry_ids)
        ]

        frozen: dict[str, FrozenMechanism] = {}
        valid_drafts = []
        from rca_agent.investigation.contract_enforcement import draft_rejection, freeze_input
        seen = set()
        rejection_codes: list[str] = []
        for draft_index, draft in enumerate(payload.mechanisms):
            self.transcript.append(EventType.MECHANISM_DRAFT_GENERATED, {
                "draft_index": draft_index, "draft": draft.model_dump(mode="json"),
            })
            error_code = draft_rejection(draft, proposals, self.__issued_entities, seen,
                                         reject_self=self.__bound_handoff_version == 7)
            if error_code is not None:
                rejection_codes.append(error_code)
                self.transcript.append(EventType.MECHANISM_DRAFT_REJECTED, {
                    "draft_index": draft_index, "error_code": error_code,
                })
                continue
            valid_drafts.append((draft_index, draft))

        for draft_index, draft in valid_drafts:
            frozen_draft = freeze_input(draft, valid_drafts)
            item = freeze_mechanism(frozen_draft)
            frozen[item.mechanism_id] = item
            self.__frozen_mechanisms[item.mechanism_id] = item
            self.transcript.append(EventType.MECHANISM_DRAFT_FROZEN, {
                "draft_index": draft_index, "mechanism": item.model_dump(mode="json"),
            })

        if not frozen:
            result = {
                "status": "completed", "registered": 0, "aligned": 0,
                "rejections": rejection_codes,
                "inquiry_dispositions": inquiry_dispositions,
                "planner_attempts": 0,
            }
            self.transcript.append(EventType.DISCOVERY_COMPLETED, result)
            return result

        handles = registry.capability_handles()
        planner_base = {
            "frozen_mechanisms": [item.model_dump(mode="json") for item in frozen.values()],
            "capabilities": [item.model_dump(mode="json") for item in handles],
            "observations": context["observations"],
            "unavailable_signals": {
                signal: reason for capability in self.__bootstrap.capabilities
                for signal, reason in capability.unavailable_signals.items()
            },
            "unsupported_derivations": {
                signal: reason for capability in self.__bootstrap.capabilities
                for signal, reason in capability.unsupported_derivations.items()
            },
        }
        compile_kwargs = {
            "discovery_query_refs": tuple(sorted({
                record.query_ref for record in self.__observation_ledger.values()
            })),
            "discovery_observation_refs": tuple(sorted(self.__observation_ledger)),
            "discovery_fact_digests": tuple(sorted({
                observation_fact_digest(record.observation)
                for record in self.__observation_ledger.values()
            })),
        }
        registered = 0
        aligned = 0
        accepted: set[str] = set()

        first_payload, first_failure = self._invoke_planner({
            **planner_base, "attempt": 1, "errors": [],
        })
        if first_payload is None:
            code = first_failure or "PLANNER_FAILED"
            rejection_codes.append(code)
            initial_failures = {
                mechanism_id: {"mechanism_id": mechanism_id, "error_code": code}
                for mechanism_id in frozen
            }
            for mechanism_id in frozen:
                self._reject_plan(mechanism_id, 1, code)
            self.transcript.append(EventType.VERIFICATION_PLAN_RETRIED, {
                "mechanism_ids": sorted(frozen),
                "errors": [initial_failures[value] for value in sorted(frozen)],
            })
            retry_payload, retry_failure = self._invoke_planner({
                **planner_base,
                "attempt": 2,
                "errors": [initial_failures[value] for value in sorted(frozen)],
                "previous_plans": [],
            })
            if retry_payload is not None:
                retry_failed, counts = self._compile_plans(
                    retry_payload,
                    frozen,
                    registry,
                    proposals,
                    compile_kwargs,
                    accepted,
                    attempt=2,
                )
                registered += counts[0]
                aligned += counts[1]
                rejection_codes.extend(
                    item["error_code"] for item in retry_failed.values()
                )
            else:
                rejection_codes.append(retry_failure or "PLANNER_RETRY_FAILED")
            result = {
                "status": "completed", "registered": registered, "aligned": aligned,
                "rejections": rejection_codes, "planner_attempts": 2,
                "inquiry_dispositions": inquiry_dispositions,
            }
            self.transcript.append(EventType.DISCOVERY_COMPLETED, result)
            return result
        failed, counts = self._compile_plans(
            first_payload, frozen, registry, proposals, compile_kwargs, accepted, attempt=1,
        )
        registered += counts[0]
        aligned += counts[1]
        rejection_codes.extend(item["error_code"] for item in failed.values())

        planner_attempts = 1
        if failed:
            planner_attempts = 2
            retry_ids = sorted(failed)
            self.transcript.append(EventType.VERIFICATION_PLAN_RETRIED, {
                "mechanism_ids": retry_ids,
                "errors": [failed[value] for value in retry_ids],
            })
            retry_context = {
                **planner_base,
                "frozen_mechanisms": [
                    frozen[value].model_dump(mode="json") for value in retry_ids
                ],
                "attempt": 2,
                "errors": [failed[value] for value in retry_ids],
                "previous_plans": [
                    failed[value]["plan"] for value in retry_ids if failed[value].get("plan")
                ],
            }
            retry_payload, retry_failure = self._invoke_planner(retry_context)
            if retry_payload is None:
                rejection_codes.append(retry_failure or "PLANNER_RETRY_FAILED")
            else:
                retry_failed, counts = self._compile_plans(
                    retry_payload,
                    {value: frozen[value] for value in retry_ids},
                    registry,
                    proposals,
                    compile_kwargs,
                    accepted,
                    attempt=2,
                )
                registered += counts[0]
                aligned += counts[1]
                rejection_codes.extend(
                    item["error_code"] for item in retry_failed.values()
                )
        result = {
            "status": "completed", "registered": registered, "aligned": aligned,
            "rejections": rejection_codes, "planner_attempts": planner_attempts,
            "inquiry_dispositions": inquiry_dispositions,
        }
        self.transcript.append(EventType.DISCOVERY_COMPLETED, result)
        return result

    def _discover_bound_candidates(self, context, proposals):
        from rca_agent.investigation.bound_formalization import (
            BoundCandidateResponse,
            BoundCandidateResponseV7,
            admit_novel,
            bind_candidate,
            bound_payload,
            candidate_binding,
            compatible_classes,
        )
        from rca_agent.investigation.discovery_handoff import pending_formalizations

        handoff = context["inquiry_handoff"]
        records = {}
        source = min(proposals)
        version = context["handoff_contract"]["version"]
        for index, candidate in enumerate(handoff["hypotheses"]):
            binding = candidate_binding(candidate, source, version=version, handoff=handoff,
                                        revision=self.__engineering_revision)
            if index >= 4:
                record = {"binding": binding, "attempt": 0, "response": None,
                          "reason_code": "FORMALIZATION_CAPACITY"}
                records[candidate["hypothesis_id"]] = record
                self.transcript.append(EventType.BOUND_CANDIDATE_FORMALIZATION_RECORDED, record)
                continue
            feedback = None
            for attempt in (1, 2):
                call_context = {
                    "binding": (binding if version == 7 else
                                {k: v for k, v in binding.items() if k != "alternative"}),
                    "type_compatibility_hints": compatible_classes(binding["mechanism"]),
                    "entity_graph": context["entity_graph"], "feedback": feedback,
                }
                ok, response = self._invoke_component_method(
                    self.__discovery, "formalize_candidate", call_context
                )
                record = {"binding": binding, "attempt": attempt, "response": None,
                          "reason_code": response["error_type"] if not ok else "INVALID_TYPED_DETAILS"}
                typed = None
                if ok:
                    try:
                        schema = BoundCandidateResponseV7 if version == 7 else BoundCandidateResponse
                        typed = schema.model_validate(response)
                        bind_candidate(binding, typed, self.__issued_entities, check_class=True)
                        record.update(response=typed.model_dump(mode="json"), reason_code=typed.reason_code)
                    except ValueError as exc:
                        # Keep the rejected provider shape only as diagnostics;
                        # `response` stays null so Replay cannot treat an
                        # unbound candidate as admitted.  Feed the exact host
                        # rejection code into the bounded retry instead of the
                        # previous generic INVALID_TYPED_DETAILS message.
                        if "typed" in locals():
                            record["rejected_response"] = typed.model_dump(mode="json")
                        record["reason_code"] = str(exc) or "INVALID_TYPED_DETAILS"
                records[candidate["hypothesis_id"]] = record
                self.transcript.append(EventType.BOUND_CANDIDATE_FORMALIZATION_RECORDED, record)
                if record["response"] is not None:
                    break
                feedback = {"code": record["reason_code"],
                            "compatible_classes": compatible_classes(binding["mechanism"]),
                            "instruction": "Return valid typed details for the SAME candidate using an issued entity; do not change its cause."}
        base = bound_payload(handoff, records, self.__issued_entities, check_class=True)
        novel = None
        novel_failure = None
        if len(base.mechanisms) < 4:
            new_context = {**context, "new_discovery_only": True,
                           "immutable_candidates": handoff["hypotheses"],
                           "already_formalized": [m.model_dump(mode="json") for m in base.mechanisms]}
            new_context.pop("inquiry_handoff", None)
            new_context.pop("handoff_contract", None)
            ok, raw = self._invoke_component_method(
                self.__discovery, "discover", new_context
            )
            if ok:
                try:
                    novel = MechanismDiscoveryPayload.model_validate(raw)
                except ValueError:
                    novel_failure = "INVALID_NEW_DISCOVERY"
            else:
                novel_failure = raw["error_type"]
        facts = [r.observation.model_dump(mode="json") for r in self.__observation_ledger.values()]
        payload = admit_novel(base, novel, handoff, facts, set(proposals), self.__issued_entities)
        self.transcript.append(EventType.DISCOVERY_HANDOFF_VALIDATED, {
            "version": version, "initial_payload": base.model_dump(mode="json"),
            "new_discovery_payload": novel.model_dump(mode="json") if novel is not None else None,
            "new_discovery_failure": novel_failure,
            "payload": payload.model_dump(mode="json"), "outcome": "bound_candidates",
            "pending_formalizations": pending_formalizations(payload, handoff),
        })
        return payload

    def _validate_discovery_handoff(self, payload, context):
        from rca_agent.investigation.discovery_handoff import (
            diagnostic_errors,
            merge_partial_handoff,
            pending_formalizations,
            quarantine_handoff,
            repair_errors,
            validate_handoff,
        )

        handoff = context["inquiry_handoff"]
        observations = [r.observation.model_dump(mode="json")
                        for r in self.__observation_ledger.values()]
        errors = validate_handoff(payload, handoff, observations)
        initial = payload
        retry = None
        retry_failure = None
        retry_errors = ()
        if errors:
            self.transcript.append(EventType.DISCOVERY_HANDOFF_REPAIR_REQUESTED, {
                "version": 4, "attempt": 1,
                "initial_payload": initial.model_dump(mode="json"),
                "errors": diagnostic_errors(errors),
                "pending_formalizations": pending_formalizations(initial, handoff),
            })
            repair_context = deepcopy(context)
            repair_context["handoff_repair"] = {
                "attempt": 1, "initial_payload": initial.model_dump(mode="json"),
                "errors": diagnostic_errors(errors),
                "pending_formalizations": pending_formalizations(initial, handoff),
                "safe_projection": quarantine_handoff(
                    initial, handoff, observations, recover_exact=True,
                ).model_dump(mode="json"),
                "instructions": (
                    "Keep all safe_projection mechanisms byte-for-byte. Correct each listed "
                    "disposition: kept links the exact ORIGINAL text, not its alternative. "
                    "original_disposition is null unless disposition is revised. Use only "
                    "observed/derived nonmissing references; missing evidence means deferred. "
                    "Do not relabel or remove a safe original mechanism."
                ),
            }
            ok, raw = self._invoke_component_method(
                self.__discovery, "discover", repair_context
            )
            if ok:
                try:
                    retry = MechanismDiscoveryPayload.model_validate(raw)
                    retry_errors = repair_errors(initial, retry, handoff, observations, recover_exact=True)
                except (ValueError, TypeError) as exc:
                    retry_failure = type(exc).__name__
            else:
                retry_failure = raw["error_type"]
            payload = (
                retry if retry is not None and not retry_errors
                else merge_partial_handoff(initial, retry, handoff, observations)
            )
        self.transcript.append(EventType.DISCOVERY_HANDOFF_VALIDATED, {
            "version": 4, "initial_payload": initial.model_dump(mode="json"),
            "initial_errors": diagnostic_errors(errors),
            "retry_payload": retry.model_dump(mode="json") if retry is not None else None,
            "retry_errors": diagnostic_errors(retry_errors), "retry_failure": retry_failure,
            "payload": payload.model_dump(mode="json"),
            "pending_formalizations": pending_formalizations(payload, handoff),
            "outcome": ("accepted" if not errors else "repaired"
                        if retry is not None and not retry_errors else "partially_repaired"
                        if payload != quarantine_handoff(initial, handoff, observations, recover_exact=True)
                        else "quarantined"),
        })
        return payload

    def _complete_identified_mechanisms(
        self,
        payload: MechanismDiscoveryPayload,
        mechanism_context: dict[str, Any],
        proposals: dict[str, HypothesisProposal],
    ) -> MechanismDiscoveryPayload:
        """Allow one semantics-preserving retry for mechanisms already named by Discovery."""
        submitted = {
            (item.source_hypothesis_id, item.mechanism.strip().casefold())
            for item in payload.mechanisms
        }
        dispositions = {
            item.hypothesis_id: item for item in payload.inquiry_dispositions
        }
        inquiry_ids = {
            item["hypothesis_id"]
            for item in mechanism_context.get("inquiry_handoff", {}).get("hypotheses", ())
        }
        allowed: list[IdentifiedMechanismHint] = []
        seen: set[tuple[str, str]] = set()
        for hint in payload.identified_but_unsubmitted_mechanisms:
            key = (hint.source_hypothesis_id, hint.mechanism.strip().casefold())
            disposition = dispositions.get(hint.inquiry_hypothesis_id)
            # The host never infers a root cause from prose. The model must copy the exact
            # phrase it already wrote into a typed hint tied to an existing source proposal.
            if (
                key in submitted
                or key in seen
                or hint.source_hypothesis_id not in proposals
                or disposition is None
                or (inquiry_ids and hint.inquiry_hypothesis_id not in inquiry_ids)
                or hint.mechanism.strip().casefold()
                not in disposition.reason.strip().casefold()
            ):
                continue
            seen.add(key)
            allowed.append(hint)
        remaining = max(0, 4 - len(payload.mechanisms))
        allowed = allowed[:remaining]
        if not allowed:
            return payload

        initial_payload = payload.model_dump(mode="json")
        allowed_payload = [item.model_dump(mode="json") for item in allowed]
        self.transcript.append(EventType.MECHANISM_COMPLETION_REQUESTED, {
            "attempt": 1,
            "initial_payload": initial_payload,
            "initial_payload_digest": sha256_digest(initial_payload),
            "allowed_hints": allowed_payload,
        })
        completion_context = deepcopy(mechanism_context)
        completion_context["completion_request"] = {
            "mode": "mechanism_completion_only",
            "allowed_hints": allowed_payload,
            "initial_discovery": initial_payload,
            "constraints": [
                "keep every allowed source_hypothesis_id and mechanism text unchanged",
                "return no mechanism that is absent from allowed_hints",
                "add no new causal direction",
            ],
        }
        ok, raw = self._invoke_component_method(
            self.__discovery, "discover", completion_context
        )
        if not ok:
            self.transcript.append(EventType.MECHANISM_COMPLETION_REJECTED, {
                "attempt": 1,
                "error_type": raw["error_type"],
            })
            return payload
        try:
            completion = MechanismDiscoveryPayload.model_validate(raw)
        except (ValueError, TypeError) as exc:
            self.transcript.append(EventType.MECHANISM_COMPLETION_REJECTED, {
                "attempt": 1,
                "error_type": type(exc).__name__,
            })
            return payload

        allowed_keys = {
            (item.source_hypothesis_id, item.mechanism.strip().casefold())
            for item in allowed
        }
        accepted = []
        completion_seen: set[tuple[str, str]] = set()
        rejections = []
        for draft in completion.mechanisms:
            key = (draft.source_hypothesis_id, draft.mechanism.strip().casefold())
            if key not in allowed_keys or key in completion_seen:
                rejections.append("MECHANISM_COMPLETION_SEMANTIC_DRIFT")
                continue
            completion_seen.add(key)
            accepted.append(draft)
        self.transcript.append(EventType.MECHANISM_COMPLETION_COMPLETED, {
            "attempt": 1,
            "payload": completion.model_dump(mode="json"),
            "accepted_mechanisms": [item.model_dump(mode="json") for item in accepted],
            "rejections": rejections,
        })
        return payload.model_copy(update={
            "mechanisms": (*payload.mechanisms, *accepted),
        })

    def _repair_discovery_format(
        self,
        raw: Any,
        proposals: dict[str, HypothesisProposal],
    ) -> MechanismDiscoveryPayload | None:
        """One optional AI call only when raw Discovery causal identity is safe."""
        if self.__discovery_format_formalizer is None:
            return None
        context: DiscoveryFormatRepairContext | None = repair_context_from_raw(
            raw,
            proposal_ids=set(proposals),
            issued_entities=self.__issued_entities,
        )
        if context is None:
            return None
        self.transcript.append(EventType.DISCOVERY_FORMAT_REPAIR_REQUESTED, {
            "attempt": 1,
            "context": context.model_dump(mode="json"),
            "context_digest": sha256_digest(context.model_dump(mode="json")),
        })
        ok, response = self._invoke_component_method(
            self.__discovery_format_formalizer, "repair", context
        )
        if not ok:
            self.transcript.append(EventType.DISCOVERY_FORMAT_REPAIR_REJECTED, {
                "reason_code": "FORMALIZER_INVOCATION_FAILED",
                "error_type": response["error_type"],
            })
            return None
        try:
            payload = MechanismDiscoveryPayload.model_validate(response)
            validate_discovery_format_repair(payload, context)
        except (TypeError, ValueError) as exc:
            code = str(exc) if str(exc).startswith("DISCOVERY_REPAIR_") else (
                "INVALID_REPAIRED_PAYLOAD"
            )
            self.transcript.append(EventType.DISCOVERY_FORMAT_REPAIR_REJECTED, {
                "reason_code": code,
            })
            return None
        self.transcript.append(EventType.DISCOVERY_FORMAT_REPAIR_COMPLETED, {
            "payload": payload.model_dump(mode="json"),
            "payload_digest": sha256_digest(payload.model_dump(mode="json")),
        })
        return payload

    def _invoke_formalizer(
        self,
        mechanism_context: dict[str, Any],
        proposals: dict[str, HypothesisProposal],
    ) -> tuple[bool, Any]:
        candidates = tuple(
            FormalizationCandidate(
                candidate_track_id=item.candidate_track_id,
                semantic_digest=item.semantic_digest,
                mechanism_summary=item.mechanism_summary,
                mechanism_class=item.mechanism_class,
                causal_entity_ref=item.causal_entity_ref,
                causal_explanation=item.causal_explanation,
                alternatives=item.alternatives,
                source_hypothesis_ids=item.source_hypothesis_ids,
            )
            for item in self.__investigation_workspace.candidates
            if item.stage in {CandidateStage.TENTATIVE, CandidateStage.ACTIVE}
            and item.disposition == CandidateDisposition.OPEN
            and item.alternatives
        )
        if not candidates:
            return False, {"error_type": "NoFormalizableCandidate"}
        formalization_context = MechanismFormalizationContext(
            alert_summary=mechanism_context["alert_summary"],
            candidates=candidates,
            entity_graph=mechanism_context["entity_graph"],
            observations=tuple(mechanism_context["observations"]),
            proposal_ids=tuple(sorted(proposals)),
        )
        self.transcript.append(EventType.MECHANISM_FORMALIZATION_STARTED, {
            "contract_version": 2 if getattr(
                self.__mechanism_formalizer, "supports_candidate_accounting", False
            ) else 1,
            "context_digest": sha256_digest(formalization_context.model_dump(mode="json")),
            "candidate_tracks": [item.candidate_track_id for item in candidates],
            "semantic_digests": [item.semantic_digest for item in candidates],
        })
        ok, raw = self._invoke_component_method(
            self.__mechanism_formalizer, "formalize", formalization_context
        )
        if not ok:
            self.transcript.append(EventType.MECHANISM_FORMALIZATION_REJECTED, raw)
            return False, raw
        try:
            supplied = MechanismFormalizationPayload.model_validate(raw)
            formalized = validate_formalization_semantics(
                supplied,
                formalization_context,
            )
            accounting_v2 = bool(getattr(
                self.__mechanism_formalizer, "supports_candidate_accounting", False
            ))
            if accounting_v2:
                try:
                    validate_formalization_accounting(formalized, formalization_context)
                    incomplete = False
                except FormalizationSemanticError:
                    incomplete = True
                if incomplete or not formalized.mechanisms:
                    initial = formalized
                    self.transcript.append(EventType.MECHANISM_FORMALIZATION_RETRIED, {
                        "attempt": 2,
                        "reason_code": (
                            "INCOMPLETE_CANDIDATE_ACCOUNTING" if incomplete
                            else "EMPTY_MECHANISM_SET"
                        ),
                        "initial_payload": initial.model_dump(mode="json"),
                    })
                    retry = getattr(self.__mechanism_formalizer, "retry_formalize", None)
                    if not callable(retry):
                        raise FormalizationSemanticError("RETRY_NOT_AVAILABLE")
                    if getattr(self.__mechanism_formalizer, "worker_backed", False):
                        retry_ok, retry_raw = self._invoke_component_method(
                            self.__mechanism_formalizer,
                            "retry_formalize",
                            formalization_context,
                            initial,
                        )
                    else:
                        retry_ok, retry_raw = self._invoke_discovery_component(
                            lambda: retry(formalization_context, initial)
                        )
                    if not retry_ok:
                        self.transcript.append(
                            EventType.MECHANISM_FORMALIZATION_REJECTED, retry_raw
                        )
                        return False, retry_raw
                    supplied = MechanismFormalizationPayload.model_validate(retry_raw)
                    formalized = validate_formalization_semantics(
                        supplied, formalization_context
                    )
                    first_mechanisms = {
                        item.candidate_track_id: item.mechanism.model_dump(mode="json")
                        for item in initial.mechanisms
                    }
                    final_mechanisms = {
                        item.candidate_track_id: item.mechanism.model_dump(mode="json")
                        for item in formalized.mechanisms
                    }
                    if any(final_mechanisms.get(track) != mechanism
                           for track, mechanism in first_mechanisms.items()):
                        raise FormalizationSemanticError("RETRY_LOST_FORMALIZED_MECHANISM")
                validate_formalization_accounting(formalized, formalization_context)
        except (ValueError, TypeError) as exc:
            failure = {
                "error_type": type(exc).__name__,
                "failure_stage": (
                    "semantic_validation"
                    if isinstance(exc, FormalizationSemanticError) else "payload_validation"
                ),
                "reason_code": (
                    exc.reason_code
                    if isinstance(exc, FormalizationSemanticError)
                    else "INVALID_FORMALIZATION_PAYLOAD"
                ),
            }
            self.transcript.append(EventType.MECHANISM_FORMALIZATION_REJECTED, failure)
            return False, failure
        payload = MechanismDiscoveryPayload(
            mechanisms=tuple(item.mechanism for item in formalized.mechanisms),
        )
        self.transcript.append(EventType.MECHANISM_FORMALIZATION_COMPLETED, {
            "payload": formalized.model_dump(mode="json"),
            "mechanism_payload_digest": sha256_digest(payload.model_dump(mode="json")),
            "source_id_rebindings": sum(
                before.mechanism.source_hypothesis_id
                != after.mechanism.source_hypothesis_id
                for before, after in zip(supplied.mechanisms, formalized.mechanisms, strict=True)
            ),
        })
        return True, payload.model_dump(mode="json")

    def _invoke_discovery_component(self, invoke) -> tuple[bool, Any]:
        from queue import Empty, Queue
        from threading import Thread

        mailbox: Queue = Queue(maxsize=1)

        def target():
            try:
                mailbox.put((True, invoke()))
            except Exception as exc:  # noqa: BLE001 - bounded structured output only
                failure = {"error_type": type(exc).__name__}
                from rca_agent.langchain_formalizer import FormalizerBoundaryError

                if isinstance(exc, FormalizerBoundaryError):
                    failure.update({
                        "failure_stage": "structured_output",
                        "reason_code": exc.reason_code,
                        "field_paths": list(exc.field_paths),
                    })
                raw_payload = getattr(exc, "raw_payload", None)
                if isinstance(raw_payload, (dict, list)):
                    failure["raw_payload"] = raw_payload
                mailbox.put((False, failure))

        Thread(target=target, daemon=True).start()
        try:
            return mailbox.get(timeout=self.__discovery_timeout)
        except Empty:
            return False, {"error_type": "TimeoutError"}

    def _invoke_component_method(
        self, component: Any, method: str, *args: Any
    ) -> tuple[bool, Any]:
        """Use direct worker lifecycle for worker-backed adapters.

        Legacy live adapters retain the existing bounded thread wrapper until
        their own factory is migrated. A worker-backed adapter already owns a
        killable child and must not be wrapped in another daemon thread.
        """
        if not getattr(component, "worker_backed", False):
            return self._invoke_discovery_component(
                lambda: getattr(component, method)(*args)
            )
        try:
            return True, getattr(component, method)(*args)
        except Exception as exc:  # noqa: BLE001 - sanitized at the Runner boundary
            failure = {"error_type": getattr(exc, "error_type", type(exc).__name__)}
            from rca_agent.langchain_formalizer import FormalizerBoundaryError

            if isinstance(exc, FormalizerBoundaryError):
                failure.update({
                    "failure_stage": "structured_output",
                    "reason_code": exc.reason_code,
                    "field_paths": list(exc.field_paths),
                })
            raw_payload = getattr(exc, "raw_payload", None)
            if isinstance(raw_payload, (dict, list)):
                failure["raw_payload"] = raw_payload
            return False, failure

    def _invoke_planner(
        self, context: dict[str, Any]
    ) -> tuple[VerificationPlanPayload | None, str | None]:
        ok, raw = self._invoke_planner_operation("plan", context)
        if not ok:
            self.transcript.append(EventType.DISCOVERY_PAYLOAD_REJECTED, {
                "stage": "verification_planner", **raw,
            })
            return None, "PLANNER_" + raw["error_type"].upper()
        try:
            return VerificationPlanPayload.model_validate(raw), None
        except (ValueError, TypeError) as exc:
            self.transcript.append(EventType.DISCOVERY_PAYLOAD_REJECTED, {
                "stage": "verification_planner", "error_type": type(exc).__name__,
            })
            return None, "PLANNER_INVALID_PAYLOAD"

    def _invoke_planner_operation(
        self, operation: str, context: dict[str, Any]
    ) -> tuple[bool, Any]:
        planner = self.__verification_planner
        assert planner is not None
        if getattr(planner, "worker_backed", False):
            try:
                return True, getattr(planner, operation)(deepcopy(context))
            except Exception as exc:  # noqa: BLE001 - sanitized at the Runner boundary
                return False, {"error_type": getattr(exc, "error_type", type(exc).__name__)}
        return self._invoke_discovery_component(
            lambda: getattr(planner, operation)(deepcopy(context))
        )

    def _compile_plans(
        self,
        payload: VerificationPlanPayload,
        mechanisms: dict[str, FrozenMechanism],
        registry: EvidenceRegistry,
        proposals: dict[str, HypothesisProposal],
        compile_kwargs: dict[str, Any],
        accepted: set[str],
        *,
        attempt: int,
    ) -> tuple[dict[str, dict[str, Any]], tuple[int, int]]:
        failed: dict[str, dict[str, Any]] = {}
        seen: set[str] = set()
        registered = 0
        aligned = 0
        for plan in payload.plans:
            self.transcript.append(EventType.VERIFICATION_PLAN_GENERATED, {
                "attempt": attempt, "plan": plan.model_dump(mode="json"),
            })
            mechanism = mechanisms.get(plan.mechanism_id)
            if mechanism is None or plan.mechanism_id in seen or plan.mechanism_id in accepted:
                code = "UNKNOWN_OR_DUPLICATE_MECHANISM"
                self._reject_plan(plan.mechanism_id, attempt, code)
                continue
            seen.add(plan.mechanism_id)
            try:
                normalized_plan, plan_repairs = registry.normalize_plan(mechanism, plan)
                compiled = registry.compile_plan(mechanism, normalized_plan, **compile_kwargs)
                if not all(self._signal_executable(signal) for signal in compiled.signals):
                    raise ValueError("query recipe is not executable in this episode")
                registration = self._register_two_stage_candidate(
                    compiled, mechanism, proposals[mechanism.source_hypothesis_id]
                )
                self.__compiled_plans[plan.mechanism_id] = normalized_plan
                self.__candidate_for_mechanism[plan.mechanism_id] = compiled.candidate.candidate_id
                accepted.add(plan.mechanism_id)
                registered += registration == "dynamic_registered"
                aligned += registration == "catalog_aligned"
                self.transcript.append(EventType.VERIFICATION_PLAN_COMPILED, {
                    "attempt": attempt,
                    "mechanism_id": plan.mechanism_id,
                    "mechanism_digest": mechanism.digest,
                    "plan": normalized_plan.model_dump(mode="json"),
                    "plan_repairs": plan_repairs,
                    "registration": registration,
                })
            except ValueError as exc:
                code = _contract_error_code(str(exc))
                from rca_agent.runner.planner_feedback import plan_feedback

                handles = registry.capability_handles()
                diagnostics = plan_feedback(mechanism, plan, handles, code)
                code = diagnostics["error_code"]
                failed[plan.mechanism_id] = {
                    "mechanism_id": plan.mechanism_id, "error_code": code,
                    "plan": plan.model_dump(mode="json"),
                    "diagnostics": diagnostics,
                }
                self._reject_plan(plan.mechanism_id, attempt, code, diagnostics=diagnostics,
                                  plan=plan, handles=handles)
        for mechanism_id in mechanisms.keys() - seen - accepted:
            code = "MISSING_VERIFICATION_PLAN"
            failed[mechanism_id] = {
                "mechanism_id": mechanism_id, "error_code": code,
            }
            self._reject_plan(mechanism_id, attempt, code)
        return failed, (registered, aligned)

    def _reject_plan(
        self, mechanism_id: str, attempt: int, error_code: str,
        *, diagnostics: dict | None = None, plan=None, handles=(),
    ) -> None:
        payload = {
            "attempt": attempt, "mechanism_id": mechanism_id, "error_code": error_code,
        }
        if diagnostics is not None:
            payload.update({"diagnostics": diagnostics, "plan": plan.model_dump(mode="json"),
                            "capabilities": [h.model_dump(mode="json") for h in handles]})
        self.transcript.append(EventType.VERIFICATION_PLAN_REJECTED, payload)

    def _register_two_stage_candidate(
        self,
        compiled,
        mechanism: FrozenMechanism,
        original: HypothesisProposal,
    ) -> str:
        existing = next((candidate for candidate in self.__decision_provider.catalog(
            compiled.candidate.causal_entity
        ) if mechanism.mechanism.casefold() in {
            candidate.candidate_id.casefold(), candidate.fault_type.casefold()
        } and candidate.family == mechanism.family), None)
        if existing is not None:
            required = {condition.signal for condition in existing.conditions}
            if not required <= {signal.signal for signal in compiled.signals}:
                raise ValueError("catalog alignment lacks executable signal recipes")
            hypothesis_id = "h-aligned-" + compiled.version[:20]
            self.transcript.append(EventType.CATALOG_CANDIDATE_ALIGNED, {
                "candidate_id": existing.candidate_id,
                "hypothesis_id": hypothesis_id,
                "source_hypothesis_id": original.hypothesis_id,
                "mechanism_id": mechanism.mechanism_id,
            })
            candidate_id = existing.candidate_id
            registration = "catalog_aligned"
        else:
            hypothesis_id = "h-discovered-" + compiled.version[:20]
            rebound = compiled.model_copy(update={"source_hypothesis_id": hypothesis_id})
            self.__decision_provider.register_dynamic(rebound)
            self.transcript.append(EventType.DYNAMIC_CANDIDATE_REGISTERED, {
                "compiled": rebound.model_dump(mode="json"),
                "source_hypothesis_id": original.hypothesis_id,
                "mechanism_id": mechanism.mechanism_id,
                "mechanism_digest": mechanism.digest,
            })
            candidate_id = compiled.candidate.candidate_id
            candidate = compiled.candidate
            self.__dynamic_options.append({
                "candidate_id": candidate.candidate_id,
                "fault_type": candidate.fault_type,
                "family": candidate.family.value,
                "required_signals": tuple(signal.signal for signal in compiled.signals),
                "required_capabilities": tuple(candidate.required_capabilities),
            })
            registration = "dynamic_registered"
        self._submit_hypothesis(HypothesisProposal(
            hypothesis_id=hypothesis_id,
            candidate_id=candidate_id,
            mechanism=mechanism.mechanism if existing is None else existing.fault_type,
            causal_entity_ref=compiled.candidate.causal_entity,
            rationale=mechanism.causal_explanation,
        ))
        queries = {
            query_fingerprint(signal.query.tool_name, signal.query.request): signal.query
            for signal in compiled.signals
        }
        self.__verification_queries.extend((hypothesis_id, query) for query in queries.values())
        return registration

    def _record_discovery_draft(
        self,
        draft_index: int,
        raw_draft,
        final_draft,
        initial_error: str | None,
        repair_actions: tuple[str, ...],
        final_status: str,
        final_error: str | None,
        registration: str,
    ) -> None:
        self.transcript.append(EventType.DISCOVERY_DRAFT_EVALUATED, {
            "draft_index": draft_index,
            "raw_draft": raw_draft.model_dump(mode="json"),
            "final_draft": final_draft.model_dump(mode="json"),
            "initial_status": "rejected" if initial_error is not None else final_status,
            "initial_error": initial_error,
            "repair_actions": repair_actions,
            "final_status": final_status,
            "final_error": final_error,
            "registration": registration,
        })

    def _discovery_entity_graph(self, registry: EvidenceRegistry) -> dict[str, Any]:
        assert self.__bootstrap is not None
        alert = self.__bootstrap.alert_entity
        nodes: dict[str, dict[str, Any]] = {
            alert.entity_ref: alert.model_dump(mode="json")
        }
        edges: set[tuple[str, str, str, str]] = set()
        for record in self.__observation_ledger.values():
            observation = record.observation
            target = None
            relation = observation.details.get("relation")
            direction = observation.details.get("direction", "outgoing")
            if observation.signal in {"related_entity", "trace_participant"} and isinstance(
                observation.value, dict
            ):
                target = observation.value
            elif observation.signal == "topology_edge":
                target = observation.details.get("neighbor")
            resolved = observation.details.get("resolved_entity")
            if isinstance(resolved, dict):
                nodes[resolved["entity_ref"]] = resolved
                edges.add((observation.entity_ref, resolved["entity_ref"],
                           "trace_resolves_to", "outgoing"))
            resolved_service = observation.details.get("resolved_service")
            if isinstance(resolved_service, dict):
                nodes[resolved_service["entity_ref"]] = resolved_service
                source = resolved["entity_ref"] if isinstance(resolved, dict) else observation.entity_ref
                edges.add((source, resolved_service["entity_ref"], "operation_of", "parent"))
            if isinstance(target, dict) and "entity_ref" in target:
                nodes[target["entity_ref"]] = target
                edges.add((observation.entity_ref, target["entity_ref"],
                           str(relation or observation.value), str(direction)))
        rules_by_signal: dict[str, list[str]] = {}
        for rule in registry.rules:
            rules_by_signal.setdefault(rule.signal, []).append(rule.rule_id)
        available_by_entity: dict[str, set[str]] = {}
        for signal in registry.signals:
            available_by_entity.setdefault(signal.entity_ref, set()).update(
                rules_by_signal.get(signal.signal, ())
            )
        return {
            "nodes": [
                {**node, "available_rule_ids": sorted(available_by_entity.get(ref, set()))}
                for ref, node in sorted(nodes.items())
            ],
            "edges": [
                {"source_entity_ref": source, "target_entity_ref": target,
                 "relation": relation, "direction": direction}
                for source, target, relation, direction in sorted(edges)
            ],
        }

    def _signal_executable(self, spec) -> bool:
        assert self.__bootstrap is not None
        if not spec.available or spec.entity_ref not in self.__issued_entities:
            return False
        capability = next((c for c in self.__bootstrap.capabilities
                           if c.tool_name == spec.query.tool_name and c.available), None)
        if capability is None:
            return False
        request = spec.query.request
        entity = getattr(request, "entity_ref", None)
        if entity is not None and entity != spec.entity_ref:
            return False
        trace = getattr(request, "trace_ref", None)
        if trace is not None and not any(
            r.observation.details.get("trace_ref") == trace
            for r in self.__observation_ledger.values()
        ):
            return False
        window = getattr(request, "window", getattr(request, "incident_window", None))
        if window is not None:
            horizon = self.__bootstrap.observation_horizon
            if (window.start < horizon.start or window.end > horizon.end
                    or window.seconds > capability.max_window_seconds):
                return False
        limit = getattr(request, "max_points", getattr(request, "max_stages", 1))
        return limit <= capability.max_results

    @property
    def has_verification_queries(self) -> bool:
        return bool(self.__verification_queries)

    def feedback_plan_available(self) -> bool:
        if (self.status != EpisodeStatus.ACTIVE or self.__feedback_planned
                or self.__verification_planner is None or self.__resolved_registry is None
                or self.__verification_queries or self.__budget_remaining <= 0
                or self.__rounds + 2 >= self.__max_rounds):
            return False
        return self._feedback_plan_target() is not None

    def _feedback_plan_target(self):
        assert self.__resolved_registry is not None
        attempted = {event.payload["query_ref"] for event in self.transcript.events
                     if event.event_type == EventType.TOOL_REQUEST}
        registry = self.__resolved_registry
        reports = self.__decision_provider.verification_reports(
            tuple(self.__proposals), tuple(self.__observation_ledger.values())
        )
        for assessment in self.preview_core_assessment().candidate_assessments:
            report = reports.get(assessment.candidate_id)
            if (not assessment.verification_pending or report is None
                    or report.state.value == "failed"):
                continue
            for mechanism_id, candidate_id in self.__candidate_for_mechanism.items():
                if candidate_id != assessment.candidate_id:
                    continue
                used = {item.capability_ref for item in
                        self.__compiled_plans[mechanism_id].predictions}
                for handle in registry.capability_handles():
                    if handle.capability_ref in used:
                        continue
                    signal = next((item for item in registry.signals
                                   if item.entity_ref == handle.entity_ref
                                   and item.signal == handle.signal), None)
                    if signal is not None and self._signal_executable(signal) and (
                        query_fingerprint(signal.query.tool_name, signal.query.request)
                        not in attempted
                    ):
                        return mechanism_id, assessment
        return None

    def extend_verification_plan(self) -> dict[str, Any]:
        """Request one additive plan revision from the existing Planner."""
        if not self.feedback_plan_available():
            return {"status": "unavailable"}
        assert self.__resolved_registry is not None and self.__bootstrap is not None
        self.__feedback_planned = True
        self.__rounds += 1
        registry = self.__resolved_registry
        target = self._feedback_plan_target()
        if target is None:
            return {"status": "unavailable"}
        mechanism_id, assessment = target
        frozen = self.__frozen_mechanisms[mechanism_id]
        previous = self.__compiled_plans[mechanism_id]
        old_candidate_id = assessment.candidate_id
        old_compiled = self.__decision_provider.dynamic_compiled(old_candidate_id)
        assert old_compiled is not None
        reports = self.__decision_provider.verification_reports(
            tuple(self.__proposals), tuple(self.__observation_ledger.values())
        )
        blockers = reports[old_candidate_id].blockers if old_candidate_id in reports else ()
        feedback = {
            "candidate_id": old_candidate_id,
            "missing_signals": assessment.missing_signals,
            "unavailable_signals": assessment.unavailable_signals,
            "blockers": blockers,
        }
        self.transcript.append(EventType.VERIFICATION_PLAN_EXTENSION_STARTED, {
            "mechanism_id": mechanism_id,
            "old_candidate_id": old_candidate_id,
            "previous_plan": previous.model_dump(mode="json"),
            "core_feedback": feedback,
        })
        context = {
            "frozen_mechanisms": [frozen.model_dump(mode="json")],
            "capabilities": [item.model_dump(mode="json")
                             for item in registry.capability_handles()],
            "observations": [record.observation.model_dump(mode="json")
                             for record in self.__observation_ledger.values()],
            "existing_plan": previous.model_dump(mode="json"),
            "core_feedback": feedback,
            "unavailable_signals": {
                signal: reason for capability in self.__bootstrap.capabilities
                for signal, reason in capability.unavailable_signals.items()
            },
        }
        structured_extension = getattr(
            self.__verification_planner, "supports_plan_extension", False
        )
        ok, raw = self._invoke_planner_operation(
            "plan_extension" if structured_extension else "plan", context
        )
        try:
            if not ok:
                raise ValueError("PLANNER_" + raw["error_type"].upper())
            if structured_extension:
                addition = VerificationPlanExtension.model_validate(raw)
                if addition.mechanism_id != mechanism_id:
                    raise ValueError("UNKNOWN_FROZEN_MECHANISM")
                proposed = VerificationPlanDraft(
                    mechanism_id=mechanism_id,
                    predictions=(*previous.predictions, *addition.added_predictions),
                )
            else:
                payload = VerificationPlanPayload.model_validate(raw)
                if len(payload.plans) != 1:
                    raise ValueError("MISSING_ADDITIVE_PLAN")
                proposed = payload.plans[0]
            plan, repairs = registry.normalize_plan(frozen, proposed)
            new_refs = added_predictions(previous, plan)
            compiled = registry.compile_plan(
                frozen, plan,
                discovery_query_refs=old_compiled.discovery_query_refs,
                discovery_observation_refs=old_compiled.discovery_observation_refs,
                discovery_fact_digests=old_compiled.discovery_fact_digests,
            )
            if not all(self._signal_executable(item) for item in compiled.signals):
                raise ValueError("CAPABILITY_NOT_EXECUTABLE")
            attempted = {event.payload["query_ref"] for event in self.transcript.events
                         if event.event_type == EventType.TOOL_REQUEST}
            old_queries = {query_fingerprint(item.query.tool_name, item.query.request)
                           for item in old_compiled.signals}
            new_queries = {
                query_fingerprint(item.query.tool_name, item.query.request): item.query
                for item in compiled.signals
                if query_fingerprint(item.query.tool_name, item.query.request)
                not in old_queries | attempted
            }
            query_cost = sum(
                next(capability.query_cost for capability in self.__bootstrap.capabilities
                     if capability.tool_name == action.tool_name)
                for action in new_queries.values()
            )
            if (not new_queries or self.__budget_remaining < query_cost
                    or self.__rounds + 2 * len(new_queries) > self.__max_rounds):
                raise ValueError("NO_NEW_EXECUTABLE_QUERY")
            if not set(new_refs) <= {item.capability_ref
                                    for item in registry.capability_handles()}:
                raise ValueError("UNKNOWN_CAPABILITY_REF")
        except (ValueError, TypeError) as exc:
            code = str(exc) if type(exc) is ValueError and str(exc).isupper() else (
                _contract_error_code(str(exc))
            )
            self.transcript.append(EventType.VERIFICATION_PLAN_EXTENSION_REJECTED, {
                "mechanism_id": mechanism_id, "error_code": code,
            })
            return {"status": "rejected", "reason": code}

        new_hypothesis_id = "h-discovered-" + compiled.version[:20]
        rebound = compiled.model_copy(update={"source_hypothesis_id": new_hypothesis_id})
        self.__decision_provider.supersede_dynamic(old_candidate_id, rebound)
        previous_proposal = next(item for item in self.__proposals
                                 if item.candidate_id == old_candidate_id)
        self.transcript.append(EventType.VERIFICATION_PLAN_EXTENDED, {
            "mechanism_id": mechanism_id,
            "mechanism_digest": frozen.digest,
            "old_candidate_id": old_candidate_id,
            "old_hypothesis_id": previous_proposal.hypothesis_id,
            "new_hypothesis_id": new_hypothesis_id,
            "previous_plan": previous.model_dump(mode="json"),
            "plan": plan.model_dump(mode="json"),
            "plan_repairs": repairs,
            "compiled": rebound.model_dump(mode="json"),
            "new_query_refs": tuple(new_queries),
        })
        self._submit_hypothesis(HypothesisProposal(
            hypothesis_id=new_hypothesis_id,
            candidate_id=rebound.candidate.candidate_id,
            mechanism=frozen.mechanism,
            causal_entity_ref=frozen.causal_entity_ref,
            rationale=frozen.causal_explanation,
            observation_refs=previous_proposal.observation_refs,
            missing_evidence=previous_proposal.missing_evidence,
        ))
        self.__verification_queries.extend((new_hypothesis_id, action)
                                           for action in new_queries.values())
        self.__compiled_plans[mechanism_id] = plan
        self.__candidate_for_mechanism[mechanism_id] = rebound.candidate.candidate_id
        self.__dynamic_options = [item for item in self.__dynamic_options
                                  if item["candidate_id"] != old_candidate_id]
        self.__dynamic_options.append({
            "candidate_id": rebound.candidate.candidate_id,
            "fault_type": rebound.candidate.fault_type,
            "family": rebound.candidate.family.value,
            "required_signals": tuple(item.signal for item in rebound.signals),
            "required_capabilities": tuple(rebound.candidate.required_capabilities),
        })
        return {"status": "extended", "mechanism_id": mechanism_id,
                "new_queries": len(new_queries)}

    def feedback_query_options(self) -> tuple[tuple[str, str, CallToolAction], ...]:
        """Unattempted frozen-plan queries for Core-reported evidence gaps."""
        available_slots = min(self.__budget_remaining,
                              (self.__max_rounds - self.__rounds) // 2)
        if (self.status != EpisodeStatus.ACTIVE or self.__verification_queries
                or available_slots <= 0
                or not isinstance(self.__decision_provider, CoreDecisionProvider)):
            return ()
        attempted = {
            event.payload["query_ref"] for event in self.transcript.events
            if event.event_type == EventType.TOOL_REQUEST
        }
        result = []
        for assessment in self.preview_core_assessment().candidate_assessments:
            compiled = self.__decision_provider.dynamic_compiled(assessment.candidate_id)
            if compiled is None or not assessment.verification_pending:
                continue
            gaps = {*assessment.missing_signals, *assessment.unavailable_signals}
            for signal in compiled.signals:
                if signal.signal not in gaps:
                    continue
                fingerprint = query_fingerprint(signal.query.tool_name, signal.query.request)
                if fingerprint in attempted:
                    continue
                for hypothesis_id in assessment.hypothesis_ids:
                    result.append((hypothesis_id, signal.signal, signal.query))
        return tuple(result[:available_slots])

    def verify_next_discovered_query(self) -> ToolResponse | AnswerArtifact | None:
        self._require_status(EpisodeStatus.ACTIVE)
        if not self.__verification_queries:
            return None
        hypothesis_id, action = self.__verification_queries[0]
        result = self._handle_granted_query(action, "verification")
        self.__verification_queries.pop(0)
        if isinstance(result, AnswerArtifact):
            return result
        if isinstance(result, ToolResponse) and result.observations:
            update = self.handle(UpdateCandidateEvidenceAction(
                hypothesis_id=hypothesis_id,
                observation_refs=tuple(o.observation_id for o in result.observations),
            ))
            if isinstance(update, AnswerArtifact):
                return update
        return result

    def verify_feedback_query(self, hypothesis_id: str, signal: str,
                              action: CallToolAction) -> ToolResponse | AnswerArtifact | None:
        if (hypothesis_id, signal, action) not in self.feedback_query_options():
            raise ActionRejectedError("query is outside the host verification plan")
        return self._handle_granted_query(action, "verification")

    def handle_startup_query(self, action: CallToolAction) -> ToolResponse | AnswerArtifact | None:
        return self._handle_granted_query(action, "startup")

    def select_startup_query(self) -> CallToolAction | None:
        """Host decision with no caller-supplied Action; record before policy execution."""
        self._require_status(EpisodeStatus.ACTIVE)
        if self.__inquiry_loop_mode != "batched":
            return None
        from rca_agent.investigation.inquiry_loop import LoopState
        from rca_agent.investigation.startup_authorization import (
            selected_startup_action,
            startup_grant,
        )
        existing = startup_grant(self.transcript.events)
        if existing is not None:
            if (self.__startup_selection is None
                    or existing["action"] != self.__startup_selection[0].model_dump(mode="json")
                    or existing["selection_event_digest"] != self.__startup_selection[1]):
                raise ActionRejectedError("startup selection was not issued by this host")
            return self.__startup_selection[0]
        self.__startup_selection = None
        state = LoopState.model_validate(self.investigation_workspace_snapshot.inquiry_loop)
        action = selected_startup_action(state, self.__bootstrap, self.transcript.events)
        if action is not None:
            self._append_workspace_event(EventType.STARTUP_ACTION_SELECTED, {
                "based_on_digest": self.transcript.events[-1].digest,
                "action": action.model_dump(mode="json"),
            })
            self.__startup_selection = (action, self.transcript.events[-1].digest)
        return action

    def _handle_granted_query(self, action: CallToolAction, source: str) -> ToolResponse | AnswerArtifact | None:
        if self.__tool_grant is not None:
            raise ActionRejectedError("nested tool execution grant")
        self.__tool_grant = (source, action)
        self.__tool_grant_validated = False
        try:
            return self.handle(action)
        finally:
            self.__tool_grant = None
            self.__tool_grant_validated = False

    def preview_core_assessment(self) -> CoreAssessmentPreview:
        """Return a read-only Core assessment of the current submitted evidence."""

        self._require_status(EpisodeStatus.ACTIVE)
        proposals = tuple(self.__proposals)
        issued_observations = tuple(
            self.__observation_ledger[item]
            for item in sorted(self.__observation_ledger)
        )
        provider = self.__decision_provider
        if isinstance(provider, CoreDecisionProvider):
            assert self.__bootstrap is not None
            return provider.preview(
                proposals,
                issued_observations,
                self.__bootstrap.capabilities,
            )
        return CoreAssessmentPreview(
            overall_decision=provider.decide(
                proposals,
                issued_observations,
                StopReason.POLICY_REQUEST,
            )
        )

    def start(self) -> ResetResponse:
        self._require_status(EpisodeStatus.CREATED)
        bootstrap = self.__environment.reset()
        self.__bootstrap = bootstrap
        self.__budget_remaining = bootstrap.query_budget.remaining
        self.__issued_entities = {bootstrap.alert_entity.entity_ref}
        self.status = EpisodeStatus.ACTIVE
        self.transcript.append(
            EventType.ENVIRONMENT_RESET,
            {"status": self.status, "bootstrap": bootstrap.model_dump(mode="json")},
        )
        return bootstrap

    def handle(self, action: PolicyAction) -> ToolResponse | AnswerArtifact | None:
        self._require_status(EpisodeStatus.ACTIVE)
        if isinstance(action, CallToolAction):
            self._admit_tool_action(action)
        self.transcript.append(EventType.POLICY_ACTION, action)
        if isinstance(action, RequestStopAction):
            self.transcript.append(EventType.STOP_REQUESTED, action)
            return self.freeze_answer(StopReason.POLICY_REQUEST)
        if self.__rounds >= self.__max_rounds:
            self.transcript.append(
                EventType.STOP_REQUESTED,
                {"reason": StopReason.MAX_ROUNDS, "rounds": self.__rounds},
            )
            return self.freeze_answer(StopReason.MAX_ROUNDS)
        self.__rounds += 1
        if isinstance(action, CallToolAction):
            return self._call_tool(action)
        if isinstance(action, SubmitHypothesisAction):
            self._submit_hypothesis(action.proposal)
            return None
        if isinstance(action, UpdateCandidateEvidenceAction):
            self._update_candidate_evidence(action)
            return None
        raise TypeError(f"unsupported typed action: {type(action).__name__}")

    def reject_raw_action(self, payload: Any, error_type: str = "invalid_policy_action") -> None:
        self._require_status(EpisodeStatus.ACTIVE)
        self.transcript.append(
            EventType.ACTION_REJECTED,
            {"error_type": error_type, "payload_digest": sha256_digest(payload)},
        )

    def record_policy_process_started(self, visible_context: dict[str, Any]) -> None:
        self._require_status(EpisodeStatus.ACTIVE)
        self.transcript.append(
            EventType.POLICY_PROCESS_STARTED,
            {
                "context_fields": sorted(visible_context),
                "context_digest": sha256_digest(visible_context),
            },
        )

    def record_policy_process_exited(self, exit_kind: str) -> None:
        self.transcript.append(EventType.POLICY_PROCESS_EXITED, {"exit_kind": exit_kind})

    def terminate_policy_failure(self, reason: StopReason, error_type: str) -> AnswerArtifact:
        self._require_status(EpisodeStatus.ACTIVE)
        event_type = (
            EventType.POLICY_TIMEOUT
            if reason == StopReason.POLICY_TIMEOUT
            else EventType.POLICY_EXCEPTION
        )
        self.transcript.append(event_type, {"error_type": error_type})
        self.transcript.append(EventType.STOP_REQUESTED, {"reason": reason})
        return self.freeze_answer(reason)

    def finish_script(self) -> AnswerArtifact:
        self._require_status(EpisodeStatus.ACTIVE)
        self.transcript.append(
            EventType.STOP_REQUESTED,
            {"reason": StopReason.SCRIPT_COMPLETED},
        )
        return self.freeze_answer(StopReason.SCRIPT_COMPLETED)

    def _finalize_investigation_workspace(self) -> None:
        """Project existing Inquiry/Discovery facts without changing investigation behavior."""

        if self.__workspace_finalized:
            return
        if self.__workspace_mode == "off":
            self.__workspace_finalized = True
            return
        discovery_started = next((
            event for event in reversed(self.transcript.events)
            if event.event_type == EventType.DISCOVERY_STARTED
            and event.payload.get("inquiry_handoff")
        ), None)
        discovery_completed = next((
            event for event in reversed(self.transcript.events)
            if event.event_type == EventType.DISCOVERY_COMPLETED
        ), None)
        handoff = discovery_started.payload["inquiry_handoff"] if discovery_started else {}
        query_links = handoff.get("query_links", ())
        links_by_hypothesis: dict[str, list[dict[str, Any]]] = {}
        for link in query_links:
            if link.get("hypothesis_id"):
                links_by_hypothesis.setdefault(link["hypothesis_id"], []).append(link)

        if discovery_completed is not None:
            processed_tracks: set[str] = set()
            for disposition in discovery_completed.payload.get("inquiry_dispositions", ()):
                candidate = next((
                    item for item in self.__investigation_workspace.candidates
                    if disposition["hypothesis_id"] in item.source_hypothesis_ids
                ), None)
                if candidate is None or candidate.candidate_track_id in processed_tracks:
                    continue
                processed_tracks.add(candidate.candidate_track_id)
                links = links_by_hypothesis.get(disposition["hypothesis_id"], ())
                observation_refs = tuple(sorted({
                    ref for link in links for ref in link.get("observation_refs", ())
                }))
                query_refs = tuple(dict.fromkeys(
                    link["query_ref"] for link in links if link.get("query_ref")
                ))
                if (discovery_started.payload.get("handoff_contract_version") in {2, 3, 4, 5, 6, 7}
                        and disposition.get("basis_observation_refs")):
                    observation_refs = tuple(sorted(disposition["basis_observation_refs"]))
                    query_refs = tuple(sorted({self.__observation_ledger[ref].query_ref
                                               for ref in observation_refs}))
                basis = TransitionBasis(
                    observation_refs=observation_refs,
                    query_refs=query_refs,
                    source_event_refs=(discovery_completed.digest,),
                    reason_code="DISCOVERY_DISPOSITION_" + disposition["disposition"].upper(),
                    rationale=disposition.get("reason") or (
                        "Discovery recorded a non-authoritative candidate disposition."
                    ),
                )
                kind = disposition["disposition"]
                if kind == "kept":
                    self._ensure_candidate_active(candidate, basis)
                elif kind == "revised":
                    formal = disposition.get("formal_mechanism")
                    if discovery_started.payload.get("handoff_contract_version") in {2, 3, 4, 5, 6, 7}:
                        if disposition.get("original_disposition") == "retained":
                            self._ensure_candidate_active(candidate, basis)
                        else:
                            self._defer_candidate(candidate, basis)
                    elif not formal:
                        self._defer_candidate(candidate, basis)
                    elif _normalized_semantic_text(formal) == _normalized_semantic_text(
                        candidate.mechanism_summary
                    ):
                        candidate = self._ensure_candidate_active(candidate, basis)
                        if formal != candidate.mechanism_summary:
                            self._append_candidate_transition(
                                candidate,
                                CandidateTransitionType.CLARIFY,
                                to_stage=candidate.stage,
                                to_disposition=candidate.disposition,
                                basis=basis,
                                mechanism_summary=formal,
                            )
                    else:
                        self._append_candidate_superseded(candidate, formal, basis)
                elif kind == "deferred":
                    self._defer_candidate(candidate, basis)
                elif kind == "discarded":
                    if observation_refs:
                        self._append_candidate_transition(
                            candidate,
                            CandidateTransitionType.DISFAVOR,
                            to_stage=candidate.stage,
                            to_disposition=CandidateDisposition.DISFAVORED,
                            basis=basis,
                        )
                    else:
                        self._defer_candidate(candidate, basis)

        compiled_by_mechanism: dict[str, list[str]] = {}
        for event in self.transcript.events:
            if (event.event_type == EventType.DYNAMIC_CANDIDATE_REGISTERED
                    and event.payload.get("mechanism_id")):
                compiled_by_mechanism.setdefault(event.payload["mechanism_id"], []).append(
                    event.payload["compiled"]["candidate"]["candidate_id"]
                )
            elif (event.event_type == EventType.CATALOG_CANDIDATE_ALIGNED
                  and event.payload.get("mechanism_id")):
                compiled_by_mechanism.setdefault(event.payload["mechanism_id"], []).append(
                    event.payload["candidate_id"]
                )
            elif event.event_type == EventType.VERIFICATION_PLAN_EXTENDED:
                compiled_by_mechanism.setdefault(event.payload["mechanism_id"], []).append(
                    event.payload["compiled"]["candidate"]["candidate_id"]
                )

        frozen_events = tuple(
            event for event in self.transcript.events
            if event.event_type == EventType.MECHANISM_DRAFT_FROZEN
        )
        for frozen_event in frozen_events:
            mechanism = frozen_event.payload["mechanism"]
            candidate = next((
                item for item in self.__investigation_workspace.candidates
                if item.stage != CandidateStage.CLOSED
                and _normalized_semantic_text(item.mechanism_summary)
                == _normalized_semantic_text(mechanism["mechanism"])
            ), None)
            if candidate is None or candidate.disposition == CandidateDisposition.DISFAVORED:
                continue
            basis = TransitionBasis(
                source_event_refs=(frozen_event.digest,),
                reason_code="FROZEN_MECHANISM_LINKED",
                rationale="The frozen mechanism preserves this investigation candidate's semantics.",
            )
            candidate = self._ensure_candidate_active(candidate, basis)
            if candidate.formal_mechanism_id is None:
                self._append_candidate_transition(
                    candidate,
                    CandidateTransitionType.FORMALIZE,
                    to_stage=CandidateStage.ACTIVE,
                    to_disposition=candidate.disposition,
                    basis=basis,
                    formal_mechanism_id=mechanism["mechanism_id"],
                    compiled_candidate_ids_added=tuple(dict.fromkeys(
                        compiled_by_mechanism.get(mechanism["mechanism_id"], ())
                    )),
                )
        self.__workspace_finalized = True

    def _ensure_candidate_active(
        self, candidate: InvestigationCandidate, basis: TransitionBasis,
    ) -> InvestigationCandidate:
        if candidate.stage == CandidateStage.TENTATIVE:
            self._append_candidate_transition(
                candidate,
                CandidateTransitionType.RETAIN,
                to_stage=CandidateStage.ACTIVE,
                to_disposition=candidate.disposition,
                basis=basis,
            )
        elif candidate.stage == CandidateStage.DEFERRED:
            self._append_candidate_transition(
                candidate,
                CandidateTransitionType.REACTIVATE,
                to_stage=CandidateStage.ACTIVE,
                to_disposition=candidate.disposition,
                basis=basis,
            )
        return self.__investigation_workspace.candidate(candidate.candidate_track_id) or candidate

    def _defer_candidate(
        self, candidate: InvestigationCandidate, basis: TransitionBasis,
    ) -> None:
        if candidate.stage in {CandidateStage.TENTATIVE, CandidateStage.ACTIVE}:
            self._append_candidate_transition(
                candidate,
                CandidateTransitionType.DEFER,
                to_stage=CandidateStage.DEFERRED,
                to_disposition=candidate.disposition,
                basis=basis,
            )

    def freeze_answer(self, termination_reason: StopReason) -> AnswerArtifact:
        self._require_status(EpisodeStatus.ACTIVE)
        for task in self.__investigation_workspace.tasks:
            if task.scheduling is not None and task.status == DiscriminationTaskStatus.OPEN:
                self._cancel_active_query(task, "EPISODE_ENDED")
        proposals = tuple(self.__proposals)
        used_observations = tuple(
            sorted({item for proposal in proposals for item in proposal.observation_refs})
        )
        missing = tuple(
            sorted({item for proposal in proposals for item in proposal.missing_evidence})
        )
        decision = self.__decision_provider.decide(
            proposals,
            tuple(self.__observation_ledger[item] for item in used_observations),
            termination_reason,
        )
        if isinstance(self.__decision_provider, CoreDecisionProvider):
            reports = self.__decision_provider.verification_reports(
                proposals, tuple(self.__observation_ledger[item] for item in used_observations),
            )
            if reports:
                self.transcript.append(EventType.DYNAMIC_VERIFICATION, {
                    "candidates": {key: report.model_dump(mode="json") for key, report in reports.items()},
                })
        self._finalize_investigation_workspace()
        self._record_mechanism_coverage_shadow()
        frozen_at = datetime.now(UTC)
        candidate_assessments = self.preview_core_assessment().candidate_assessments
        fields = {
            "episode_id": self.episode_id,
            "termination_reason": termination_reason,
            "policy_proposals": proposals,
            "authoritative_decision": decision,
            "observation_refs": used_observations,
            "missing_evidence": missing,
            "inquiry_candidates": inquiry_outcomes(self.transcript.events, candidate_assessments),
            "investigation_workspace": (
                self.investigation_workspace_snapshot
                if self.__workspace_mode in {"shadow", "active"} else None
            ),
            "transcript_digest": self.transcript.head_digest,
            "component_versions": {
                "rca-agent": _package_version("rca-agent"),
                "rca-agent-domain": _package_version("rca-agent"),
            },
            "frozen_at": frozen_at,
        }
        artifact = AnswerArtifact(
            **fields,
            artifact_digest=sha256_digest(_jsonable(fields)),
        )
        self.__answer = artifact
        self.status = EpisodeStatus.ANSWER_FROZEN
        self.transcript.append(EventType.ANSWER_FROZEN, artifact)
        return artifact

    def _record_mechanism_coverage_shadow(self) -> None:
        """Append one replayable diagnostic without changing inference state."""

        if self.__coverage_mode == "off":
            return
        assert self.__coverage_mode == "shadow"
        assert self.__bootstrap is not None
        coverage_input = build_mechanism_coverage_input(
            self.investigation_workspace_snapshot,
            self.__observation_ledger.values(),
            self.__bootstrap,
            formal_candidates=project_formal_coverage_candidates(self.transcript.events),
        )
        assessment = MechanismCoveragePolicy().evaluate(coverage_input)
        self.transcript.append(
            EventType.MECHANISM_COVERAGE_SHADOW_RECORDED,
            MechanismCoverageRecord(
                mode="shadow",
                input=coverage_input,
                assessment=assessment,
            ),
        )

    def begin_evaluation(self) -> None:
        self._require_status(EpisodeStatus.ANSWER_FROZEN)
        if self.__answer is None or not self.__answer.verify_digest():
            raise EpisodeStateError("a valid frozen answer is required for evaluation")
        if any(item.event_type == EventType.EVALUATION_STARTED for item in self.transcript.events):
            raise EpisodeStateError("evaluation was already started")
        self.transcript.append(
            EventType.EVALUATION_STARTED,
            {"answer_artifact_digest": self.__answer.artifact_digest},
        )

    def complete_evaluation(self, evaluation_artifact_digest: str) -> None:
        self._require_status(EpisodeStatus.ANSWER_FROZEN)
        if not self.transcript.events or (
            self.transcript.events[-1].event_type != EventType.EVALUATION_STARTED
        ):
            raise EpisodeStateError("evaluation must be started before completion")
        self.transcript.append(
            EventType.EVALUATION_COMPLETED,
            {"evaluation_artifact_digest": evaluation_artifact_digest},
        )
        self.status = EpisodeStatus.EVALUATED

    def _call_tool(self, action: CallToolAction) -> ToolResponse | AnswerArtifact:
        source = self._admit_tool_action(action)
        query_ref = query_fingerprint(action.tool_name, action.request)
        self.transcript.append(
            EventType.TOOL_REQUEST,
            {
                "tool_name": action.tool_name,
                "request": action.request.model_dump(mode="json"),
                "query_ref": query_ref,
                "budget_before": self.__budget_remaining,
                **({"execution_source": source} if source is not None else {}),
            },
        )
        if source == "startup":
            self.__startup_selection = None
        tool = getattr(self.__environment, action.tool_name.value)
        try:
            response: ToolResponse = tool(action.request)
        except Exception as exc:  # noqa: BLE001 - exception type only crosses the transcript
            self.transcript.append(
                EventType.TOOL_EXCEPTION,
                {"tool_name": action.tool_name, "error_type": type(exc).__name__},
            )
            self.transcript.append(
                EventType.STOP_REQUESTED,
                {"reason": StopReason.TOOL_FAILURE},
            )
            return self.freeze_answer(StopReason.TOOL_FAILURE)
        self.transcript.append(EventType.TOOL_RESPONSE, response)
        self.transcript.append(
            EventType.QUERY_COST,
            {
                "query_ref": response.query_ref,
                **response.cost.model_dump(mode="json"),
            },
        )
        self.__budget_remaining = response.cost.budget_after
        if any(
            item.observation_id in self.__observation_ledger
            and self.__observation_ledger[item.observation_id] != IssuedObservation(
                item, response.query_ref, not response.truncated and response.status == ToolStatus.SUCCESS
            )
            for item in response.observations
        ):
            self.transcript.append(EventType.TOOL_EXCEPTION, {
                "error_type": "issued_observation_changed", "tool_name": action.tool_name,
            })
            return self.freeze_answer(StopReason.TOOL_FAILURE)
        self.__observation_refs.update(item.observation_id for item in response.observations)
        self.__observation_ledger.update(
            {
                item.observation_id: IssuedObservation(
                    item, response.query_ref,
                    not response.truncated and response.status == ToolStatus.SUCCESS,
                )
                for item in response.observations
            }
        )
        self.__issued_entities.update(_find_entity_refs(response.model_dump(mode="json")))
        if response.query_ref != query_ref:
            raise RuntimeError("environment query fingerprint diverged from the request")
        self._complete_active_discrimination(response)
        if response.status == ToolStatus.BUDGET_EXHAUSTED:
            self.transcript.append(
                EventType.STOP_REQUESTED,
                {"reason": StopReason.BUDGET_EXHAUSTED},
            )
            return self.freeze_answer(StopReason.BUDGET_EXHAUSTED)
        if response.status == ToolStatus.QUERY_FAILED:
            self.transcript.append(
                EventType.TOOL_EXCEPTION,
                {"tool_name": action.tool_name, "error_type": "modeled_query_failure"},
            )
            self.transcript.append(
                EventType.STOP_REQUESTED,
                {"reason": StopReason.TOOL_FAILURE},
            )
            return self.freeze_answer(StopReason.TOOL_FAILURE)
        return response

    def _admit_tool_action(self, action: CallToolAction) -> str | None:
        if self.__inquiry_loop_mode != "batched":
            return None
        from rca_agent.investigation.inquiry_loop import LoopState, admitted_batch_member

        state = LoopState.model_validate(self.investigation_workspace_snapshot.inquiry_loop)
        grant = self.__tool_grant
        if grant is not None and grant[1] == action:
            source = grant[0]
            if self.__tool_grant_validated:
                return source
            if source == "startup":
                from rca_agent.investigation.startup_authorization import (
                    selected_startup_action,
                    startup_grant,
                )
                selected = startup_grant(self.transcript.events)
                if (self.__startup_selection is not None
                        and self.__startup_selection[0] == action
                        and selected is not None and selected["action"] == action.model_dump(mode="json")
                        and selected["selection_event_digest"] == self.__startup_selection[1]
                        and selected_startup_action(state, self.__bootstrap,
                                                   self.transcript.events) == action):
                    self.__tool_grant_validated = True
                    return source
            elif source == "verification" and state.phase == "handoff" and (
                (bool(self.__verification_queries) and action == self.__verification_queries[0][1])
                or any(action == candidate for _, _, candidate in self.feedback_query_options())
            ):
                self.__tool_grant_validated = True
                return source
            raise ActionRejectedError("query has no valid host execution grant")
        try:
            admitted_batch_member(state, action)
        except ValueError as exc:
            raise ActionRejectedError(str(exc)) from exc
        return "inquiry_batch"

    def _complete_active_discrimination(self, response: ToolResponse) -> None:
        from rca_agent.investigation.contract_enforcement import execution_binding_responses
        eligible_bindings = (execution_binding_responses(self.transcript.events)
                             if self.__engineering_revision == 2 else None)
        if self.__workspace_mode == "shadow":
            for task in self.__investigation_workspace.tasks:
                if (task.status != DiscriminationTaskStatus.OPEN or not task.source_binding_id
                        or task.legacy_query_ref != response.query_ref or not task.capability_options):
                    continue
                if eligible_bindings is not None and task.source_binding_id not in eligible_bindings:
                    continue
                option = task.capability_options[0]
                sequence = len(self.transcript.events) + 1
                attempt = DiscriminationAttempt(
                    attempt_id=discrimination_attempt_id(self.episode_id, task.task_id,
                                                       option.capability_ref, response.query_ref),
                    task_id=task.task_id, capability_ref=option.capability_ref,
                    query_ref=response.query_ref, query_fingerprint=response.query_ref,
                    query_cost=option.query_cost, independent_group=option.independent_group,
                    registry_digest=task.last_evaluated_registry_digest,
                    selection_reason_codes=("DECLARED_SHADOW_BINDING",), selected_sequence=sequence)
                self._append_workspace_event(EventType.DISCRIMINATION_TASK_SELECTED,
                    DiscriminationTaskSelected(task_id=task.task_id, selection_mode="shadow",
                                               attempt=attempt, sequence=sequence))
                usable = response.status == ToolStatus.SUCCESS and bool(response.observations)
                self._append_workspace_event(EventType.DISCRIMINATION_TASK_COMPLETED,
                    DiscriminationTaskCompleted(task_id=task.task_id, attempt_id=attempt.attempt_id,
                        to_status=(DiscriminationTaskStatus.AMBIGUOUS if usable
                                   else DiscriminationTaskStatus.UNVERIFIABLE),
                        observation_refs=tuple(o.observation_id for o in response.observations),
                        tool_status=response.status.value,
                        outcome=(AttemptOutcome.AMBIGUOUS if usable else AttemptOutcome.NO_DATA
                                 if response.status == ToolStatus.SUCCESS else AttemptOutcome.QUERY_FAILED),
                        reason_code="SHADOW_OBSERVATION_NOT_SEMANTICALLY_ADJUDICATED",
                        sequence=len(self.transcript.events) + 1))
            return
        if (
            self.__workspace_mode != "active"
            or self.__active_selected_task_id is None
            or self.__active_selected_attempt_id is None
        ):
            return
        task = self.__investigation_workspace.task(self.__active_selected_task_id)
        if task is None:
            raise RuntimeError("active discrimination task disappeared")
        if (eligible_bindings is not None and task.source_binding_id is not None
                and task.source_binding_id not in eligible_bindings):
            raise ValueError("task response predates its execution binding")
        attempt = next(
            (item for item in task.attempts if item.attempt_id == self.__active_selected_attempt_id),
            None,
        )
        if attempt is None or attempt.query_ref != response.query_ref:
            return
        observations = tuple(item.observation_id for item in response.observations)
        success = response.status == ToolStatus.SUCCESS
        to_status = (
            DiscriminationTaskStatus.AMBIGUOUS
            if success and observations else DiscriminationTaskStatus.UNVERIFIABLE
        )
        outcome = (
            AttemptOutcome.AMBIGUOUS
            if success and observations
            else (AttemptOutcome.NO_DATA if success else AttemptOutcome.QUERY_FAILED)
        )
        sequence = len(self.transcript.events) + 1
        self._append_workspace_event(
            EventType.DISCRIMINATION_TASK_COMPLETED,
            DiscriminationTaskCompleted(
                task_id=task.task_id,
                attempt_id=attempt.attempt_id,
                to_status=to_status,
                observation_refs=observations,
                tool_status=response.status.value,
                outcome=outcome,
                reason_code=(
                    "ACTIVE_OBSERVATION_NOT_SEMANTICALLY_ADJUDICATED"
                    if to_status == DiscriminationTaskStatus.AMBIGUOUS
                    else "QUERY_PRODUCED_NO_USABLE_DISCRIMINATOR"
                ),
                sequence=sequence,
            ),
        )
        self.__active_selected_task_id = None
        self.__active_selected_attempt_id = None
        for shared in self.__investigation_workspace.tasks:
            if (shared.status != DiscriminationTaskStatus.SELECTED
                    or shared.legacy_query_ref != response.query_ref):
                continue
            if (eligible_bindings is not None and shared.source_binding_id is not None
                    and shared.source_binding_id not in eligible_bindings):
                continue
            self._append_workspace_event(EventType.DISCRIMINATION_TASK_COMPLETED,
                DiscriminationTaskCompleted(task_id=shared.task_id,
                    attempt_id=shared.selected_attempt_id, to_status=to_status,
                    observation_refs=observations, tool_status=response.status.value,
                    outcome=outcome, reason_code="SHARED_OBSERVATION_NOT_SEMANTICALLY_ADJUDICATED",
                    sequence=len(self.transcript.events) + 1))

    def _submit_hypothesis(self, proposal: HypothesisProposal) -> None:
        unknown_observations = sorted(set(proposal.observation_refs) - self.__observation_refs)
        if unknown_observations or proposal.causal_entity_ref not in self.__issued_entities:
            self.transcript.append(
                EventType.ACTION_REJECTED,
                {
                    "error_type": "unissued_reference",
                    "proposal_digest": sha256_digest(proposal),
                },
            )
            raise ActionRejectedError("hypothesis references values not issued by this episode")
        if proposal.hypothesis_id in {item.hypothesis_id for item in self.__proposals}:
            self.transcript.append(
                EventType.ACTION_REJECTED,
                {"error_type": "duplicate_hypothesis_id", "hypothesis_id": proposal.hypothesis_id},
            )
            raise ActionRejectedError("hypothesis ID was already submitted")
        self.__proposals.append(proposal)
        self.transcript.append(EventType.HYPOTHESIS_SUBMITTED, proposal)

    def _update_candidate_evidence(self, action: UpdateCandidateEvidenceAction) -> None:
        unknown_observations = sorted(set(action.observation_refs) - self.__observation_refs)
        if unknown_observations:
            self.transcript.append(
                EventType.ACTION_REJECTED,
                {
                    "error_type": "unissued_reference",
                    "hypothesis_id": action.hypothesis_id,
                    "action_digest": sha256_digest(action),
                },
            )
            raise ActionRejectedError("candidate evidence references values not issued by this episode")
        index = next(
            (
                item_index
                for item_index, proposal in enumerate(self.__proposals)
                if proposal.hypothesis_id == action.hypothesis_id
            ),
            None,
        )
        if index is None:
            self.transcript.append(
                EventType.ACTION_REJECTED,
                {
                    "error_type": "unknown_hypothesis_id",
                    "hypothesis_id": action.hypothesis_id,
                },
            )
            raise ActionRejectedError("candidate evidence update requires a submitted hypothesis")
        proposal = self.__proposals[index]
        updated = proposal.model_copy(
            update={
                "observation_refs": tuple(
                    sorted({*proposal.observation_refs, *action.observation_refs})
                ),
                "missing_evidence": tuple(
                    sorted({*proposal.missing_evidence, *action.missing_evidence})
                ),
            }
        )
        self.__proposals[index] = updated
        self.transcript.append(
            EventType.CANDIDATE_EVIDENCE_UPDATED,
            {
                "hypothesis_id": action.hypothesis_id,
                "observation_refs": action.observation_refs,
                "missing_evidence": action.missing_evidence,
            },
        )

    def _require_status(self, expected: EpisodeStatus) -> None:
        if self.status != expected:
            raise EpisodeStateError(
                f"operation requires {expected.value}, current status is {self.status.value}"
            )


def _package_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "unknown"


def _normalized_semantic_text(value: str) -> str:
    return " ".join(value.casefold().split())


def _jsonable(value: dict[str, Any]) -> dict[str, Any]:
    from pydantic import TypeAdapter

    return TypeAdapter(dict[str, Any]).dump_python(value, mode="json")


def _find_entity_refs(value: Any) -> set[str]:
    if isinstance(value, dict):
        return {item for nested in value.values() for item in _find_entity_refs(nested)}
    if isinstance(value, list | tuple):
        return {item for nested in value for item in _find_entity_refs(nested)}
    if isinstance(value, str) and ENTITY_REF_RE.fullmatch(value):
        return {value}
    return set()
