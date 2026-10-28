from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict, Field

from rca_agent.core.v3.catalog import mechanism_catalog
from rca_agent.investigation import InvestigationWorkspaceSnapshot
from rca_agent.langchain_policy import (
    CandidateOption,
    LangChainPolicyAdapter,
    PolicyContext,
)
from rca_agent.models import (
    AnswerArtifact,
    CallToolAction,
    CandidateEvidenceStatus,
    CandidateFrontier,
    CoreAssessmentPreview,
    DecisionStatus,
    EpisodeStatus,
    PolicyAction,
    RequestStopAction,
    StopReason,
    ToolResponse,
    UpdateCandidateEvidenceAction,
    parse_policy_action,
)
from rca_agent.reasoning import (
    ReasoningAdapter,
    ReasoningFailure,
    ReasoningRequest,
    ReasoningResponse,
)
from rca_agent.runner import EpisodeRunner
from rca_agent.tools.fingerprint import sha256_digest

ROUTING_PROJECTION_FIELDS = frozenset({
    "stage",
    "needs_continue",
    "next_action_ref",
    "candidate_frontier",
    "stop_signal",
    "handoff_signal",
    "core_assessment",
    "reasoning_request_ref",
    "reasoning_response_ref",
})
"""Fields that are safe for LangGraph to retain as a routing projection."""

COMPATIBILITY_CACHE_FIELDS = frozenset({
    "history",
    "core_assessment",
    "candidate_frontier",
    "investigation_workspace",
    "answer",
})
"""Legacy state fields retained for API compatibility, never as fact sources."""

EPHEMERAL_ROUTING_FIELDS = frozenset({
    "action",
    "discovery_requested",
    "deferred_stop",
    "feedback_attempted",
})
"""One-step proposal/routing caches; authority remains in EpisodeRunner."""


class InvestigationGraphState(TypedDict, total=False):
    """LangGraph state with an explicit non-authoritative compatibility boundary.

    ``routing_projection`` is the only durable graph projection.  The flat fields
    below are retained because callers and old graph checkpoints still expose
    them, but they are compatibility/ephemeral caches.  They must never be used
    to authorize a tool call, admission, stop, decision, or answer mutation.
    """

    # Compatibility cache: historical graph/debug output only.
    history: tuple[dict[str, Any], ...]
    # Ephemeral proposal cache, validated and executed by EpisodeRunner.
    action: PolicyAction
    # Compatibility cache: the latest Runner-owned preview/frontier projection.
    core_assessment: CoreAssessmentPreview
    candidate_frontier: CandidateFrontier
    # Compatibility cache: Workspace is Runner/transcript-derived.
    investigation_workspace: InvestigationWorkspaceSnapshot
    # Compatibility cache: a frozen Runner artifact, never a graph-owned answer.
    answer: AnswerArtifact
    # Ephemeral routing suggestions, rechecked against Runner state by routers.
    discovery_requested: bool
    deferred_stop: bool
    feedback_attempted: bool
    # Authoritative read-only routing projection supplied by Runner queries.
    routing_projection: RoutingProjection


class RoutingProjection(BaseModel):
    """Read-only routing projection derived from the EpisodeRunner.

    It carries references and previews only.  It contains no Workspace ledger,
    budget, permission, Answer, or decision authority owned by the graph.
    The graph must return every proposal to the Runner for validation and
    execution; this projection is never a second source of control facts.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    stage: str
    needs_continue: bool
    next_action_ref: str | None = None
    candidate_frontier: CandidateFrontier
    stop_signal: str | None = None
    handoff_signal: str | None = None
    core_assessment: CoreAssessmentPreview
    reasoning_request_ref: str | None = None
    reasoning_response_ref: str | None = None


class _PolicyReasoningAdapter:
    """Typed bridge from the legacy Policy adapter to the reasoning facade."""

    def __init__(self, policy: LangChainPolicyAdapter) -> None:
        self._policy = policy

    def invoke(self, request: ReasoningRequest) -> ReasoningResponse:
        if request.role != "inquiry" or request.operation != "next_action":
            return ReasoningResponse(
                role="inquiry",
                operation=request.operation,
                kind="next_action",
                proposal={},
                failure=ReasoningFailure(
                    kind="unsupported_operation", phase="inquiry", retryable=False,
                ),
            )
        try:
            context = PolicyContext.model_validate(request.context)
            action = self._policy.next_action(context)
        except Exception as exc:  # noqa: BLE001 - typed facade boundary
            return ReasoningResponse(
                role="inquiry",
                operation=request.operation,
                kind="next_action",
                proposal={},
                failure=ReasoningFailure(
                    kind=type(exc).__name__, phase="inquiry", retryable=True,
                ),
            )
        return ReasoningResponse(
            role="inquiry",
            operation=request.operation,
            kind="next_action",
            proposal={"action": action.model_dump(mode="json")},
        )


class LangGraphRunReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    answer: AnswerArtifact
    graph_steps: int = Field(ge=1)
    history: tuple[dict[str, Any], ...]
    final_preview: CoreAssessmentPreview
    final_frontier: CandidateFrontier
    final_workspace: InvestigationWorkspaceSnapshot
    routing_projection: RoutingProjection | None = None


class LangGraphInvestigation:
    """Non-authoritative orchestration around the existing trusted EpisodeRunner."""

    def __init__(
        self,
        runner: EpisodeRunner,
        policy: LangChainPolicyAdapter,
        discovery_notes_provider: Callable[[tuple[dict[str, Any], ...]],
                                           dict[str, Any] | None] | None = None,
        inquiry_rounds_provider: Callable[[], tuple[dict[str, Any], ...]] | None = None,
        enable_core_feedback: bool = False,
        reasoning_adapter: ReasoningAdapter | None = None,
    ) -> None:
        self._runner = runner
        self._policy = policy
        self._discovery_notes_provider = discovery_notes_provider
        self._inquiry_rounds_provider = inquiry_rounds_provider
        if runner.inquiry_loop_mode == "batched" and inquiry_rounds_provider is not None:
            raise ValueError("batched Inquiry records rounds through its accepted host steps")
        if runner.inquiry_loop_mode != policy.inquiry_loop_mode:
            raise ValueError("Inquiry loop mode differs between host and Policy")
        if (runner.inquiry_loop_mode == "batched"
                and runner.inquiry_query_limit != policy.inquiry_query_limit):
            raise ValueError("Inquiry query limit differs between host and Policy")
        self._recorded_inquiry_rounds = 0
        self._enable_core_feedback = enable_core_feedback
        self._reasoning_adapter = reasoning_adapter or _PolicyReasoningAdapter(policy)
        self._bootstrap = None
        self._candidate_options: tuple[CandidateOption, ...] = ()
        self._graph_steps = 0
        self._last_preview: CoreAssessmentPreview | None = None
        self._last_reasoning_request_ref: str | None = None
        self._last_reasoning_response_ref: str | None = None
        self._graph = self._build_graph()

    def run(self, *, recursion_limit: int = 100) -> LangGraphRunReport:
        if self._runner.status != EpisodeStatus.CREATED:
            raise ValueError("LangGraph investigation requires a newly created EpisodeRunner")
        self._bootstrap = self._runner.start()
        self._candidate_options = _candidate_options(
            self._bootstrap.alert_entity.entity_ref,
            self._runner.excluded_catalog_candidate_ids,
        )
        initial_preview = self._runner.preview_core_assessment()
        initial_frontier = _candidate_frontier(initial_preview, self._candidate_options)
        initial_workspace = self._runner.investigation_workspace_snapshot
        self._last_preview = initial_preview
        state = self._graph.invoke(
            {
                "history": (),
                "core_assessment": initial_preview,
                "candidate_frontier": initial_frontier,
                "investigation_workspace": initial_workspace,
                "routing_projection": self._routing_projection(),
            },
            {"recursion_limit": recursion_limit},
        )
        # The graph may carry an answer cache, but only the Runner can freeze
        # one.  Read it back from the authority after graph execution.
        answer = self._runner.answer
        if answer is None:
            raise RuntimeError("LangGraph completed without a frozen answer")
        final_preview = self._last_preview or initial_preview
        final_frontier = _candidate_frontier(final_preview, self._candidate_options)
        projection = self._routing_projection(
            stop_signal=self._runner.status.value,
            next_action_ref=None,
            reasoning_request_ref=self._last_reasoning_request_ref,
            reasoning_response_ref=self._last_reasoning_response_ref,
        )
        return LangGraphRunReport(
            answer=answer,
            graph_steps=self._graph_steps,
            history=state.get("history", ()),
            final_preview=final_preview,
            final_frontier=final_frontier,
            final_workspace=(
                answer.investigation_workspace
                or state.get("investigation_workspace", initial_workspace)
            ),
            routing_projection=projection,
        )

    def _build_graph(self):
        builder = StateGraph(InvestigationGraphState)
        builder.add_node("policy", self._policy_node)
        builder.add_node("execute", self._execute_node)
        builder.add_node("assess", self._assess_node)
        builder.add_node("finalize", self._finalize_node)
        builder.add_node("discover", self._discovery_node)
        builder.add_node("verify_discovery", self._verify_discovery_node)
        builder.add_node("feedback", self._feedback_node)
        builder.add_edge(START, "policy")
        builder.add_conditional_edges(
            "policy",
            self._route_after_policy,
            {"execute": "execute", "discover": "discover", "end": END},
        )
        builder.add_conditional_edges(
            "execute",
            self._route_after_execute,
            {"assess": "assess", "end": END},
        )
        builder.add_conditional_edges(
            "assess",
            self._route_after_assess,
            {"policy": "policy", "finalize": "finalize", "discover": "discover",
             "verify_discovery": "verify_discovery", "feedback": "feedback"},
        )
        builder.add_edge("discover", "assess")
        builder.add_conditional_edges("verify_discovery", self._route_after_execute,
                                      {"assess": "assess", "end": END})
        builder.add_conditional_edges("feedback", self._route_after_execute,
                                      {"assess": "assess", "end": END})
        builder.add_edge("finalize", END)
        return builder.compile(name="rca-agent-investigation")

    def _routing_projection(
        self,
        *,
        next_action_ref: str | None = None,
        stop_signal: str | None = None,
        reasoning_request_ref: str | None = None,
        reasoning_response_ref: str | None = None,
    ) -> RoutingProjection:
        """Read the current routing view from Runner-owned state.

        ``preview_core_assessment`` is available only while an episode is
        active.  Once the Runner freezes an answer, the last active preview is
        retained as an immutable compatibility cache for reporting; no graph
        state is consulted to reconstruct a decision.
        """

        if self._runner.status == EpisodeStatus.ACTIVE:
            preview = self._runner.preview_core_assessment()
            self._last_preview = preview
        elif self._last_preview is not None:
            preview = self._last_preview
        else:
            raise RuntimeError("routing projection requested before Runner preview")
        frontier = _candidate_frontier(preview, self._candidate_options)
        request_ref = reasoning_request_ref or (
            "reasoning-request-"
            + sha256_digest({
                "episode_id": self._runner.episode_id,
                "stage": self._runner.status.value,
                "preview": sha256_digest(preview.model_dump(mode="json")),
                "frontier": sha256_digest(frontier.model_dump(mode="json")),
            })[:20]
        )
        self._last_reasoning_request_ref = request_ref
        if reasoning_response_ref is not None:
            self._last_reasoning_response_ref = reasoning_response_ref
        handoff_signal = (
            "inquiry_handoff"
            if self._runner.inquiry_loop_phase == "handoff"
            else None
        )
        return RoutingProjection(
            stage=self._runner.status.value,
            needs_continue=self._runner.status == EpisodeStatus.ACTIVE,
            next_action_ref=next_action_ref,
            candidate_frontier=frontier,
            stop_signal=stop_signal,
            handoff_signal=handoff_signal,
            core_assessment=preview,
            reasoning_request_ref=request_ref,
            reasoning_response_ref=reasoning_response_ref,
        )

    def _policy_node(self, state: InvestigationGraphState) -> InvestigationGraphState:
        self._graph_steps += 1
        assert self._bootstrap is not None
        try:
            projection = self._routing_projection()
            if self._runner.inquiry_loop_mode == "batched":
                self._runner.select_startup_query()
            action = None
            for _ in range(12):
                try:
                    policy_context = PolicyContext(
                        bootstrap=self._bootstrap,
                        candidate_options=self._candidate_options,
                        history=state.get("history", ()),
                        core_assessment=projection.core_assessment,
                        candidate_frontier=projection.candidate_frontier,
                        investigation_workspace=(
                            self._runner.investigation_workspace_snapshot.model_dump(mode="json")
                            if self._runner.inquiry_loop_mode == "batched" else None),
                        inquiry_state_view=(self._runner.inquiry_loop_view()
                            if self._runner.inquiry_loop_mode == "batched" else None),
                    )
                    request = ReasoningRequest(
                        role="inquiry",
                        operation="next_action",
                        context=policy_context.model_dump(mode="json"),
                    )
                    reasoning_response = self._reasoning_adapter.invoke(request)
                    if reasoning_response.failure is not None:
                        raise RuntimeError(reasoning_response.failure.kind)
                    if reasoning_response.kind != "next_action":
                        raise ValueError("reasoning adapter returned a non-action proposal")
                    proposed = parse_policy_action(reasoning_response.proposal["action"])
                finally:
                    for diagnostic in self._policy.drain_inquiry_diagnostics():
                        self._runner.record_inquiry_attempt(diagnostic)
                    for diagnostic in self._policy.drain_inquiry_loop_attempts():
                        self._runner.record_inquiry_loop_attempt(diagnostic)
                for step, view in self._policy.drain_inquiry_loop_steps():
                    self._runner.record_inquiry_loop_step(step, input_view=view)
                if self._inquiry_rounds_provider is not None:
                    rounds = self._inquiry_rounds_provider()
                    for record in rounds[self._recorded_inquiry_rounds:]:
                        self._runner.record_inquiry_round(record)
                        self._recorded_inquiry_rounds += 1
                action = self._runner.route_active_inquiry_action(proposed)
                self._policy.record_routed_action(proposed, action)
                if action is not None:
                    break
            if action is None:
                raise RuntimeError("active inquiry routing suppressed too many queued actions")
        except Exception as exc:  # noqa: BLE001 - only the error type crosses the trust boundary
            self._runner.reject_raw_action(
                {"source": "langchain_policy"},
                error_type=type(exc).__name__,
            )
            answer = self._runner.terminate_policy_failure(
                StopReason.POLICY_INVALID_ACTION,
                type(exc).__name__,
            )
            return {"answer": answer}
        action_ref = "action-" + sha256_digest(action.model_dump(mode="json"))[:20]
        reasoning_response_ref = (
            "reasoning-response-"
            + sha256_digest(reasoning_response.model_dump(mode="json"))[:20]
        )
        self._last_reasoning_request_ref = (
            "reasoning-request-" + sha256_digest(request.model_dump(mode="json"))[:20]
        )
        self._last_reasoning_response_ref = reasoning_response_ref
        return {
            "action": action,
            "discovery_requested": isinstance(action, RequestStopAction)
            and self._runner.should_discover(stopping=True),
            "routing_projection": self._routing_projection(
                next_action_ref=action_ref,
                reasoning_request_ref=self._last_reasoning_request_ref,
                reasoning_response_ref=reasoning_response_ref,
            ),
        }

    def _discovery_node(self, state: InvestigationGraphState) -> InvestigationGraphState:
        self._graph_steps += 1
        notes = (self._discovery_notes_provider(state.get("history", ()))
                 if self._discovery_notes_provider is not None else None)
        if self._runner.inquiry_loop_mode == "batched":
            notes = self._runner.inquiry_loop_handoff()
        result = self._runner.discover_candidates(inquiry_handoff=notes)
        assert self._bootstrap is not None
        self._candidate_options = (
            *_candidate_options(
                self._bootstrap.alert_entity.entity_ref,
                self._runner.excluded_catalog_candidate_ids,
            ),
            *(CandidateOption.model_validate(o) for o in self._runner.dynamic_candidate_options),
        )
        return {"discovery_requested": False,
                "deferred_stop": state.get("discovery_requested", False),
                "history": (*state.get("history", ()), {"kind": "discovery", "result": result})}

    def _verify_discovery_node(self, state: InvestigationGraphState) -> InvestigationGraphState:
        self._graph_steps += 1
        result = self._runner.verify_next_discovered_query()
        if isinstance(result, AnswerArtifact):
            return {"answer": result}
        return {"history": (*state.get("history", ()), {
            "kind": "discovery_query_completed",
            "result": result.model_dump(mode="json") if result is not None else None,
        })}

    def _feedback_node(self, state: InvestigationGraphState) -> InvestigationGraphState:
        self._graph_steps += 1
        if self._runner.feedback_plan_available():
            planned = self._runner.extend_verification_plan()
            return {"feedback_attempted": True,
                    "history": (*state.get("history", ()),
                                {"kind": "core_feedback_plan", "result": planned})}
        records = []
        for hypothesis_id, signal, action in self._runner.feedback_query_options()[:2]:
            response = self._runner.verify_feedback_query(hypothesis_id, signal, action)
            if isinstance(response, AnswerArtifact):
                return {"answer": response, "feedback_attempted": True}
            observation_refs = ()
            if isinstance(response, ToolResponse) and response.observations:
                observation_refs = tuple(item.observation_id for item in response.observations)
                update = self._runner.handle(UpdateCandidateEvidenceAction(
                    hypothesis_id=hypothesis_id, observation_refs=observation_refs,
                ))
                if isinstance(update, AnswerArtifact):
                    return {"answer": update, "feedback_attempted": True}
            records.append({"hypothesis_id": hypothesis_id, "signal": signal,
                            "observation_refs": observation_refs})
            preview = self._runner.preview_core_assessment()
            if preview.overall_decision.status in {
                DecisionStatus.CONFIRMED_ROOT_CAUSE, DecisionStatus.DISCOVERED_VERIFIED,
            }:
                break
        return {"feedback_attempted": True,
                "history": (*state.get("history", ()), {"kind": "core_feedback",
                                                  "queries": records})}

    def _execute_node(self, state: InvestigationGraphState) -> InvestigationGraphState:
        self._graph_steps += 1
        action = state["action"]
        if (self._runner.inquiry_loop_mode == "batched"
                and isinstance(action, CallToolAction)
                and self._policy.consume_startup_action(action)):
            result = self._runner.handle_startup_query(action)
        else:
            result = self._runner.handle(action)
        if isinstance(result, AnswerArtifact):
            serialized_result: dict[str, Any] = {
                "kind": "answer",
                "value": result.model_dump(mode="json"),
            }
            answer = result
        elif isinstance(result, ToolResponse):
            serialized_result = {
                "kind": "tool_response",
                "value": result.model_dump(mode="json"),
            }
            answer = None
        elif isinstance(action, UpdateCandidateEvidenceAction):
            serialized_result = {"kind": "candidate_evidence_updated", "value": None}
            answer = None
        else:
            serialized_result = {"kind": "hypothesis_accepted", "value": None}
            answer = None
        record = {
            "action": action.model_dump(mode="json"),
            "result": serialized_result,
        }
        update: InvestigationGraphState = {
            "history": (*state.get("history", ()), record),
        }
        if answer is not None:
            update["answer"] = answer
        return update

    def _assess_node(self, state: InvestigationGraphState) -> InvestigationGraphState:
        del state
        self._graph_steps += 1
        projection = self._routing_projection()
        return {
            "core_assessment": projection.core_assessment,
            "candidate_frontier": projection.candidate_frontier,
            "investigation_workspace": self._runner.investigation_workspace_snapshot,
            "routing_projection": projection.model_copy(update={
                "stop_signal": (
                    None if self._runner.status == EpisodeStatus.ACTIVE
                    else self._runner.status.value
                ),
            }),
        }

    def _finalize_node(self, state: InvestigationGraphState) -> InvestigationGraphState:
        del state
        self._graph_steps += 1
        answer = self._runner.handle(
            RequestStopAction(summary="Core assessment or deferred policy stop completed")
        )
        if not isinstance(answer, AnswerArtifact):
            raise TypeError("Core-confirmed finalization did not freeze an answer")
        return {"answer": answer}

    def _route_after_policy(
        self, state: InvestigationGraphState,
    ) -> Literal["execute", "discover", "end"]:
        if self._runner.answer is not None or self._runner.status != EpisodeStatus.ACTIVE:
            return "end"
        return (
            "discover"
            if state.get("discovery_requested") and self._runner.should_discover(stopping=True)
            else "execute"
        )

    def _route_after_execute(
        self, state: InvestigationGraphState,
    ) -> Literal["assess", "end"]:
        del state
        return (
            "end"
            if self._runner.answer is not None or self._runner.status != EpisodeStatus.ACTIVE
            else "assess"
        )

    def _route_after_assess(self, state: InvestigationGraphState):
        projection = self._routing_projection()
        if self._runner.has_verification_queries:
            return "verify_discovery"
        if (
            projection.core_assessment.overall_decision.status
            in {
                DecisionStatus.CONFIRMED_ROOT_CAUSE,
                DecisionStatus.DISCOVERED_VERIFIED,
            }
        ):
            return "finalize"
        if (self._runner.inquiry_loop_mode == "batched"
                and self._runner.inquiry_loop_phase in {"awaiting_results", "awaiting_assessment"}):
            return "policy"
        if self._runner.should_discover():
            return "discover"
        if (self._enable_core_feedback and not state.get("feedback_attempted")
                and (self._runner.feedback_plan_available()
                     or self._runner.feedback_query_options())):
            return "feedback"
        if (projection.core_assessment.overall_decision.status
                == DecisionStatus.MULTIPLE_SUPPORTED):
            return "finalize"
        if state.get("deferred_stop"):
            return "finalize"
        return "policy"


def _candidate_options(
    entity_ref: str,
    excluded_candidate_ids: frozenset[str] = frozenset(),
) -> tuple[CandidateOption, ...]:
    return tuple(
        CandidateOption(
            candidate_id=item.candidate_id,
            fault_type=item.fault_type,
            family=item.family,
            required_signals=tuple(
                condition.signal
                for condition in item.conditions
                if condition.condition_id in item.positive_evidence_requirements
            ),
            required_capabilities=tuple(item.required_capabilities),
        )
        for item in mechanism_catalog(entity_ref)
        if item.candidate_id not in excluded_candidate_ids
    )


def _candidate_frontier(
    preview: CoreAssessmentPreview,
    options: tuple[CandidateOption, ...],
) -> CandidateFrontier:
    statuses = {item.candidate_id: item.evidence_status for item in preview.candidate_assessments}
    active_statuses = {
        CandidateEvidenceStatus.UNASSESSED,
        CandidateEvidenceStatus.INCONCLUSIVE,
        CandidateEvidenceStatus.SUPPORTED,
    }
    return CandidateFrontier(
        active_candidate_ids=tuple(
            sorted(item for item, status in statuses.items() if status in active_statuses)
        ),
        supported_candidate_ids=tuple(
            sorted(
                item
                for item, status in statuses.items()
                if status == CandidateEvidenceStatus.SUPPORTED
            )
        ),
        contradicted_candidate_ids=tuple(
            sorted(
                item
                for item, status in statuses.items()
                if status == CandidateEvidenceStatus.CONTRADICTED
            )
        ),
        inconclusive_candidate_ids=tuple(
            sorted(
                item
                for item, status in statuses.items()
                if status
                in {
                    CandidateEvidenceStatus.UNASSESSED,
                    CandidateEvidenceStatus.INCONCLUSIVE,
                }
            )
        ),
        confirmed_candidate_ids=tuple(
            sorted(
                item
                for item, status in statuses.items()
                if status == CandidateEvidenceStatus.CONFIRMED
            )
        ),
        discovered_verified_candidate_ids=tuple(
            sorted(item for item, status in statuses.items()
                   if status == CandidateEvidenceStatus.DISCOVERED_VERIFIED)
        ),
        unsubmitted_candidate_ids=tuple(
            sorted(item.candidate_id for item in options if item.candidate_id not in statuses)
        ),
    )
