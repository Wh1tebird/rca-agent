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
)
from rca_agent.runner import EpisodeRunner


class InvestigationGraphState(TypedDict, total=False):
    history: tuple[dict[str, Any], ...]
    action: PolicyAction
    core_assessment: CoreAssessmentPreview
    candidate_frontier: CandidateFrontier
    investigation_workspace: InvestigationWorkspaceSnapshot
    answer: AnswerArtifact
    discovery_requested: bool
    deferred_stop: bool
    feedback_attempted: bool


class LangGraphRunReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    answer: AnswerArtifact
    graph_steps: int = Field(ge=1)
    history: tuple[dict[str, Any], ...]
    final_preview: CoreAssessmentPreview
    final_frontier: CandidateFrontier
    final_workspace: InvestigationWorkspaceSnapshot


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
        self._bootstrap = None
        self._candidate_options: tuple[CandidateOption, ...] = ()
        self._graph_steps = 0
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
        state = self._graph.invoke(
            {
                "history": (),
                "core_assessment": initial_preview,
                "candidate_frontier": initial_frontier,
                "investigation_workspace": initial_workspace,
            },
            {"recursion_limit": recursion_limit},
        )
        answer = state.get("answer")
        if answer is None:
            raise RuntimeError("LangGraph completed without a frozen answer")
        return LangGraphRunReport(
            answer=answer,
            graph_steps=self._graph_steps,
            history=state.get("history", ()),
            final_preview=state.get("core_assessment", initial_preview),
            final_frontier=state.get("candidate_frontier", initial_frontier),
            final_workspace=(
                answer.investigation_workspace
                or state.get("investigation_workspace", initial_workspace)
            ),
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

    def _policy_node(self, state: InvestigationGraphState) -> InvestigationGraphState:
        self._graph_steps += 1
        assert self._bootstrap is not None
        try:
            if self._runner.inquiry_loop_mode == "batched":
                self._runner.select_startup_query()
            action = None
            for _ in range(12):
                try:
                    proposed = self._policy.next_action(
                        PolicyContext(
                            bootstrap=self._bootstrap,
                            candidate_options=self._candidate_options,
                            history=state.get("history", ()),
                            core_assessment=state["core_assessment"],
                            candidate_frontier=state["candidate_frontier"],
                            investigation_workspace=(
                                self._runner.investigation_workspace_snapshot.model_dump(mode="json")
                                if self._runner.inquiry_loop_mode == "batched" else None),
                            inquiry_state_view=(self._runner.inquiry_loop_view()
                                if self._runner.inquiry_loop_mode == "batched" else None),
                        )
                    )
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
        return {"action": action, "discovery_requested": isinstance(action, RequestStopAction)
                and self._runner.should_discover(stopping=True)}

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
        preview = self._runner.preview_core_assessment()
        return {
            "core_assessment": preview,
            "candidate_frontier": _candidate_frontier(preview, self._candidate_options),
            "investigation_workspace": self._runner.investigation_workspace_snapshot,
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

    @staticmethod
    def _route_after_policy(state: InvestigationGraphState) -> Literal["execute", "discover", "end"]:
        if "answer" in state:
            return "end"
        return "discover" if state.get("discovery_requested") else "execute"

    @staticmethod
    def _route_after_execute(state: InvestigationGraphState) -> Literal["assess", "end"]:
        return "end" if "answer" in state else "assess"

    def _route_after_assess(self, state: InvestigationGraphState):
        if self._runner.has_verification_queries:
            return "verify_discovery"
        if (
            state["core_assessment"].overall_decision.status
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
        if (state["core_assessment"].overall_decision.status
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
