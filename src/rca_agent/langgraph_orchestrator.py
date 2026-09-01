from __future__ import annotations

from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict, Field

from rca_agent.core.v3.catalog import mechanism_catalog
from rca_agent.langchain_policy import (
    CandidateOption,
    LangChainPolicyAdapter,
    PolicyContext,
)
from rca_agent.models import (
    AnswerArtifact,
    AuthoritativeDecision,
    DecisionStatus,
    EpisodeStatus,
    PolicyAction,
    RequestStopAction,
    StopReason,
    ToolResponse,
)
from rca_agent.runner import EpisodeRunner


class InvestigationGraphState(TypedDict, total=False):
    history: tuple[dict[str, Any], ...]
    action: PolicyAction
    core_assessment: AuthoritativeDecision
    answer: AnswerArtifact


class LangGraphRunReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    answer: AnswerArtifact
    graph_steps: int = Field(ge=1)
    history: tuple[dict[str, Any], ...]
    final_preview: AuthoritativeDecision


class LangGraphInvestigation:
    """Non-authoritative orchestration around the existing trusted EpisodeRunner."""

    def __init__(self, runner: EpisodeRunner, policy: LangChainPolicyAdapter) -> None:
        self._runner = runner
        self._policy = policy
        self._bootstrap = None
        self._candidate_options: tuple[CandidateOption, ...] = ()
        self._graph_steps = 0
        self._graph = self._build_graph()

    def run(self, *, recursion_limit: int = 100) -> LangGraphRunReport:
        if self._runner.status != EpisodeStatus.CREATED:
            raise ValueError("LangGraph investigation requires a newly created EpisodeRunner")
        self._bootstrap = self._runner.start()
        self._candidate_options = _candidate_options(self._bootstrap.alert_entity.entity_ref)
        initial_preview = self._runner.preview_core_assessment()
        state = self._graph.invoke(
            {
                "history": (),
                "core_assessment": initial_preview,
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
        )

    def _build_graph(self):
        builder = StateGraph(InvestigationGraphState)
        builder.add_node("policy", self._policy_node)
        builder.add_node("execute", self._execute_node)
        builder.add_node("assess", self._assess_node)
        builder.add_node("finalize", self._finalize_node)
        builder.add_edge(START, "policy")
        builder.add_conditional_edges(
            "policy",
            self._route_after_policy,
            {"execute": "execute", "end": END},
        )
        builder.add_conditional_edges(
            "execute",
            self._route_after_execute,
            {"assess": "assess", "end": END},
        )
        builder.add_conditional_edges(
            "assess",
            self._route_after_assess,
            {"policy": "policy", "finalize": "finalize"},
        )
        builder.add_edge("finalize", END)
        return builder.compile(name="rca-agent-investigation")

    def _policy_node(self, state: InvestigationGraphState) -> InvestigationGraphState:
        self._graph_steps += 1
        assert self._bootstrap is not None
        try:
            action = self._policy.next_action(
                PolicyContext(
                    bootstrap=self._bootstrap,
                    candidate_options=self._candidate_options,
                    history=state.get("history", ()),
                    core_assessment=state["core_assessment"],
                )
            )
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
        return {"action": action}

    def _execute_node(self, state: InvestigationGraphState) -> InvestigationGraphState:
        self._graph_steps += 1
        action = state["action"]
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
        return {"core_assessment": self._runner.preview_core_assessment()}

    def _finalize_node(self, state: InvestigationGraphState) -> InvestigationGraphState:
        del state
        self._graph_steps += 1
        answer = self._runner.handle(
            RequestStopAction(summary="internal Core preview has confirmed sufficient evidence")
        )
        if not isinstance(answer, AnswerArtifact):
            raise TypeError("Core-confirmed finalization did not freeze an answer")
        return {"answer": answer}

    @staticmethod
    def _route_after_policy(state: InvestigationGraphState) -> Literal["execute", "end"]:
        return "end" if "answer" in state else "execute"

    @staticmethod
    def _route_after_execute(state: InvestigationGraphState) -> Literal["assess", "end"]:
        return "end" if "answer" in state else "assess"

    @staticmethod
    def _route_after_assess(state: InvestigationGraphState) -> Literal["policy", "finalize"]:
        if state["core_assessment"].status == DecisionStatus.CONFIRMED_ROOT_CAUSE:
            return "finalize"
        return "policy"


def _candidate_options(entity_ref: str) -> tuple[CandidateOption, ...]:
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
    )
