from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from rca_agent.models.queries import (
    CompareMetricWindowsRequest,
    GetMetricSeriesRequest,
    GetServiceTraceSummaryRequest,
    GetTraceStageBreakdownRequest,
    InspectTopologyRequest,
    ListRelatedEntitiesRequest,
)


class ActionType(StrEnum):
    CALL_TOOL = "call_tool"
    SUBMIT_HYPOTHESIS = "submit_hypothesis"
    UPDATE_CANDIDATE_EVIDENCE = "update_candidate_evidence"
    REQUEST_STOP = "request_stop"


class ToolName(StrEnum):
    GET_METRIC_SERIES = "get_metric_series"
    COMPARE_METRIC_WINDOWS = "compare_metric_windows"
    LIST_RELATED_ENTITIES = "list_related_entities"
    GET_SERVICE_TRACE_SUMMARY = "get_service_trace_summary"
    GET_TRACE_STAGE_BREAKDOWN = "get_trace_stage_breakdown"
    INSPECT_TOPOLOGY = "inspect_topology"


ToolRequest: TypeAlias = (
    GetMetricSeriesRequest
    | CompareMetricWindowsRequest
    | ListRelatedEntitiesRequest
    | GetServiceTraceSummaryRequest
    | GetTraceStageBreakdownRequest
    | InspectTopologyRequest
)

_REQUEST_BY_TOOL: dict[ToolName, type[BaseModel]] = {
    ToolName.GET_METRIC_SERIES: GetMetricSeriesRequest,
    ToolName.COMPARE_METRIC_WINDOWS: CompareMetricWindowsRequest,
    ToolName.LIST_RELATED_ENTITIES: ListRelatedEntitiesRequest,
    ToolName.GET_SERVICE_TRACE_SUMMARY: GetServiceTraceSummaryRequest,
    ToolName.GET_TRACE_STAGE_BREAKDOWN: GetTraceStageBreakdownRequest,
    ToolName.INSPECT_TOPOLOGY: InspectTopologyRequest,
}


class CallToolAction(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    action_type: Literal[ActionType.CALL_TOOL] = ActionType.CALL_TOOL
    tool_name: ToolName
    request: ToolRequest

    @model_validator(mode="before")
    @classmethod
    def bind_request_to_tool(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        tool = ToolName(value.get("tool_name"))
        request_type = _REQUEST_BY_TOOL[tool]
        return {**value, "request": request_type.model_validate(value.get("request"))}

    @model_validator(mode="after")
    def validate_request_type(self) -> CallToolAction:
        if not isinstance(self.request, _REQUEST_BY_TOOL[self.tool_name]):
            raise TypeError("request type does not match the selected tool")
        return self


class HypothesisProposal(BaseModel):
    """A non-authoritative policy proposal with no threshold, weight, or decision field."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    hypothesis_id: str = Field(pattern=r"^h-[a-z0-9-]{1,40}$")
    candidate_id: str | None = Field(default=None, pattern=r"^[A-Za-z][A-Za-z0-9_]{1,63}$")
    mechanism: str = Field(min_length=1, max_length=500)
    causal_entity_ref: str = Field(pattern=r"^ent-[a-z0-9-]{3,40}$")
    rationale: str = Field(min_length=1, max_length=2000)
    observation_refs: tuple[str, ...] = Field(default=(), max_length=100)
    missing_evidence: tuple[str, ...] = Field(default=(), max_length=100)


class SubmitHypothesisAction(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    action_type: Literal[ActionType.SUBMIT_HYPOTHESIS] = ActionType.SUBMIT_HYPOTHESIS
    proposal: HypothesisProposal


class UpdateCandidateEvidenceAction(BaseModel):
    """Attach issued episode facts to one already-registered hypothesis."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    action_type: Literal[ActionType.UPDATE_CANDIDATE_EVIDENCE] = (
        ActionType.UPDATE_CANDIDATE_EVIDENCE
    )
    hypothesis_id: str = Field(pattern=r"^h-[a-z0-9-]{1,40}$")
    observation_refs: tuple[str, ...] = Field(default=(), min_length=1, max_length=100)
    missing_evidence: tuple[str, ...] = Field(default=(), max_length=100)


class StopReason(StrEnum):
    POLICY_REQUEST = "policy_request"
    SCRIPT_COMPLETED = "script_completed"
    MAX_ROUNDS = "max_rounds"
    BUDGET_EXHAUSTED = "budget_exhausted"
    POLICY_INVALID_ACTION = "policy_invalid_action"
    POLICY_CRASH = "policy_crash"
    POLICY_TIMEOUT = "policy_timeout"
    TOOL_FAILURE = "tool_failure"


class RequestStopAction(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    action_type: Literal[ActionType.REQUEST_STOP] = ActionType.REQUEST_STOP
    reason: Literal[StopReason.POLICY_REQUEST] = StopReason.POLICY_REQUEST
    summary: str = Field(min_length=1, max_length=1000)


PolicyAction: TypeAlias = Annotated[
    CallToolAction
    | SubmitHypothesisAction
    | UpdateCandidateEvidenceAction
    | RequestStopAction,
    Field(discriminator="action_type"),
]

POLICY_ACTION_ADAPTER = TypeAdapter(PolicyAction)


def parse_policy_action(value: Any) -> PolicyAction:
    return POLICY_ACTION_ADAPTER.validate_python(value)
