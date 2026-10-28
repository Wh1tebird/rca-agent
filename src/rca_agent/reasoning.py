"""Model-facing reasoning adapter facade.

This contract is intentionally proposal-only.  It carries no Runner, Workspace,
Environment, Transcript, Observation ledger or decision authority.
"""

from __future__ import annotations

from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

ReasoningRole = Literal["inquiry", "discovery", "formalizer", "planner"]
ProposalKind = Literal[
    "candidate_proposal",
    "query_proposal",
    "formalized_hypothesis",
    "verification_plan",
    "next_action",
    "stop_suggestion",
    "handoff_suggestion",
]
_AUTHORITY_KEYS = frozenset({
    "answer", "budget", "contradict", "decision", "observation", "permission",
    "root_cause", "stop_authority", "support", "winner", "winning_hypothesis_id",
})


class ReasoningRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    role: ReasoningRole
    operation: str = Field(min_length=1, max_length=80)
    context: dict[str, Any]
    protocol_version: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def context_is_json(self) -> ReasoningRequest:
        import json

        try:
            json.dumps(self.context, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("ReasoningRequest context must be JSON serializable") from exc
        return self


class ReasoningFailure(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: str = Field(min_length=1, max_length=120)
    phase: str = Field(min_length=1, max_length=80)
    retryable: bool = False
    detail_fields: tuple[str, ...] = Field(default=(), max_length=8)


class ReasoningResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    role: ReasoningRole
    operation: str = Field(min_length=1, max_length=80)
    kind: ProposalKind
    proposal: dict[str, Any]
    failure: ReasoningFailure | None = None
    protocol_version: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def proposal_has_no_authority(self) -> ReasoningResponse:
        forbidden = _AUTHORITY_KEYS & set(self.proposal)
        if forbidden:
            raise ValueError(f"reasoning proposal contains authority fields: {sorted(forbidden)}")
        if self.failure is not None and self.proposal:
            raise ValueError("failed reasoning response cannot carry a proposal")
        return self


class ReasoningAdapter(Protocol):
    """Replaceable model-facing proposal interface without Runtime authority."""

    def invoke(self, request: ReasoningRequest) -> ReasoningResponse: ...


class LegacyReasoningAdapter:
    """Compatibility facade over one existing typed adapter.

    It does not grant permissions or commit any runtime state; the wrapped
    adapter remains a parent-side compatibility object. Worker-backed adapters
    can be wrapped here without changing their child-side boundary.
    """

    def __init__(self, role: ReasoningRole, adapter: Any, *, operation_map: dict[str, str]) -> None:
        self.role = role
        self.adapter = adapter
        self.operation_map = dict(operation_map)

    def invoke(self, request: ReasoningRequest) -> ReasoningResponse:
        if request.role != self.role:
            raise ValueError("ReasoningRequest role does not match adapter role")
        method_name = self.operation_map.get(request.operation)
        if method_name is None or not callable(getattr(self.adapter, method_name, None)):
            return ReasoningResponse(
                role=self.role,
                operation=request.operation,
                kind=_kind_for_role(self.role),
                proposal={},
                failure=ReasoningFailure(
                    kind="unsupported_operation", phase=self.role, retryable=False,
                ),
            )
        try:
            result = getattr(self.adapter, method_name)(request.context)
            if isinstance(result, BaseModel):
                proposal = result.model_dump(mode="json")
            elif isinstance(result, dict):
                proposal = result
            else:
                raise TypeError("reasoning adapter returned a non-object")
            return ReasoningResponse(
                role=self.role,
                operation=request.operation,
                kind=_kind_for_role(self.role),
                proposal=proposal,
            )
        except Exception as exc:  # noqa: BLE001 - failure crosses a typed boundary
            return ReasoningResponse(
                role=self.role,
                operation=request.operation,
                kind=_kind_for_role(self.role),
                proposal={},
                failure=ReasoningFailure(
                    kind=type(exc).__name__, phase=self.role, retryable=True,
                ),
            )


def _kind_for_role(role: ReasoningRole) -> ProposalKind:
    return {
        "inquiry": "next_action",
        "discovery": "candidate_proposal",
        "formalizer": "formalized_hypothesis",
        "planner": "verification_plan",
    }[role]
