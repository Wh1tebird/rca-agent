from __future__ import annotations

import pytest
from pydantic import ValidationError

from rca_agent.reasoning import (
    LegacyReasoningAdapter,
    ReasoningFailure,
    ReasoningRequest,
    ReasoningResponse,
)
from rca_agent.runner.component_adapter import (
    ComponentAdapterSpec,
    WorkerBackedPlannerAdapter,
)


def planner() -> WorkerBackedPlannerAdapter:
    return WorkerBackedPlannerAdapter(ComponentAdapterSpec(
        component="planner",
        factory_id="test.fake.planner",
        config={"response": {"plans": []}},
    ))


def test_legacy_planner_and_reasoning_facade_have_same_typed_result() -> None:
    direct = planner().plan({"frozen": []}).model_dump(mode="json")
    facade = LegacyReasoningAdapter("planner", planner(), operation_map={"plan": "plan"})
    response = facade.invoke(ReasoningRequest(
        role="planner", operation="plan", context={"frozen": []},
    ))
    assert response.kind == "verification_plan"
    assert response.proposal == direct
    assert response.failure is None


def test_reasoning_contract_rejects_authority_fields_and_non_json_context() -> None:
    with pytest.raises(ValidationError, match="authority fields"):
        ReasoningResponse(
            role="discovery",
            operation="discover",
            kind="candidate_proposal",
            proposal={"winner": "candidate"},
        )
    with pytest.raises(ValidationError, match="JSON serializable"):
        ReasoningRequest(role="planner", operation="plan", context={"bad": object()})


def test_reasoning_adapter_failure_is_typed_and_has_no_runtime_authority() -> None:
    adapter = LegacyReasoningAdapter("planner", planner(), operation_map={"missing": "missing"})
    response = adapter.invoke(ReasoningRequest(
        role="planner", operation="missing", context={},
    ))
    assert response.proposal == {}
    assert response.failure == ReasoningFailure(
        kind="unsupported_operation", phase="planner", retryable=False,
    )
    assert "answer" not in response.model_dump(mode="json")


def test_worker_backed_reasoning_timeout_stays_a_failure() -> None:
    adapter = LegacyReasoningAdapter(
        "planner",
        WorkerBackedPlannerAdapter(ComponentAdapterSpec(
            component="planner",
            factory_id="test.fake.planner",
            config={"sleep_ms": 250},
        ), timeout_seconds=0.03),
        operation_map={"plan": "plan"},
    )
    response = adapter.invoke(ReasoningRequest(role="planner", operation="plan", context={}))
    assert response.failure is not None
    assert response.failure.kind == "TimeoutError"
    assert response.proposal == {}
