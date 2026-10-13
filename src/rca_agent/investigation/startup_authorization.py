"""Rebuild the host's bounded startup policy from issued requests and responses."""
from typing import Any

from rca_agent.models import CallToolAction


def selected_startup_action(state: Any, bootstrap: Any,
                            events: tuple[Any, ...]) -> CallToolAction | None:
    if (bootstrap is None or not state.verify_digest() or state.phase != "planning"
            or state.planning_steps or state.batches):
        return None
    # Use the actual acquisition policy, with model selection disabled. This replay
    # executes no tools and adds no observations or candidate evidence.
    from rca_agent.inquiry_policy import InquiryInvestigationPolicy

    class StartupOnlyPolicy(InquiryInvestigationPolicy):
        def _inquiry_screen(self, context):
            return None

    policy = StartupOnlyPolicy(None)
    history: list[dict[str, Any]] = []
    context = {"bootstrap": bootstrap.model_dump(mode="json"), "history": history}
    pending = None
    for event in events:
        kind = str(getattr(event.event_type, "value", event.event_type))
        if kind in {"stop_requested", "discovery_started", "answer_frozen"}:
            return None
        if kind == "tool_request":
            if event.payload.get("execution_source") not in {None, "startup"}:
                return None
            expected = policy.invoke({"context": context}).action
            actual = CallToolAction.model_validate({"tool_name": event.payload["tool_name"],
                                                    "request": event.payload["request"]})
            if expected != actual or pending is not None:
                raise ValueError("startup request differs from deterministic host selection")
            pending = actual
        elif kind == "tool_response" and pending is not None:
            history.append({"action": pending.model_dump(mode="json"),
                            "result": {"kind": "tool_response", "value": event.payload}})
            pending = None
    if pending is not None:
        return None
    action = policy.invoke({"context": context}).action
    return action if isinstance(action, CallToolAction) else None


def startup_grant(events: tuple[Any, ...]) -> dict[str, Any] | None:
    """A selection survives only its own matching PolicyAction, before consumption."""
    grant = None
    for event in events:
        kind = str(getattr(event.event_type, "value", event.event_type))
        if kind == "startup_action_selected":
            grant = {**event.payload, "selection_event_digest": event.digest}
        elif grant is not None and not (kind == "policy_action"
                                       and event.payload == grant["action"]):
            grant = None
    return grant
