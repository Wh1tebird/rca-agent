"""Validation shared by live planning and Recorded Replay."""
from __future__ import annotations

from rca_agent.core.dynamic import VerificationPlanDraft


def added_predictions(
    previous: VerificationPlanDraft, revised: VerificationPlanDraft,
) -> tuple[str, ...]:
    """Require a bounded additive revision of exactly the same frozen mechanism."""
    if previous.mechanism_id != revised.mechanism_id:
        raise ValueError("plan revision changed frozen mechanism")
    original = {item.model_dump_json() for item in previous.predictions}
    proposed = {item.model_dump_json() for item in revised.predictions}
    if not original <= proposed:
        raise ValueError("plan revision removed or changed a frozen prediction")
    new = tuple(item for item in revised.predictions if item.model_dump_json() not in original)
    if not 1 <= len(new) <= 2:
        raise ValueError("plan revision requires one or two new predictions")
    return tuple(item.capability_ref for item in new)
