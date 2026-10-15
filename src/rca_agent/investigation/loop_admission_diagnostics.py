"""Independent, side-effect-free checks shared by batched admission and retry feedback."""
from __future__ import annotations

from typing import Any

from rca_agent.investigation.inquiry_loop import (
    LoopIdentityError,
    LoopState,
    LoopStep,
    PendingDecisionError,
    actionable_pending_ids,
)
from rca_agent.models import CallToolAction
from rca_agent.tools.fingerprint import query_fingerprint


class LoopContractIssues(ValueError):
    """Several independent, fully parsed step fields violate host admission."""

    code = "LOOP_CONTRACT_ISSUES"

    def __init__(self, errors: list[ValueError]) -> None:
        self.issues = [issue_payload(error) for error in errors]
        self.fields = tuple(dict.fromkeys(field for issue in self.issues
                                          for field in issue["fields"]))
        super().__init__("multiple independent Inquiry admission errors")


def issue_payload(error: ValueError) -> dict[str, Any]:
    return {
        "code": getattr(error, "code", type(error).__name__),
        "fields": list(getattr(error, "fields", ("pending_decisions",)
                       if isinstance(error, PendingDecisionError) else getattr(error, "fields", ()))),
        "conflict_id": getattr(error, "conflict_id", None),
        "conflict_status": getattr(error, "conflict_status", None),
        "query_fingerprint": getattr(error, "query_fingerprint", None),
        "instruction": getattr(error, "instruction", None),
        "expected": list(getattr(error, "expected", ())),
        "missing": list(getattr(error, "missing", ())),
        "unexpected": list(getattr(error, "unexpected", ())),
        "duplicate": list(getattr(error, "duplicate", ())),
        "invalid_states": dict(getattr(error, "invalid_states", {})),
    }


def independent_issues(state: LoopState, step: LoopStep,
                       menu: dict[str, dict[str, Any]]) -> list[ValueError]:
    """Check independent identities only after LoopStep parsing and state binding.

    A malformed pending set has no reliable post-decision ledger, so physical-query
    conflict checks are skipped in that case rather than guessed.
    """
    issues: list[ValueError] = []
    eligible = actionable_pending_ids(state)
    submitted = tuple(item.pending_id for item in step.pending_decisions)
    duplicate = {item for item in submitted if submitted.count(item) > 1}
    known = {item.pending_id: item for item in state.pending}
    missing = eligible - set(submitted)
    unexpected = set(submitted) - eligible
    if missing or unexpected or duplicate:
        invalid_states = {item: known[item].status for item in unexpected if item in known}
        issues.append(PendingDecisionError(
            expected=eligible, submitted=submitted, missing=missing,
            unexpected=unexpected, duplicate=duplicate, invalid_states=invalid_states,
        ))

    draft = step.draft
    if draft is None:
        return issues
    candidates = {item.hypothesis["hypothesis_id"]: item for item in state.candidates}
    needs = {item.need["need_id"]: item for item in state.needs}
    for item in draft.hypotheses:
        old = candidates.get(item.hypothesis_id)
        if old is not None and old.hypothesis != item.model_dump(mode="json"):
            issues.append(LoopIdentityError(
                code="CANDIDATE_IDENTITY_CHANGED", field="draft.hypotheses[].hypothesis_id",
                identifier=item.hypothesis_id,
                instruction="Keep the old candidate unchanged or create a new hypothesis_id and supersede it.",
            ))
    for item in draft.needs:
        old = needs.get(item.need_id)
        if old is not None and old.need != item.model_dump(mode="json"):
            issues.append(LoopIdentityError(
                code="NEED_IDENTITY_CHANGED", field="draft.needs[].need_id",
                identifier=item.need_id,
                instruction=("Keep the old Need unchanged or create a new need_id and set "
                             f"need_updates[].supersedes_need_id={item.need_id}."),
            ))

    if missing or unexpected or duplicate:
        return issues
    cancelled = {item.pending_id for item in step.pending_decisions
                 if item.disposition == "cancel"}
    active = [item for item in state.pending
              if item.status != "cancelled" and item.pending_id not in cancelled]
    seen_fingerprints: set[str] = set()
    for choice in draft.choices:
        option = menu.get(choice.query_ref)
        if option is None:
            continue  # normalize_loop_step owns menu membership; do not infer an Action.
        action = CallToolAction.model_validate(option["action"])
        fingerprint = query_fingerprint(action.tool_name, action.request)
        if fingerprint in seen_fingerprints:
            continue  # shared choices for one physical query are legal.
        seen_fingerprints.add(fingerprint)
        conflict = next((item for item in active if item.query_fingerprint == fingerprint), None)
        if conflict is None:
            continue
        if conflict.status in {"pending", "deferred", "unavailable"}:
            instruction = (f"Dispose existing {conflict.pending_id} in pending_decisions; "
                           "select its pending_id only if retained and available. Do not recreate it.")
        else:
            instruction = "This scheduled or executed query is read-only; choose another query or hand off."
        issues.append(LoopIdentityError(
            code="DUPLICATE_PHYSICAL_QUERY", field="draft.choices[].query_ref",
            identifier=conflict.pending_id, status=conflict.status,
            fingerprint=fingerprint, instruction=instruction,
        ))
    return issues


def admit_independent_issues(state: LoopState, step: LoopStep,
                             menu: dict[str, dict[str, Any]]) -> None:
    errors = independent_issues(state, step, menu)
    if len(errors) == 1:
        raise errors[0]
    if errors:
        raise LoopContractIssues(errors)
