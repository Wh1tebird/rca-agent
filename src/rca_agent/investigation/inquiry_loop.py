"""Host-owned, non-authoritative state machine for bounded Inquiry batches.

The same transition function is used before execution and by Recorded Replay. Its
model opinions never enter Core evidence admission or discrimination task status.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rca_agent.investigation.inquiry_contract import (
    action_fingerprint,
    original_alternative_id,
    selected_bindings,
    validate_bindings,
    validate_choice,
)
from rca_agent.langchain_acquisition import (
    InquiryDraftV2,
    InquiryHypothesis,
    InquiryNeed,
    parse_inquiry_draft,
)
from rca_agent.models import CallToolAction, Observation, ToolResponse
from rca_agent.tools.fingerprint import query_fingerprint, sha256_digest

LOOP_VERSION = 1
HOST_LOOP_VERSION_LEGACY = 3
HOST_LOOP_VERSION = 4
VIEW_LIMIT = 128 * 1024


class StrictInquiryHypothesis(InquiryHypothesis):
    model_config = ConfigDict(extra="forbid", frozen=True)
    alternative_id: str | None = Field(
        default=None,
        description=(
            "Host-derived identity for the original alternative. Leave null; do not invent "
            "an alt-* ID. The host derives it from the alternative text."
        ),
    )


class StrictInquiryNeed(InquiryNeed):
    model_config = ConfigDict(extra="forbid", frozen=True)
    need_id: str = Field(
        min_length=1, max_length=40,
        description=("Immutable Need identity. Reuse an existing ID only with identical Need content. "
                     "For changed content create a new need_id and set "
                     "need_updates[].supersedes_need_id to the old ID."),
    )


class LoopDraftV2(InquiryDraftV2):
    model_config = ConfigDict(extra="forbid", frozen=True)
    hypotheses: tuple[StrictInquiryHypothesis, ...] = Field(min_length=1, max_length=4)
    needs: tuple[StrictInquiryNeed, ...] = Field(min_length=1, max_length=12)


class LoopAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    binding_id: str
    query_ref: str = Field(
        min_length=1,
        description=(
            "Result fingerprint only. Copy query_fingerprint from one item in "
            "last_batch.results (normally qry-...). Do not use a menu key "
            "such as aq-... or q1."
        ),
    )
    observation_refs: tuple[str, ...] = ()
    status: Literal["answered", "partial", "ambiguous", "unavailable"]
    candidate_effect: Literal["more_relevant", "less_relevant", "unchanged", "unclear"]
    alternative_effect: Literal["more_relevant", "less_relevant", "unchanged", "unclear"]
    remaining_gap: str = Field(max_length=1000)
    reason: str = Field(min_length=1, max_length=1000)


class LoopCandidateUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    hypothesis_id: str
    priority: int = Field(ge=0, le=3)
    disposition: Literal["active", "deferred"]
    reason: str = Field(min_length=1, max_length=1000)
    supersedes_hypothesis_id: str | None = None


class LoopNeedUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    need_id: str
    disposition: Literal["open", "deferred", "closed"]
    reason: str = Field(min_length=1, max_length=1000)
    supersedes_need_id: str | None = Field(
        default=None,
        description="Old Need ID replaced by this update's NEW need_id; never edit the old Need in place.",
    )


class LoopPendingDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    pending_id: str
    disposition: Literal["retain", "defer", "cancel"]
    reason: str = Field(min_length=1, max_length=1000)
    basis_observation_refs: tuple[str, ...] = Field(default=(), max_length=100)


class LoopStep(BaseModel):
    """Strict model step. `draft` uses the published Inquiry-v2 schema when present."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    contract_version: int | None = None
    based_on_revision: int = Field(ge=0)
    based_on_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    batch_assessments: tuple[LoopAssessment, ...] = Field(default=(), max_length=16)
    candidate_updates: tuple[LoopCandidateUpdate, ...] = Field(default=(), max_length=20)
    need_updates: tuple[LoopNeedUpdate, ...] = Field(default=(), max_length=40)
    pending_decisions: tuple[LoopPendingDecision, ...] = Field(default=(), max_length=8)
    draft: LoopDraftV2 | None = None
    next_batch: tuple[str, ...] = Field(
        default=(), max_length=2,
        description="Non-empty when decision=continue; use handoff with stop_reason when no query remains.",
    )
    batch_reason: str | None = Field(default=None, max_length=1000)
    decision: Literal["continue", "handoff"]
    stop_reason: str | None = Field(default=None, max_length=1000)

    @model_validator(mode="after")
    def check_decision(self):
        if self.contract_version not in {None, LOOP_VERSION} or type(self.contract_version) not in {int, type(None)}:
            raise ValueError("unsupported Inquiry loop version")
        if self.decision == "continue":
            if not self.next_batch or not self.batch_reason or self.stop_reason:
                raise ValueError("continuing Inquiry requires a justified nonempty batch")
        elif self.next_batch or self.draft is not None or not self.stop_reason:
            raise ValueError("handoff requires a reason and no new query proposal")
        return self


class LoopAttempt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    step: int = Field(ge=1, le=6)
    attempt: int = Field(ge=1, le=2)
    status: Literal["parsed", "failed"]
    code: str = Field(min_length=1, max_length=100, pattern=r"^[A-Za-z][A-Za-z_]*$")
    input_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    raw_digest: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    fields: tuple[str, ...] = Field(default=(), max_length=20)
    raw_shape: dict[str, str] = Field(default_factory=dict, max_length=20)
    repair_feedback: dict[str, Any] = Field(default_factory=dict, max_length=20)
    pending_diagnostics: dict[str, Any] = Field(default_factory=dict, max_length=20)
    stage: Literal["adapter", "normalize", "accept", "unknown"] | None = None
    error_type: str | None = Field(default=None, max_length=100)
    state_revision: int | None = Field(default=None, ge=0)
    state_digest: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    elapsed_ms: int | None = Field(default=None, ge=0)
    usage: dict[str, int] | None = None


class PendingDecisionError(ValueError):
    """Structured, non-authoritative diagnostics for pending-set admission."""

    def __init__(self, *, expected: set[str], submitted: tuple[str, ...],
                 missing: set[str], unexpected: set[str], duplicate: set[str],
                 invalid_states: dict[str, str] | None = None) -> None:
        self.code = "PENDING_DECISION_SET_MISMATCH"
        self.expected = tuple(sorted(expected))
        self.submitted = submitted
        self.missing = tuple(sorted(missing))
        self.unexpected = tuple(sorted(unexpected))
        self.duplicate = tuple(sorted(duplicate))
        self.invalid_states = dict(sorted((invalid_states or {}).items()))
        details = {
            "missing": self.missing,
            "unexpected": self.unexpected,
            "duplicate": self.duplicate,
            "invalid_states": self.invalid_states,
        }
        super().__init__(f"pending decision set mismatch: {details}")


class AssessmentReferenceError(ValueError):
    """The model used a menu reference where a completed-result fingerprint is required."""

    def __init__(self, *, supplied: str, available: tuple[str, ...]) -> None:
        self.code = "ASSESSMENT_QUERY_REF_INVALID"
        self.fields = ("batch_assessments[].query_ref",)
        self.supplied = supplied
        self.expected_format = "qry-<result fingerprint from last_batch.results>"
        self.available_result_ids = tuple(available)
        super().__init__(
            "assessment query_ref must copy a query_fingerprint from last_batch.results; "
            f"got {supplied!r}, available={self.available_result_ids}"
        )


class AssessmentEvidenceError(ValueError):
    """The assessment cites evidence that cannot support its declared status."""

    def __init__(self, *, kind: str, fields: tuple[str, ...], details: str) -> None:
        self.code = "ASSESSMENT_EVIDENCE_INVALID"
        self.fields = fields
        self.evidence_error = kind
        super().__init__(details)


class LoopNormalizationError(ValueError):
    """Safe, field-level feedback for host normalization failures."""

    def __init__(self, *, code: str, field: str, supplied: Any = None,
                 expected_format: str | None = None, details: str = "") -> None:
        self.code = code
        self.fields = (field,)
        self.supplied = supplied
        self.expected_format = expected_format
        super().__init__(details or code)


class LoopIdentityError(ValueError):
    """Field-level, bounded diagnostics for immutable Need and query identities."""

    def __init__(self, *, code: str, field: str, identifier: str,
                 status: str | None = None, fingerprint: str | None = None,
                 instruction: str) -> None:
        self.code = code
        self.fields = (field,)
        self.conflict_id = identifier
        self.conflict_status = status
        self.query_fingerprint = fingerprint
        self.instruction = instruction
        self.safe_details = f"{field} conflicts with existing {identifier}"
        super().__init__(self.safe_details)


class LoopPending(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    pending_id: str
    query_fingerprint: str
    query_ref: str
    action: dict[str, Any]
    bindings: tuple[dict[str, Any], ...]
    hypotheses: tuple[dict[str, Any], ...]
    needs: tuple[dict[str, Any], ...]
    choices: tuple[dict[str, Any], ...]
    created_step: int
    status: Literal["pending", "deferred", "cancelled", "scheduled", "executed", "unavailable"]
    reason: str | None = None


class LoopBatchResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    pending_id: str
    query_fingerprint: str
    status: str
    observation_refs: tuple[str, ...]
    observations: tuple[dict[str, Any], ...]
    truncated: bool
    response_sequence: int


class LoopBatch(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    batch_id: str
    step: int
    pending_ids: tuple[str, ...]
    original_order: tuple[str, ...]
    actual_order: tuple[str, ...] = ()
    results: tuple[LoopBatchResult, ...] = ()
    assessed: bool = False
    unassessed_reason: str | None = None


class LoopCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    hypothesis: dict[str, Any]
    priority: int = Field(ge=0, le=3)
    disposition: Literal["active", "deferred", "superseded"]
    source_step: int
    supersedes_hypothesis_id: str | None = None
    reason: str | None = None


class LoopNeed(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    need: dict[str, Any]
    disposition: Literal["open", "deferred", "closed", "superseded"]
    source_step: int
    supersedes_need_id: str | None = None
    reason: str | None = None


class LoopState(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    contract_version: Literal[1] = LOOP_VERSION
    episode_id: str
    revision: int = 0
    planning_steps: int = 0
    review_steps: int = 0
    phase: Literal["planning", "awaiting_results", "awaiting_assessment", "handoff"] = "planning"
    candidates: tuple[LoopCandidate, ...] = ()
    needs: tuple[LoopNeed, ...] = ()
    pending: tuple[LoopPending, ...] = ()
    batches: tuple[LoopBatch, ...] = ()
    assessments: tuple[LoopAssessment, ...] = ()
    digest: str

    @classmethod
    def initial(cls, episode_id: str) -> LoopState:
        fields = {"episode_id": episode_id}
        return cls(**fields, digest=_digest({"contract_version": LOOP_VERSION, **fields,
             "revision": 0, "planning_steps": 0, "review_steps": 0, "phase": "planning",
             "candidates": [], "needs": [], "pending": [], "batches": [], "assessments": []}))

    def verify_digest(self) -> bool:
        return _digest(self.model_dump(mode="json", exclude={"digest"})) == self.digest


def _digest(value: Any) -> str:
    return sha256_digest(value)


def _state(previous: LoopState, **changes: Any) -> LoopState:
    fields = previous.model_dump(mode="json", exclude={"digest"})
    fields.update(changes)
    return LoopState(**fields, digest=_digest(fields))


def admitted_batch_member(state: LoopState, action: CallToolAction) -> LoopPending:
    """Resolve one unspent physical query from the current accepted batch."""
    if not state.verify_digest() or state.phase != "awaiting_results" or not state.batches:
        raise ValueError("no accepted Inquiry batch is awaiting this query")
    batch = state.batches[-1]
    if batch.assessed or batch.unassessed_reason or len(batch.results) >= len(batch.pending_ids):
        raise ValueError("Inquiry batch execution has ended")
    raw = action.model_dump(mode="json")
    fingerprint = query_fingerprint(action.tool_name, action.request)
    members = [p for p in state.pending if p.pending_id in batch.pending_ids
               and p.status == "scheduled" and p.action == raw
               and p.query_fingerprint == fingerprint]
    if len(members) != 1 or any(r.pending_id == members[0].pending_id for r in batch.results):
        raise ValueError("query has no unspent grant in the accepted Inquiry batch")
    return members[0]


def actionable_pending_ids(state: LoopState) -> set[str]:
    """Return the host-owned set that requires a disposition this step."""
    return {item.pending_id for item in state.pending
            if item.status in {"pending", "deferred", "unavailable"}}


def pending_public_projection(state: LoopState) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split the complete ledger into actionable entries and read-only history."""
    actionable: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    actionable_ids = actionable_pending_ids(state)
    for item in state.pending:
        value = item.model_dump(mode="json", exclude={"hypotheses", "needs", "choices"})
        if item.pending_id in actionable_ids:
            actionable.append(value)
        else:
            history.append(value)
    return actionable, history


def loop_attempt_input(view: dict[str, Any], *, repair_feedback: dict[str, Any] | None = None) -> dict[str, Any]:
    """Canonical public input envelope used for attempt binding and Replay."""
    return {
        "view": view,
        "schema": provider_loop_schema(),
        "repair_feedback": repair_feedback or {},
    }


def loop_attempt_digest(view: dict[str, Any], *, repair_feedback: dict[str, Any] | None = None) -> str:
    """Bind v4 attempts to full provider input while preserving v3 history."""
    if view.get("view", {}).get("inquiry_loop_version", 3) < HOST_LOOP_VERSION:
        return sha256_digest(view)
    return sha256_digest(loop_attempt_input(view, repair_feedback=repair_feedback))


def provider_loop_schema() -> dict[str, Any]:
    """Provider-facing schema with explicit outer-loop and nested-draft versions."""
    schema = deepcopy(LoopStep.model_json_schema())
    properties = schema.setdefault("properties", {})
    properties["contract_version"] = {"const": LOOP_VERSION}
    assessment = schema.get("$defs", {}).get("LoopAssessment")
    if isinstance(assessment, dict):
        assessment.setdefault("properties", {})["query_ref"] = {
            "type": "string",
            "description": (
                "Result fingerprint copied from last_batch.results[].query_fingerprint "
                "(normally qry-...). Never use a menu query ref (aq-...) here."
            ),
        }
    choice = schema.get("$defs", {}).get("InquiryChoiceV2")
    if isinstance(choice, dict):
        choice.setdefault("properties", {})["query_ref"] = {
            "type": "string",
            "description": (
                "Menu query ref copied from the current menu (aq-... or its qN alias). "
                "This is different from an assessment result fingerprint."
            ),
        }
    schema["required"] = [key for key in schema.get("required", ())
                           if key != "contract_version"] + ["contract_version"]
    nested = schema.get("$defs", {}).get("LoopDraftV2")
    if isinstance(nested, dict):
        nested.setdefault("properties", {})["contract_version"] = {"const": 2}
        nested["required"] = [key for key in nested.get("required", ())
                               if key != "contract_version"] + ["contract_version"]
    return schema


def normalize_loop_step(raw: dict[str, Any], menu: dict[str, dict[str, Any]], *,
                        menu_aliases: dict[str, Any] | None = None) -> LoopStep:
    """Map only aliases from this exact menu and run the real Inquiry-v2 parser."""
    data = dict(raw)
    if data.get("contract_version") is None:
        data["contract_version"] = LOOP_VERSION
    if data.get("draft") is not None:
        source = dict(data["draft"])
        LoopDraftV2.model_validate(source)
        if menu_aliases is None:
            aliases = {f"q{index}": ref for index, ref in enumerate(menu, start=1)}
        else:
            aliases = {}
            for alias, value in menu_aliases.items():
                ref = value.get("menu_ref") if isinstance(value, dict) else value
                if ref not in menu:
                    raise ValueError("menu alias does not resolve to the current host menu")
                if isinstance(value, dict) and value.get("action_fingerprint"):
                    action = CallToolAction.model_validate(menu[ref]["action"])
                    if value["action_fingerprint"] != action_fingerprint(action):
                        raise ValueError("menu alias fingerprint changed")
                aliases[alias] = ref
        source["choices"] = [
            {**choice, "query_ref": aliases.get(choice.get("query_ref"), choice.get("query_ref"))}
            for choice in source.get("choices", ())
        ]
        parsed = parse_inquiry_draft(source, live_version=2, strict=True)
        hypotheses = []
        for hypothesis in parsed.hypotheses:
            expected = original_alternative_id(hypothesis.model_dump(mode="json"))
            if hypothesis.alternative_id not in {None, expected}:
                raise LoopNormalizationError(
                    code="ORIGINAL_ALTERNATIVE_ID_MISMATCH",
                    field="draft.hypotheses[].alternative_id",
                    supplied=hypothesis.alternative_id,
                    expected_format="null (host derives the ID from alternative text)",
                    details="alternative_id must be null; the host derives original alternative identity",
                )
            hypotheses.append(hypothesis.model_copy(update={"alternative_id": expected}))
        parsed = parsed.model_copy(update={"hypotheses": tuple(hypotheses)})
        needs = {need.need_id: need for need in parsed.needs}
        for choice in parsed.choices:
            option = menu.get(choice.query_ref)
            if option is None:
                raise ValueError("new query is not in the current host menu")
            validate_choice(needs[choice.need_id], choice,
                            CallToolAction.model_validate(option["action"]))
        data["draft"] = LoopDraftV2.model_validate(parsed.model_dump(mode="json"))
        data["next_batch"] = tuple(
            aliases.get(item, item) for item in data.get("next_batch", ())
        )
    return LoopStep.model_validate(data)


def accept_loop_step(
    state: LoopState, step: LoopStep, *, menu: dict[str, dict[str, Any]],
    batch_limit: int, query_limit: int, visible_refs: set[str] | None = None,
    budget_remaining: int | None = None, offered_refs: set[str] | None = None,
) -> tuple[LoopState, dict[str, Any] | None]:
    """Validate the entire decision before returning an immutable new state and round."""
    if not state.verify_digest() or step.based_on_revision != state.revision or step.based_on_digest != state.digest:
        raise ValueError("stale Inquiry loop state")
    if state.phase == "handoff" or not 1 <= batch_limit <= 2 or not 1 <= query_limit <= 10:
        raise ValueError("Inquiry loop is closed or host limits are invalid")
    typed_menu = {ref: {**item, "action": CallToolAction.model_validate(item["action"])}
                  for ref, item in menu.items()}
    if state.phase == "awaiting_results":
        raise ValueError("batch has not completed")
    last = state.batches[-1] if state.batches else None
    if last is not None and not last.assessed:
        if len(last.results) != len(last.pending_ids):
            raise ValueError("last batch is incomplete")
        bindings = {binding["binding_id"]: (pending, result)
                    for pending in state.pending for result in last.results
                    if pending.pending_id == result.pending_id for binding in pending.bindings}
        if len(step.batch_assessments) != len(bindings) or {
            item.binding_id for item in step.batch_assessments
        } != set(bindings):
            raise ValueError("assessment must cover every last-batch binding")
        available_result_ids = tuple(result.query_fingerprint for result in last.results)
        for item in step.batch_assessments:
            pending, result = bindings[item.binding_id]
            if item.query_ref != result.query_fingerprint:
                raise AssessmentReferenceError(
                    supplied=item.query_ref, available=available_result_ids
                )
            if not set(item.observation_refs) <= set(result.observation_refs):
                raise AssessmentEvidenceError(
                    kind="extra_observation",
                    fields=("batch_assessments[].observation_refs",),
                    details=(
                        "assessment observation_refs must be a subset of the result's "
                        f"observation_refs; available={result.observation_refs}"
                    ),
                )
            if visible_refs is None or not set(item.observation_refs) <= visible_refs:
                raise ValueError("assessment cites a result outside the recorded view")
            binding = next(b for b in pending.bindings if b["binding_id"] == item.binding_id)
            if item.status == "answered" and (binding["mode"] != "direct" or result.truncated
                                               or result.status != "success" or not item.observation_refs):
                    raise AssessmentEvidenceError(
                        kind="incomplete_or_exploratory",
                        fields=("batch_assessments[].status",),
                        details="answered requires a complete direct successful result",
                    )
            if item.status == "answered":
                observations = {o["observation_id"]: Observation.model_validate(o)
                                for o in result.observations}
                target = binding["measurement_target"]
                accepted_signals = {target["signal"]}
                if target["measurement_kind"] == "window_comparison":
                    accepted_signals.update({
                        f'{target["signal"]}_window_comparison',
                        f'{target["signal"]}_baseline_ratio',
                    })
                if not any(
                    observations[ref].data_status.value in {"observed", "derived"}
                    and observations[ref].value is not None
                    and observations[ref].signal in accepted_signals
                    and observations[ref].entity_ref == target["entity_ref"]
                    for ref in item.observation_refs
                ):
                    raise AssessmentEvidenceError(
                        kind="target_mismatch",
                        fields=("batch_assessments[].status", "batch_assessments[].observation_refs"),
                        details=(
                            "Need lacks a complete directly matched value; answered requires "
                            "an observed or derived value matching "
                            f"target signal={target['signal']!r} entity_ref={target['entity_ref']!r}"
                        ),
                    )
    elif step.batch_assessments:
        raise ValueError("no batch is available for assessment")

    from rca_agent.investigation.loop_admission_diagnostics import admit_independent_issues

    admit_independent_issues(state, step, menu)
    if offered_refs is not None and step.draft is not None:
        hidden = {choice.query_ref for choice in step.draft.choices} - offered_refs
        if hidden:
            raise ValueError("query is not in the current projected Inquiry menu")
    pending = {p.pending_id: p for p in state.pending}
    offered_actions = {query_fingerprint(CallToolAction.model_validate(option["action"]).tool_name,
                         CallToolAction.model_validate(option["action"]).request): option["action"]
                       for option in menu.values()}
    issued_refs = {ref for batch in state.batches for result in batch.results
                   for ref in result.observation_refs}
    for decision in step.pending_decisions:
        if not set(decision.basis_observation_refs) <= issued_refs or (
            visible_refs is not None and not set(decision.basis_observation_refs) <= visible_refs
        ):
            raise ValueError("pending decision cites an unissued result")
        previous = pending[decision.pending_id]
        status = {"retain": "pending", "defer": "deferred", "cancel": "cancelled"}[decision.disposition]
        reason = decision.reason
        if (status == "pending" and offered_actions.get(previous.query_fingerprint)
                != previous.action):
            status = "unavailable"
            reason = "Exact accepted query is no longer offered: " + decision.reason
        pending[decision.pending_id] = previous.model_copy(update={"status": status, "reason": reason})

    candidates = {item.hypothesis["hypothesis_id"]: item for item in state.candidates}
    needs = {item.need["need_id"]: item for item in state.needs}
    parsed = step.draft
    new_by_ref: dict[str, list[dict[str, Any]]] = {}
    if parsed is not None:
        hypo_by_id = {item.hypothesis_id: item.model_dump(mode="json") for item in parsed.hypotheses}
        need_by_id = {item.need_id: item.model_dump(mode="json") for item in parsed.needs}
        for item in parsed.hypotheses:
            raw = hypo_by_id[item.hypothesis_id]
            candidates.setdefault(item.hypothesis_id, LoopCandidate(
                hypothesis=raw, priority=1, disposition="active", source_step=state.planning_steps + 1))
        for item in parsed.needs:
            raw = need_by_id[item.need_id]
            needs.setdefault(item.need_id, LoopNeed(
                need=raw, disposition="open", source_step=state.planning_steps + 1))
        for choice in parsed.choices:
            new_by_ref.setdefault(choice.query_ref, []).append(choice.model_dump(mode="json"))
        for ref, choices in new_by_ref.items():
            action = CallToolAction.model_validate(menu[ref]["action"])
            fingerprint = query_fingerprint(action.tool_name, action.request)
            identifier = "pending-" + _digest({"episode": state.episode_id,
                        "step": state.planning_steps + 1, "query": fingerprint})[:20]
            bindings = selected_bindings(parsed, typed_menu, [action], state.planning_steps + 1)
            associated_need_ids = tuple(dict.fromkeys(c["need_id"] for c in choices))
            associated_needs = tuple(need_by_id[item] for item in associated_need_ids)
            associated_hypotheses = tuple(hypo_by_id[item] for item in dict.fromkeys(
                need["hypothesis_id"] for need in associated_needs))
            pending[identifier] = LoopPending(
                pending_id=identifier, query_fingerprint=fingerprint, query_ref=ref,
                action=action.model_dump(mode="json"), bindings=tuple(bindings),
                hypotheses=associated_hypotheses, needs=associated_needs,
                choices=tuple(choices), created_step=state.planning_steps + 1,
                status="pending")

    for update in step.candidate_updates:
        item = candidates.get(update.hypothesis_id)
        if item is None:
            raise ValueError("candidate update references an unknown hypothesis")
        if update.supersedes_hypothesis_id:
            old = candidates.get(update.supersedes_hypothesis_id)
            if (old is None or old.hypothesis["hypothesis_id"] == update.hypothesis_id
                    or item.source_step != state.planning_steps + 1
                    or item.supersedes_hypothesis_id not in {None, update.supersedes_hypothesis_id}):
                raise ValueError("candidate revision source is invalid")
            candidates[update.supersedes_hypothesis_id] = old.model_copy(update={"disposition": "superseded"})
        candidates[update.hypothesis_id] = item.model_copy(update={"priority": update.priority,
            "disposition": update.disposition, "reason": update.reason,
            "supersedes_hypothesis_id": update.supersedes_hypothesis_id})
    for update in step.need_updates:
        item = needs.get(update.need_id)
        if item is None:
            raise ValueError("Need update references an unknown Need")
        if update.supersedes_need_id:
            old = needs.get(update.supersedes_need_id)
            if (old is None or old.need["need_id"] == update.need_id
                    or item.source_step != state.planning_steps + 1
                    or item.supersedes_need_id not in {None, update.supersedes_need_id}):
                raise ValueError("Need revision source is invalid")
            needs[update.supersedes_need_id] = old.model_copy(update={"disposition": "superseded"})
        needs[update.need_id] = item.model_copy(update={"disposition": update.disposition,
            "reason": update.reason, "supersedes_need_id": update.supersedes_need_id})

    if len({p.query_fingerprint for p in pending.values()
            if p.status in {"pending", "deferred", "unavailable"}}) > 8:
        raise ValueError("pending query capacity exceeded")
    round_record = None
    batches = list(state.batches)
    if last is not None and not last.assessed:
        batches[-1] = last.model_copy(update={"assessed": True})
    if step.decision == "continue":
        if state.planning_steps >= 5 or sum(p.status == "executed" for p in state.pending) >= query_limit:
            raise ValueError("Inquiry query planning limit reached")
        selected: list[LoopPending] = []
        for identifier in step.next_batch:
            item = pending.get(identifier)
            if item is None:
                item = next((p for p in pending.values()
                             if p.query_ref == identifier and p.created_step == state.planning_steps + 1), None)
            if item is None or item.status != "pending":
                raise ValueError("batch selects an unavailable pending query")
            selected.append(item)
        fingerprints = [p.query_fingerprint for p in selected]
        if len(selected) > batch_limit or len(fingerprints) != len(set(fingerprints)):
            raise ValueError("batch exceeds physical query limit or repeats a query")
        if sum(p.status == "executed" for p in state.pending) + len(selected) > query_limit:
            raise ValueError("Inquiry query limit reached")
        if budget_remaining is not None:
            costs = [next((option["query_cost"] for option in menu.values()
                if query_fingerprint(CallToolAction.model_validate(option["action"]).tool_name,
                                     CallToolAction.model_validate(option["action"]).request)
                == item.query_fingerprint), None) for item in selected]
            if any(cost is None for cost in costs) or sum(costs) > budget_remaining:
                raise ValueError("insufficient budget for the entire Inquiry batch")
        offered = {query_fingerprint(CallToolAction.model_validate(o["action"]).tool_name,
                 CallToolAction.model_validate(o["action"]).request): o for o in menu.values()}
        if any(
            p.query_fingerprint not in offered
            or CallToolAction.model_validate(offered[p.query_fingerprint]["action"]).model_dump(mode="json")
            != CallToolAction.model_validate(p.action).model_dump(mode="json")
            for p in selected
        ):
            raise ValueError("pending query no longer resolves to the exact current menu action")
        hypotheses = {h["hypothesis_id"]: h for p in selected for h in p.hypotheses}
        selected_needs = {n["need_id"]: n for p in selected for n in p.needs}
        choices = [c for p in selected for c in p.choices]
        if parsed is not None:
            hypotheses.update({h.hypothesis_id: h.model_dump(mode="json") for h in parsed.hypotheses})
            selected_needs.update({n.need_id: n.model_dump(mode="json") for n in parsed.needs})
        if len(hypotheses) > 4 or len(selected_needs) > 12 or len(choices) > 8:
            raise ValueError("selected round exceeds the Inquiry-v2 schema")
        round_draft = parse_inquiry_draft({"contract_version": 2,
            "hypotheses": list(hypotheses.values()), "needs": list(selected_needs.values()),
            "choices": choices}, live_version=2, strict=True)
        menu_by_ref = {p.query_ref: typed_menu[p.query_ref] for p in selected}
        bindings = selected_bindings(round_draft, menu_by_ref,
            [CallToolAction.model_validate(p.action) for p in selected], state.planning_steps + 1)
        validate_bindings(round_draft, bindings, state.planning_steps + 1, menu_by_ref,
                          max_bindings=16)
        round_record = {"round": state.planning_steps + 1,
            "draft": round_draft.model_dump(mode="json"), "selected_actions": bindings}
        for item in selected:
            pending[item.pending_id] = item.model_copy(update={"status": "scheduled",
                "bindings": tuple(b for b in bindings if b["query_ref"] == item.query_ref)})
        batch_id = "batch-" + _digest({"episode": state.episode_id,
                    "step": state.planning_steps + 1, "members": fingerprints})[:20]
        batches.append(LoopBatch(batch_id=batch_id, step=state.planning_steps + 1,
            pending_ids=tuple(p.pending_id for p in selected),
            original_order=tuple(fingerprints)))
        phase = "awaiting_results"
        planning_steps = state.planning_steps + 1
        review_steps = state.review_steps
    else:
        if state.review_steps or state.planning_steps == 0 and not step.batch_assessments:
            raise ValueError("empty Inquiry handoff is not a review")
        phase = "handoff"
        planning_steps = state.planning_steps
        review_steps = state.review_steps + 1
    next_state = _state(state, revision=state.revision + 1,
        planning_steps=planning_steps, review_steps=review_steps, phase=phase,
        candidates=tuple(c.model_dump(mode="json") for c in candidates.values()),
        needs=tuple(n.model_dump(mode="json") for n in needs.values()),
        pending=tuple(p.model_dump(mode="json") for p in pending.values()),
        batches=tuple(b.model_dump(mode="json") for b in batches),
        assessments=tuple(a.model_dump(mode="json") for a in (*state.assessments, *step.batch_assessments)))
    return next_state, round_record


def apply_loop_response(state: LoopState, response: ToolResponse, sequence: int) -> LoopState:
    if state.phase != "awaiting_results" or not state.batches:
        return state
    batch = state.batches[-1]
    ids = {p.pending_id: p for p in state.pending}
    member = next((ids[p] for p in batch.pending_ids
                   if ids[p].query_fingerprint == response.query_ref), None)
    if member is None or any(r.pending_id == member.pending_id for r in batch.results):
        return state
    result = LoopBatchResult(pending_id=member.pending_id,
        query_fingerprint=response.query_ref, status=response.status.value,
        observation_refs=tuple(o.observation_id for o in response.observations),
        observations=tuple(o.model_dump(mode="json") for o in response.observations),
        truncated=response.truncated, response_sequence=sequence)
    batch = batch.model_copy(update={"results": (*batch.results, result),
        "actual_order": (*batch.actual_order, response.query_ref)})
    pending = tuple(p.model_copy(update={"status": "executed"})
                    if p.pending_id == member.pending_id else p for p in state.pending)
    batches = (*state.batches[:-1], batch)
    return _state(state, revision=state.revision + 1, pending=tuple(p.model_dump(mode="json") for p in pending),
        batches=tuple(b.model_dump(mode="json") for b in batches),
        phase="awaiting_assessment" if len(batch.results) == len(batch.pending_ids) else "awaiting_results")


def visible_observations(events: tuple[Any, ...], *, limit: int = VIEW_LIMIT) -> dict[str, Any]:
    """Select whole typed observations, most recent first; never truncate JSON fragments."""
    import json

    rows = [Observation.model_validate(raw) for event in events
            if str(getattr(event.event_type, "value", event.event_type)) == "tool_response"
            for raw in event.payload.get("observations", ())]
    selected: list[dict[str, Any]] = []
    omitted: list[str] = []
    size = 0
    for observation in reversed(rows):
        raw = observation.model_dump(mode="json")
        encoded = json.dumps(raw, ensure_ascii=False, separators=(",", ":")).encode()
        if size + len(encoded) <= limit:
            selected.append(raw)
            size += len(encoded)
        else:
            omitted.append(observation.observation_id)
    selected.reverse()
    return {"observations": selected, "omitted_refs": omitted,
            "omitted_count": len(omitted), "limit_bytes": limit,
            "truncation_reason": "projection_limit" if omitted else None}


def make_loop_view(state: LoopState, events: tuple[Any, ...], bootstrap: Any, *,
                   budget_remaining: int, query_limit: int, batch_limit: int,
                   workspace_mode: str, host_loop_version: int = HOST_LOOP_VERSION_LEGACY
                   ) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Build the deterministic input and current typed menu from issued episode facts."""
    import json

    from rca_agent.investigation.inquiry_contract import (
        action_fingerprint,
        project_inquiry_menu,
        public_query,
    )
    from rca_agent.investigation.query_strategy import issued_entities
    from rca_agent.langchain_acquisition import query_options

    responses = [ToolResponse.model_validate(event.payload) for event in events
                 if str(getattr(event.event_type, "value", event.event_type)) == "tool_response"]
    observations = tuple(o for response in responses for o in response.observations)
    used = {event.payload["query_ref"] for event in events
            if str(getattr(event.event_type, "value", event.event_type)) == "tool_request"}
    options = query_options({"bootstrap": bootstrap.model_dump(mode="json")},
                            issued_entities(bootstrap, observations)[:8])
    costs = {str(getattr(cap.tool_name, "value", cap.tool_name)): cap.query_cost
             for cap in bootstrap.capabilities}
    menu = {ref: {**item, "action": item["action"].model_dump(mode="json"),
                  "query_cost": costs[item["action"].tool_name.value]}
            for ref, item in options.items()
            if query_fingerprint(item["action"].tool_name, item["action"].request) not in used}
    existing = used | {item.query_fingerprint for item in state.pending
                       if item.status != "cancelled"}
    projection = project_inquiry_menu(
        menu,
        [item.need for item in state.needs if item.disposition == "open"],
        existing_query_fingerprints=existing,
        budget_remaining=budget_remaining,
    )
    latest = state.batches[-1] if state.batches else None
    last_results = [] if latest is None else [
        {"pending_id": result.pending_id, "query_fingerprint": result.query_fingerprint,
         "status": result.status, "observation_refs": result.observation_refs,
         "truncated": result.truncated, "response_sequence": result.response_sequence,
         "bindings": next(p.bindings for p in state.pending if p.pending_id == result.pending_id)}
        for result in latest.results
    ]
    actionable_pending, pending_history = pending_public_projection(state)
    view: dict[str, Any] = {
        "contract_version": LOOP_VERSION, "episode_id": state.episode_id,
        "based_on_revision": state.revision, "based_on_digest": state.digest,
        "phase": state.phase, "planning_steps": state.planning_steps,
        "review_steps": state.review_steps, "remaining_planning_steps": max(0, 5-state.planning_steps),
        "remaining_review_steps": max(0, 1-state.review_steps),
        "query_limit": query_limit, "batch_limit": batch_limit,
        "executed_queries": sum(p.status == "executed" for p in state.pending),
        "budget_remaining": budget_remaining, "workspace_mode": workspace_mode,
        "last_batch": None if latest is None else {"batch_id": latest.batch_id,
            "pending_ids": latest.pending_ids, "results": last_results,
            "assessed": latest.assessed},
        "candidates": [c.model_dump(mode="json") for c in state.candidates],
        "needs": [n.model_dump(mode="json") for n in state.needs],
        # Keep the historical key as an actionable-only alias in v4; the explicit
        # names below are authoritative for new consumers.
        "pending": actionable_pending,
        "assessments": [a.model_dump(mode="json") for a in state.assessments],
        "observations": [], "omitted_refs": [], "omitted_count": 0,
        "truncation_reason": None,
        "menu": {f"q{i}": {**public_query({**item,
                 "action": CallToolAction.model_validate(item["action"])}),
                 "query_cost": item["query_cost"]}
                 for i, (ref, item) in enumerate(menu.items(), 1)},
    }
    if host_loop_version >= HOST_LOOP_VERSION:
        full_aliases = {ref: f"q{i}" for i, ref in enumerate(menu, 1)}

        def public_menu(items):
            return {
                full_aliases[ref]: {**public_query({**item,
                    "action": CallToolAction.model_validate(item["action"])}),
                    "menu_ref": ref,
                    "action_fingerprint": action_fingerprint(
                        CallToolAction.model_validate(item["action"])),
                    "query_cost": item["query_cost"]}
                for ref, item in items.items()
            }

        focused_public = public_menu(projection["focused"])
        exploration_public = (public_menu(projection["exploration"])
                              if projection["mode"] != "focused" else {})
        exposed_public = public_menu(projection["exposed"])
        aliases = {
            alias: {"menu_ref": item["menu_ref"],
                    "action_fingerprint": item["action_fingerprint"]}
            for alias, item in {**focused_public, **exploration_public}.items()
        }
        view["inquiry_loop_version"] = host_loop_version
        view["menu"] = exposed_public
        view["focused_menu"] = focused_public
        view["exploration_menu"] = exploration_public
        view["menu_mode"] = projection["mode"]
        view["menu_aliases"] = aliases
        view["offered_menu_refs"] = sorted(
            {item["menu_ref"] for item in aliases.values()})
        view["menu_projection"] = {
            "mode": projection["mode"],
            "eligible_count": projection["eligible_count"],
            "focused_count": projection["focused_count"],
            "exposed_count": len(exposed_public),
            "exploration_count": len(exploration_public),
            "min_focused": projection["min_focused"],
            "max_focused": projection["max_focused"],
        }
        view["menu_projection_digest"] = sha256_digest({
            "state_digest": state.digest,
            "existing_query_fingerprints": sorted(existing),
            "focused_refs": list(projection["focused"]),
            "exploration_refs": list(projection["exploration"]),
            "exposed_refs": list(projection["exposed"]),
            "menu_aliases": aliases,
        })
        view["actionable_pending"] = actionable_pending
        view["pending_history"] = pending_history
        view["expected_actionable_ids"] = [item["pending_id"] for item in actionable_pending]
        view["allowed_pending_dispositions"] = ["retain", "defer", "cancel"]
        view["existing_queries"] = [
            {"pending_id": item.pending_id, "query_fingerprint": item.query_fingerprint,
             "status": item.status} for item in state.pending
        ]
    # The public menu projection is independently bounded by host rules; it is
    # excluded from the 128 KiB result/state projection budget and still recorded.
    menu_public = view.pop("menu")
    focused_public = view.pop("focused_menu", None)
    exploration_public = view.pop("exploration_menu", None)
    menu_mode = view.pop("menu_mode", None)
    menu_aliases = view.pop("menu_aliases", None)
    offered_menu_refs = view.pop("offered_menu_refs", None)
    menu_projection = view.pop("menu_projection", None)
    menu_projection_digest = view.pop("menu_projection_digest", None)
    def byte_size() -> int:
        return len(json.dumps(view, ensure_ascii=False, separators=(",", ":")).encode())

    if byte_size() > VIEW_LIMIT:
        raise ValueError("Inquiry state metadata exceeds the bounded projection")
    current_refs = {ref for row in last_results for ref in row["observation_refs"]}
    ordered = [o for o in observations if o.observation_id in current_refs]
    ordered += [o for o in reversed(observations) if o.observation_id not in current_refs]
    seen: set[str] = set()
    for observation in ordered:
        if observation.observation_id in seen:
            continue
        seen.add(observation.observation_id)
        raw = observation.model_dump(mode="json")
        view["observations"].append(raw)
        if byte_size() > VIEW_LIMIT:
            view["observations"].pop()
            view["omitted_refs"].append(observation.observation_id)
    view["omitted_count"] = len(view["omitted_refs"])
    if view["omitted_refs"]:
        view["truncation_reason"] = "projection_limit"
    if byte_size() > VIEW_LIMIT:
        raise ValueError("Inquiry view omission metadata exceeds its bound")
    view["menu"] = menu_public
    if host_loop_version >= HOST_LOOP_VERSION:
        view["focused_menu"] = focused_public
        view["exploration_menu"] = exploration_public
        view["menu_mode"] = menu_mode
        view["menu_aliases"] = menu_aliases
        view["offered_menu_refs"] = offered_menu_refs
        view["menu_projection"] = menu_projection
        view["menu_projection_digest"] = menu_projection_digest
    return json.loads(json.dumps(view, ensure_ascii=False)), json.loads(
        json.dumps(menu, ensure_ascii=False))


def loop_handoff(state: LoopState, rounds: list[dict[str, Any]],
                 events: tuple[Any, ...]) -> dict[str, Any]:
    """Project the reviewed authoritative state into the existing Discovery boundary."""
    from rca_agent.investigation.contract_enforcement import execution_binding_responses
    from rca_agent.investigation.inquiry_contract import normalized
    from rca_agent.langchain_acquisition import InquiryHandoff, inquiry_candidate_records

    if state.phase != "handoff" or not state.verify_digest() or not rounds:
        raise ValueError("Inquiry loop has no reviewed Discovery handoff")
    records = list(inquiry_candidate_records(rounds))
    priorities = {(
        normalized(item.hypothesis["mechanism"]), normalized(item.hypothesis["alternative"])
    ): item for item in state.candidates}
    original_order = {id(item): index for index, item in enumerate(records)}
    records.sort(key=lambda item: (-priorities[(normalized(item.hypothesis.mechanism),
                         normalized(item.hypothesis.alternative))].priority,
                         original_order[id(item)]))
    by_key = {(normalized(item.hypothesis.mechanism),
               normalized(item.hypothesis.alternative)): item.hypothesis.hypothesis_id
              for item in records}
    responses = execution_binding_responses(events)
    links = []
    for record in rounds:
        hypotheses = {h["hypothesis_id"]: h for h in record["draft"]["hypotheses"]}
        needs = {n["need_id"]: n for n in record["draft"]["needs"]}
        for selected in record["selected_actions"]:
            response = responses.get(selected["binding_id"])
            if response is None:
                continue
            need = needs[selected["need_id"]]
            hypothesis = hypotheses[need["hypothesis_id"]]
            links.append({"binding_id": selected["binding_id"],
                "source_round": record["round"], "source_need_id": need["need_id"],
                "mode": selected["mode"], "measurement_target": selected["measurement_target"],
                "question": need["question"],
                "hypothesis_id": by_key[(normalized(hypothesis["mechanism"]),
                                          normalized(hypothesis["alternative"]))],
                "signal": selected["signal"], "entity_ref": selected["entity_ref"],
                "query_ref": response["query_ref"], "status": response["status"],
                "observation_refs": [item["observation_id"]
                                     for item in response["observations"]]})
    return InquiryHandoff(
        revision=rounds[-1]["round"], hypotheses=tuple(item.hypothesis for item in records),
        candidate_history=tuple(records), needs=rounds[-1]["draft"]["needs"],
        query_links=tuple(links), loop_contract_version=1,
        loop_state_revision=state.revision, loop_state_digest=state.digest,
        candidate_priorities=tuple({"hypothesis_id": item.hypothesis["hypothesis_id"],
            "priority": item.priority, "disposition": item.disposition,
            "reason": item.reason} for item in state.candidates),
        all_needs=tuple(item.model_dump(mode="json") for item in state.needs),
        need_assessments=tuple(item.model_dump(mode="json") for item in state.assessments),
        pending_outcomes=tuple({"pending_id": item.pending_id, "status": item.status,
            "reason": item.reason, "query_fingerprint": item.query_fingerprint,
            "binding_ids": [b["binding_id"] for b in item.bindings]}
            for item in state.pending),
    ).model_dump(mode="json")
