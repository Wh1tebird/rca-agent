"""The existing Inquiry role with a host-bound, result-driven batch envelope."""
from __future__ import annotations

import json
from time import monotonic
from typing import Any, Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from pydantic import BaseModel, ValidationError

from rca_agent.inquiry_policy import InquiryInvestigationPolicy, InquiryUnavailableError
from rca_agent.investigation.inquiry_loop import (
    LoopState,
    LoopStep,
    accept_loop_step,
    loop_attempt_digest,
    normalize_loop_step,
    provider_loop_schema,
)
from rca_agent.langchain_acquisition import InquiryOutputError
from rca_agent.models import CallToolAction
from rca_agent.tools.fingerprint import sha256_digest


class LangChainBatchedInquiryAdapter:
    """One bounded step call evaluates the previous batch and chooses the next."""

    contract_version = 2

    def __init__(self, runnable: Runnable[dict[str, Any], Any]) -> None:
        self.runnable = runnable

    @classmethod
    def from_chat_model(
        cls, model: Any, *,
        structured_method: Literal["function_calling", "json_mode", "json_schema"] = "json_schema",
    ) -> LangChainBatchedInquiryAdapter:
        prompt = ChatPromptTemplate.from_messages([
            ("system", ("You are the same non-authoritative Inquiry role. Assess the last fully "
             "executed batch using only visible results. Use actionable_pending and its "
             "expected_actionable_ids for retain/defer/cancel decisions; pending_history is "
             "read-only and must never receive a disposition. In batch_assessments, query_ref is a completed-result "
             "fingerprint copied exactly from last_batch.results[].query_fingerprint (normally qry-...); never use "
             "the menu query ref (aq-... or qN alias) there. In draft.choices, query_ref is the menu ref. "
             "observation_refs must come only from that result's observation_refs. In draft.hypotheses, leave "
             "alternative_id null; the host derives original alternative identity and rejects invented alt-* IDs. "
             "For exploratory bindings, a result is never complete evidence: use status partial, ambiguous, or "
             "unavailable. Use status answered only when a direct binding has a complete successful observation "
             "matching the requested entity, signal, and measurement kind. "
             "Read existing_queries before drafting: each pending_id, query_fingerprint, and status "
             "is an existing physical query. Never recreate one in draft.choices. Only actionable "
             "pending/deferred/unavailable items belong in pending_decisions; executed history is read-only. "
             "Existing Need IDs and their contents are immutable. Reuse an old need_id only with exactly "
             "the same content; for a changed Need create a new need_id and set "
             "need_updates[].supersedes_need_id to the old ID. "
             "Then choose at most two physical queries from this menu "
             "or hand off. Model assessments cannot claim Core verification. Follow this schema: "
             "{step_schema}")),
            ("human", "Host state and current menu:\n{state_json}\nRepair feedback for this attempt (empty on first attempt):\n{repair_feedback_json}"),
        ]).partial(step_schema=json.dumps(provider_loop_schema(), ensure_ascii=False))
        structured = model.with_structured_output(LoopStep, method=structured_method)
        return cls(prompt | structured)

    def step(self, incident: dict[str, Any], view: dict[str, Any],
             repair_feedback: dict[str, Any] | None = None) -> dict[str, Any]:
        repair_feedback = repair_feedback or {}
        payload = {"state": view["view"], "state_json": json.dumps(view["view"], ensure_ascii=False),
                   "schema": provider_loop_schema(), "incident": incident,
                   "repair_feedback": repair_feedback,
                   "repair_feedback_json": json.dumps(repair_feedback, ensure_ascii=False)}
        raw = self.runnable.invoke(payload)
        if isinstance(raw, dict) and "parsed" in raw and "raw" in raw:
            response = raw.get("parsed")
            if response is None:
                message = raw.get("raw")
                calls = getattr(message, "tool_calls", ()) or ()
                if len(calls) == 1 and calls[0].get("name") == "LoopStep":
                    response = calls[0].get("args")
                else:
                    content = getattr(message, "content", "")
                    if not isinstance(content, str) or not content.strip():
                        raise InquiryOutputError("empty_response")
                    try:
                        response = json.loads(content)
                    except json.JSONDecodeError as exc:
                        raise InquiryOutputError("invalid_json") from exc
            raw = response
        if isinstance(raw, BaseModel):
            raw = raw.model_dump(mode="json")
        if not isinstance(raw, dict):
            raise InquiryOutputError("schema_mismatch", fields=("response",))
        return raw


class BatchedInquiryInvestigationPolicy(InquiryInvestigationPolicy):
    """Keeps only a proposal queue; Runner/Workspace commit every actual state change."""

    loop_mode = "batched"

    def __init__(self, inquiry: LangChainBatchedInquiryAdapter, *, query_limit: int = 10) -> None:
        super().__init__(inquiry, query_limit=query_limit)
        self._loop_steps: list[tuple[dict[str, Any], dict[str, Any]]] = []
        self._loop_attempts: list[dict[str, Any]] = []
        self._latest_loop_view: dict[str, Any] | None = None
        self._startup_action: CallToolAction | None = None

    def invoke(self, input: dict[str, Any], config: Any = None,
               **kwargs: Any):
        envelope = super().invoke(input, config, **kwargs)
        self._startup_action = (envelope.action if isinstance(envelope.action, CallToolAction)
                                and self._pending_selected_action is None else None)
        return envelope

    def consume_startup_action(self, action: CallToolAction) -> bool:
        matched = self._startup_action == action
        self._startup_action = None
        return matched

    def drain_inquiry_loop_steps(self) -> tuple[tuple[dict[str, Any], dict[str, Any]], ...]:
        items = tuple(self._loop_steps)
        self._loop_steps.clear()
        return items

    def drain_inquiry_loop_attempts(self) -> tuple[dict[str, Any], ...]:
        items = tuple(self._loop_attempts)
        self._loop_attempts.clear()
        return items

    def _inquiry_screen(self, context: dict[str, Any]) -> CallToolAction | None:
        if self.selection_queue:
            action = self.selection_queue.pop(0)
            self.selected_actions.append(action)
            self._pending_selected_action = action
            self._acquisition_actions += 1
            return action
        if self._inquiry_finished:
            return None
        bundle = context.get("inquiry_state_view")
        if not isinstance(bundle, dict):
            raise InquiryUnavailableError("batched Inquiry lacks host state")
        self._latest_loop_view = bundle
        state = LoopState.model_validate(context["investigation_workspace"]["inquiry_loop"])
        if state.phase == "awaiting_results":
            raise InquiryUnavailableError("batch ended before its accepted members completed")
        if state.phase == "handoff":
            self._inquiry_finished = True
            return None
        for attempt in (1, 2):
            raw = None
            started = monotonic()
            repair_feedback = context.get("batched_repair_feedback", {}) if attempt == 2 else {}
            stage = "adapter"
            try:
                raw = self.inquiry.step(context, bundle, repair_feedback=repair_feedback)
                stage = "normalize"
                step = normalize_loop_step(raw, bundle["menu"])
                visible = {item["observation_id"] for item in bundle["view"]["observations"]}
                stage = "accept"
                next_state, round_record = accept_loop_step(state, step,
                    menu=bundle["menu"], batch_limit=bundle["view"]["batch_limit"],
                    query_limit=self.query_limit, visible_refs=visible,
                    budget_remaining=bundle["view"]["budget_remaining"])
                self._loop_steps.append((step.model_dump(mode="json"), bundle))
                self._loop_attempts.append({"step": state.planning_steps + 1,
                    "attempt": attempt, "status": "parsed", "code": "parsed",
                    "input_digest": loop_attempt_digest(bundle,
                        repair_feedback=repair_feedback), "raw_digest": sha256_digest(raw),
                    "repair_feedback": repair_feedback,
                    "pending_diagnostics": {},
                    "stage": None, "error_type": None,
                    "state_revision": state.revision, "state_digest": state.digest,
                    "elapsed_ms": int((monotonic() - started) * 1000)})
                if round_record is not None:
                    self.selection_calls = next_state.planning_steps
                    self.inquiry_drafts.append(round_record)
                    self.selection_queue = [CallToolAction.model_validate(item["action"])
                                            for item in round_record["selected_actions"]]
                    # Physical shared bindings may duplicate an Action; the Runner
                    # executes each batch member once, preserving all binding records.
                    seen_actions: set[str] = set()
                    unique_actions = []
                    for action in self.selection_queue:
                        key = action.model_dump_json()
                        if key not in seen_actions:
                            seen_actions.add(key)
                            unique_actions.append(action)
                    self.selection_queue = unique_actions
                else:
                    self._inquiry_finished = True
                return self._inquiry_screen(context) if self.selection_queue else None
            except Exception as exc:
                code = (exc.code if isinstance(exc, InquiryOutputError)
                        else getattr(exc, "code", type(exc).__name__))
                fields = (tuple(str(field) for field in exc.fields)
                          if isinstance(exc, InquiryOutputError) else
                          tuple(".".join(str(part) for part in (
                              (*error["loc"][:-1], "extra") if error["type"] == "extra_forbidden"
                              else error["loc"]))[:120]
                                for error in exc.errors()[:20])
                          if isinstance(exc, ValidationError) else
                          ("pending_decisions",) if getattr(exc, "code", None) == "PENDING_DECISION_SET_MISMATCH" else ())
                if not fields and getattr(exc, "fields", ()):
                    fields = tuple(str(field) for field in exc.fields)
                if not fields:
                    fields = (stage,)
                safe_details = getattr(exc, "safe_details", None)
                if (safe_details is None and stage == "accept" and isinstance(exc, ValueError)
                        and getattr(exc, "code", None) is None):
                    # Keep the host strict, but make otherwise opaque admission failures
                    # actionable on the single allowed retry without exposing raw provider data.
                    safe_details = str(exc)[:500]
                    code = "LOOP_ADMISSION_REJECTED"
                self.selection_errors.append(code)
                safe_keys = set(LoopStep.model_fields)
                self._loop_attempts.append({"step": state.planning_steps + 1,
                    "attempt": attempt, "status": "failed", "code": code,
                    "input_digest": loop_attempt_digest(bundle,
                        repair_feedback=repair_feedback),
                    "raw_digest": sha256_digest(raw) if isinstance(raw, dict) else None,
                    "repair_feedback": repair_feedback,
                    "pending_diagnostics": {
                        "expected": list(getattr(exc, "expected", ())),
                        "missing": list(getattr(exc, "missing", ())),
                        "unexpected": list(getattr(exc, "unexpected", ())),
                        "duplicate": list(getattr(exc, "duplicate", ())),
                        "invalid_states": dict(getattr(exc, "invalid_states", {})),
                        "supplied": getattr(exc, "supplied", None),
                        "expected_format": getattr(exc, "expected_format", None),
                        "available_result_ids": list(getattr(exc, "available_result_ids", ())),
                        "evidence_error": getattr(exc, "evidence_error", None),
                        "details": safe_details,
                        "conflict_id": getattr(exc, "conflict_id", None),
                        "conflict_status": getattr(exc, "conflict_status", None),
                        "query_fingerprint": getattr(exc, "query_fingerprint", None),
                        "repair_instruction": getattr(exc, "instruction", None),
                    },
                    "fields": fields,
                    "raw_shape": {key: type(value).__name__ for key, value in raw.items()
                                  if key in safe_keys} if isinstance(raw, dict) else {},
                    "stage": stage if stage in {"adapter", "normalize", "accept"} else "unknown",
                    "error_type": type(exc).__name__,
                    "state_revision": state.revision, "state_digest": state.digest,
                    "elapsed_ms": int((monotonic() - started) * 1000)})
                if attempt == 2 or getattr(exc, "status_code", None) is not None:
                    raise InquiryUnavailableError("batched Inquiry attempts exhausted") from exc
                context["batched_repair_feedback"] = {
                    "code": code,
                    "fields": list(fields),
                    "missing": list(getattr(exc, "missing", ())),
                    "unexpected": list(getattr(exc, "unexpected", ())),
                    "duplicate": list(getattr(exc, "duplicate", ())),
                    "invalid_states": dict(getattr(exc, "invalid_states", {})),
                    "supplied": getattr(exc, "supplied", None),
                    "expected_format": getattr(exc, "expected_format", None),
                    "available_result_ids": list(getattr(exc, "available_result_ids", ())),
                    "evidence_error": getattr(exc, "evidence_error", None),
                    "details": safe_details,
                    "conflict_id": getattr(exc, "conflict_id", None),
                    "conflict_status": getattr(exc, "conflict_status", None),
                    "query_fingerprint": getattr(exc, "query_fingerprint", None),
                    "repair_instruction": getattr(exc, "instruction", None),
                    "instruction": (
                        "Repair only the declared fields against the same host state. "
                        "For batch_assessments[].query_ref copy an available result ID "
                        "exactly; menu refs are valid only in draft.choices. "
                        "For exploratory results use status partial, ambiguous, or unavailable; "
                        "answered requires a complete direct result matching the Need."
                    ),
                }
        return None
