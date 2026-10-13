"""GT-free inquiry acquisition policy for provider-backed investigations.

This is the application-side counterpart of the evaluation acquisition arm. It only
collects bounded evidence and preserves revisable ideas for Discovery; it has no Core
decision authority and never loads benchmark or evaluator data.
"""
from __future__ import annotations

import re
from typing import Any, Literal

from langchain_core.runnables import Runnable
from pydantic import BaseModel, ConfigDict, Field

from rca_agent.langchain_acquisition import (
    InquiryHandoff,
    InquiryOutputError,
    LangChainInquiryAdapter,
    inquiry_candidate_records,
    query_options,
)
from rca_agent.langchain_policy import PolicyEnvelope
from rca_agent.models import (
    CallToolAction,
    GetServiceTraceSummaryRequest,
    GetTraceStageBreakdownRequest,
    HypothesisProposal,
    ListRelatedEntitiesRequest,
    RequestStopAction,
    SubmitHypothesisAction,
    ToolName,
)


class InquiryUnavailableError(RuntimeError):
    """Bounded Inquiry attempts failed; do not silently bypass investigation."""


class InquiryAttempt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    round: int = Field(ge=1, le=5)
    attempt: int = Field(ge=0, le=2)
    status: Literal["accepted", "failed", "unavailable"]
    code: str = Field(min_length=1, max_length=100, pattern=r"^[A-Za-z][A-Za-z_ ]*$")
    fields: tuple[str, ...] = Field(max_length=8)
    stage: Literal["inquiry_contract"] | None = Field(default=None, exclude_if=lambda v: v is None)


def validate_inquiry_attempt(record: dict[str, Any]) -> dict[str, Any]:
    attempt = InquiryAttempt.model_validate(record)
    if ((attempt.attempt == 0) != (attempt.status == "unavailable")
            or (attempt.status == "accepted" and attempt.code != "accepted")
            or (attempt.status == "unavailable" and attempt.code != "no_offered_queries")):
        raise ValueError("inconsistent Inquiry attempt")
    for field in attempt.fields:
        if (len(field) > 180 or not re.fullmatch(r"[A-Za-z0-9_.:]+", field)
                or field.split(".")[0].split(":")[0] not in {
            "hypotheses", "needs", "choices", "response", "contract_version",
        }):
            raise ValueError("invalid Inquiry diagnostic field")
    return attempt.model_dump(mode="json")


class InquiryInvestigationPolicy(Runnable[dict[str, Any], PolicyEnvelope]):
    """Acquire evidence through Inquiry, then hand retained ideas to Discovery."""

    def __init__(self, inquiry: LangChainInquiryAdapter, *, query_limit: int = 10) -> None:
        if not 1 <= query_limit <= 10:
            raise ValueError("inquiry query limit must be between one and ten")
        self.inquiry = inquiry
        from rca_agent.investigation.contract_enforcement import validate_inquiry_host
        self.inquiry_contract_version = validate_inquiry_host(getattr(inquiry, "contract_version", 2))
        self.query_limit = query_limit
        self.selection_calls = 0
        self.selection_errors: list[str] = []
        self.selection_diagnostics: list[dict[str, Any]] = []
        self._diagnostic_events: list[dict[str, Any]] = []
        self.selected_actions: list[CallToolAction] = []
        self.executed_actions: list[CallToolAction] = []
        self.selection_queue: list[CallToolAction] = []
        self.inquiry_drafts: list[dict[str, Any]] = []
        self._binding_history_offsets: dict[int, int] = {}
        self._step = 0
        self._tried_trace_refs: set[str] = set()
        self._summarized_entities: set[str] = set()
        self._expanded_entities: set[str] = set()
        self._acquisition_actions = 0
        self._pending_expand: str | None = None
        self._inquiry_finished = False
        self._routing_feedback_seen = False
        self._pending_selected_action: CallToolAction | None = None

    def drain_inquiry_diagnostics(self) -> tuple[dict[str, Any], ...]:
        records = tuple(self._diagnostic_events)
        self._diagnostic_events.clear()
        return records

    def record_routed_action(
        self, proposed: object, routed: object | None,
    ) -> None:
        """Count an Inquiry query only after active routing actually lets it execute."""

        if self._pending_selected_action is None or proposed != self._pending_selected_action:
            if isinstance(routed, CallToolAction) and any(
                selected["action"] == routed.model_dump(mode="json")
                for record in self.inquiry_drafts for selected in record["selected_actions"]
            ):
                self._routing_feedback_seen = True
                self.executed_actions.append(routed)
            return
        self._routing_feedback_seen = True
        if isinstance(routed, CallToolAction):
            self.executed_actions.append(routed)
        self._pending_selected_action = None

    def _query_count(self) -> int:
        return len(self.executed_actions if self._routing_feedback_seen else self.selected_actions)

    def invoke(
        self,
        input: dict[str, Any],
        config: Any = None,
        **kwargs: Any,
    ) -> PolicyEnvelope:
        del config, kwargs
        context = input["context"]
        if self._step == 0:
            self._step += 1
            self._acquisition_actions += 1
            bootstrap = context["bootstrap"]
            entity_ref = bootstrap["alert_entity"]["entity_ref"]
            self._summarized_entities.add(entity_ref)
            return PolicyEnvelope(action=CallToolAction(
                tool_name=ToolName.GET_SERVICE_TRACE_SUMMARY,
                request=GetServiceTraceSummaryRequest(
                    entity_ref=entity_ref,
                    window=bootstrap["observation_horizon"],
                    min_duration_ms=100,
                    max_traces=4,
                ),
            ))

        observations = _history_observations(context.get("history", ()))
        has_stage_context = any(
            item.get("signal") in {"sql_duration_share", "connection_duration_ms"}
            for item in observations
        )
        participant_seen = any(
            item.get("signal") == "trace_participant" for item in observations
        )
        trace_refs = [
            item.get("details", {}).get("trace_ref")
            for item in observations
            if item.get("signal") == "operation_trace_duration_ms"
        ]
        next_trace = next(
            (ref for ref in trace_refs if ref and ref not in self._tried_trace_refs),
            None,
        )
        if not has_stage_context and not participant_seen and next_trace is not None:
            self._tried_trace_refs.add(next_trace)
            self._acquisition_actions += 1
            return PolicyEnvelope(action=CallToolAction(
                tool_name=ToolName.GET_TRACE_STAGE_BREAKDOWN,
                request=GetTraceStageBreakdownRequest(trace_ref=next_trace, max_stages=12),
            ))

        if participant_seen or has_stage_context:
            action = self._inquiry_screen(context)
            if action is not None:
                return PolicyEnvelope(action=action)
            self._inquiry_finished = True

        if not has_stage_context and not self._inquiry_finished and self._acquisition_actions < 16:
            action = self._entity_exploration(context)
            if action is not None:
                return PolicyEnvelope(action=action)

        if self._step == 1:
            self._step += 1
            bootstrap = context["bootstrap"]
            return PolicyEnvelope(action=SubmitHypothesisAction(
                proposal=HypothesisProposal(
                    hypothesis_id="h-runtime-discovery",
                    candidate_id="runtimeUnknown",
                    mechanism="unexplained operation latency",
                    causal_entity_ref=bootstrap["alert_entity"]["entity_ref"],
                    rationale=(
                        "The bounded investigation requires formalization of the retained "
                        "Inquiry candidates against the issued evidence."
                    ),
                )
            ))
        return PolicyEnvelope(action=RequestStopAction(
            summary="The bounded Inquiry investigation completed available verification."
        ))

    def _entity_exploration(self, context: dict[str, Any]) -> CallToolAction | None:
        entities = _history_entities(context)
        pending = next(
            (entity for entity in entities
             if entity["entity_ref"] == self._pending_expand
             and entity["entity_ref"] not in self._expanded_entities),
            None,
        )
        if pending is not None:
            self._expanded_entities.add(pending["entity_ref"])
            self._pending_expand = None
            self._acquisition_actions += 1
            return CallToolAction(
                tool_name=ToolName.LIST_RELATED_ENTITIES,
                request=ListRelatedEntitiesRequest(
                    entity_ref=pending["entity_ref"], max_results=50,
                ),
            )
        service = next(
            (entity for entity in entities
             if entity["entity_type"] == "apm.service"
             and entity["entity_ref"] not in self._summarized_entities),
            None,
        )
        if service is not None:
            self._summarized_entities.add(service["entity_ref"])
            self._pending_expand = service["entity_ref"]
            self._acquisition_actions += 1
            return CallToolAction(
                tool_name=ToolName.GET_SERVICE_TRACE_SUMMARY,
                request=GetServiceTraceSummaryRequest(
                    entity_ref=service["entity_ref"],
                    window=context["bootstrap"]["observation_horizon"],
                    min_duration_ms=100,
                    max_traces=4,
                ),
            )
        expandable = next(
            (entity for entity in entities
             if entity["entity_type"] in {"apm.operation", "apm.service"}
             and entity["entity_ref"] not in self._expanded_entities),
            None,
        )
        if expandable is None:
            return None
        self._expanded_entities.add(expandable["entity_ref"])
        self._acquisition_actions += 1
        return CallToolAction(
            tool_name=ToolName.LIST_RELATED_ENTITIES,
            request=ListRelatedEntitiesRequest(
                entity_ref=expandable["entity_ref"], max_results=50,
            ),
        )

    def _inquiry_screen(self, context: dict[str, Any]) -> CallToolAction | None:
        if self._query_count() >= self.query_limit:
            self.selection_queue.clear()
            return None
        if not self.selection_queue and self.selection_calls < 5:
            self.selection_calls += 1
            options = query_options(context, _history_entities(context)[:8])
            # The newer Inquiry contract uses a deterministic bounded menu so
            # its provider request remains within the host context budget. The
            # legacy contract predates that menu projection and its frozen
            # callers receive the complete typed capability menu; silently
            # truncating it changes the legacy query-reference namespace.
            if self.inquiry_contract_version >= 2:
                options = dict(list(options.items())[:64])
            used = {action.model_dump_json() for action in (
                self.executed_actions if self._routing_feedback_seen else self.selected_actions
            )}
            options = {
                ref: item for ref, item in options.items()
                if item["action"].model_dump_json() not in used
            }
            if not options:
                self._diagnostic_events.append({
                    "round": self.selection_calls, "attempt": 0,
                    "status": "unavailable", "code": "no_offered_queries", "fields": [],
                })
                return None
            observations = _history_observations(context.get("history", ()))
            incident: dict[str, Any] = {
                "alert": context["bootstrap"]["alert_summary"],
                "issued_entities": _history_entities(context)[:8],
                "recent_observations": [{
                    "signal": item.get("signal"),
                    "entity_ref": item.get("entity_ref"),
                    "data_status": item.get("data_status"),
                    "value": str(item.get("value"))[:500],
                } for item in observations[-60:]],
                "previous_inquiry": {
                    "hypotheses": self.inquiry_drafts[-1]["draft"]["hypotheses"],
                    "needs": self.inquiry_drafts[-1]["draft"]["needs"],
                } if self.inquiry_drafts else None,
                "carried_candidates": [
                    {
                        "hypothesis_id": item.hypothesis.hypothesis_id,
                        "mechanism": item.hypothesis.mechanism,
                        "alternative": item.hypothesis.alternative,
                    }
                    for item in inquiry_candidate_records(self.inquiry_drafts)
                ],
            }
            self._binding_history_offsets[self.selection_calls] = len(context.get("history", ()))
            self._plan_inquiry_round(incident, options)
        if not self.selection_queue:
            return None
        action = self.selection_queue.pop(0)
        self.selected_actions.append(action)
        self._pending_selected_action = action
        self._acquisition_actions += 1
        return action

    def _plan_inquiry_round(
        self,
        incident: dict[str, Any],
        options: dict[str, dict],
    ) -> None:
        for attempt in range(2):
            try:
                draft, actions = self.inquiry.plan(incident, options)
                from rca_agent.langchain_acquisition import parse_inquiry_draft
                draft = parse_inquiry_draft(draft, live_version=self.inquiry_contract_version)
                remaining = self.query_limit - self._query_count()
                from rca_agent.investigation.inquiry_contract import (
                    selected_bindings,
                    validate_bindings,
                )
                bindings = selected_bindings(draft, options, actions[:remaining], self.selection_calls)
                if draft.contract_version == 2:
                    validate_bindings(draft, bindings, self.selection_calls, options)
                self.selection_queue = list(actions[:remaining])
                self.inquiry_drafts.append({
                    "round": self.selection_calls,
                    "attempt": attempt + 1,
                    "draft": draft.model_dump(mode="json"),
                    "selected_actions": bindings,
                })
                self._diagnostic_events.append({
                    "round": self.selection_calls, "attempt": attempt + 1,
                    "status": "accepted", "code": "accepted", "fields": [],
                })
                return
            except Exception as exc:
                if isinstance(exc, InquiryOutputError):
                    self.selection_diagnostics.append({
                        "round": self.selection_calls,
                        "attempt": attempt + 1,
                        **exc.diagnostic(),
                    })
                safe_errors = {
                    "duplicate inquiry identifiers",
                    "inquiry need has no hypothesis",
                    "invalid inquiry choices",
                    "inquiry choice is not offered",
                }
                code = exc.code if isinstance(exc, InquiryOutputError) else (
                    str(exc)
                    if type(exc) is ValueError and str(exc) in safe_errors
                    else type(exc).__name__
                )
                self.selection_errors.append(code)
                fields = list(exc.fields) if isinstance(exc, InquiryOutputError) else []
                self._diagnostic_events.append({
                    "round": self.selection_calls, "attempt": attempt + 1,
                    "status": "failed", "code": code, "fields": fields,
                    "stage": "inquiry_contract",
                })
                if getattr(exc, "status_code", None) is not None or attempt == 1:
                    raise InquiryUnavailableError("Inquiry attempts exhausted") from exc
                incident["validation_error"] = (
                    f"Previous response failed: {code}; fields: {', '.join(fields)}. "
                    "Repair these fields without changing your causal ideas. Return the "
                    "requested schema, using unique offered qN aliases as query_refs and "
                    "declared need_ids. Every need must reference a declared hypothesis_id."
                )

    def discovery_handoff(self, history: tuple[dict[str, Any], ...]) -> dict[str, Any] | None:
        if not self.inquiry_drafts:
            return None
        candidates = inquiry_candidate_records(self.inquiry_drafts)
        candidate_by_key = {
            (_normalize(item.hypothesis.mechanism), _normalize(item.hypothesis.alternative)):
            item.hypothesis.hypothesis_id
            for item in candidates
        }
        from rca_agent.investigation.contract_enforcement import history_binding_responses
        responses = history_binding_responses(self.inquiry_drafts, history, self._binding_history_offsets)
        links = []
        for round_record in self.inquiry_drafts:
            questions = {item["need_id"]: item for item in round_record["draft"]["needs"]}
            round_candidates = {
                item["hypothesis_id"]: candidate_by_key[
                    (_normalize(item["mechanism"]), _normalize(item["alternative"]))
                ]
                for item in round_record["draft"]["hypotheses"]
            }
            for selected in round_record["selected_actions"]:
                response = responses.get((round_record["round"], selected["need_id"], selected["query_ref"]))
                if response is None:
                    continue
                need = questions[selected["need_id"]]
                links.append({
                    **({"binding_id": selected["binding_id"], "source_round": round_record["round"],
                        "source_need_id": selected["need_id"], "mode": selected["mode"],
                        "measurement_target": selected["measurement_target"]}
                       if "binding_id" in selected else {}),
                    "question": need["question"],
                    "hypothesis_id": round_candidates[need["hypothesis_id"]],
                    "signal": selected["signal"],
                    "entity_ref": selected["entity_ref"],
                    "query_ref": response["query_ref"],
                    "status": response["status"],
                    "observation_refs": tuple(
                        item["observation_id"] for item in response["observations"]
                    ),
                })
        latest = self.inquiry_drafts[-1]
        return InquiryHandoff(
            revision=latest["round"],
            hypotheses=tuple(item.hypothesis for item in candidates),
            needs=latest["draft"]["needs"],
            query_links=links[:10],
            candidate_history=candidates,
        ).model_dump(mode="json")


def _history_observations(history: tuple[dict[str, Any], ...]) -> list[dict[str, Any]]:
    observations = []
    for record in history:
        result = record.get("result", {})
        value = result.get("value") if isinstance(result, dict) else None
        if not isinstance(value, dict):
            continue
        observations.extend(value.get("observations", ()))
    return observations


def _history_entities(context: dict[str, Any]) -> list[dict[str, Any]]:
    bootstrap = context["bootstrap"]
    ranked: dict[str, tuple[int, dict[str, Any]]] = {
        bootstrap["alert_entity"]["entity_ref"]: (0, bootstrap["alert_entity"]),
    }
    for observation in _history_observations(context.get("history", ())):
        if observation.get("signal") not in {"related_entity", "trace_participant"}:
            continue
        entity = observation.get("value")
        if not isinstance(entity, dict):
            continue
        details = observation.get("details", {})
        rank = 8
        if observation.get("signal") == "trace_participant":
            rank = int(details.get("critical_rank", 6))
        elif details.get("relation") == "operation_of":
            rank = 0
        elif details.get("relation") == "calls" and details.get("direction") == "outgoing":
            rank = 1
        elif details.get("relation") == "calls":
            rank = 2
        existing = ranked.get(entity["entity_ref"])
        if existing is None or rank < existing[0]:
            ranked[entity["entity_ref"]] = (rank, entity)
    return [item[1] for item in sorted(
        ranked.values(),
        key=lambda item: (item[0], item[1]["display_name"]),
    )]


def _normalize(value: str) -> str:
    return " ".join(value.casefold().split())
