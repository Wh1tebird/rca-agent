"""Run one post-freeze RCA100 Catalog-holdout experiment with real discovery."""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

from langchain_core.runnables import Runnable

from evaluation.ground_truth import FlatGroundTruthRepository
from rca_agent.core.runtime_registry import rca100_evidence_registry
from rca_agent.deepseek_policy import create_deepseek_chat_model, create_deepseek_discovery_pipeline
from rca_agent.environment import Rca100Environment
from rca_agent.langchain_acquisition import (
    InquiryHandoff,
    InquiryOutputError,
    LangChainAcquisitionSelector,
    LangChainInquiryAdapter,
    inquiry_candidate_records,
    query_options,
)
from rca_agent.langchain_policy import LangChainPolicyAdapter, PolicyEnvelope
from rca_agent.langgraph_orchestrator import LangGraphInvestigation
from rca_agent.local_model import DEFAULT_LOCAL_MODEL, create_local_chat_model
from rca_agent.models import (
    CallToolAction,
    GetMetricSeriesRequest,
    GetServiceTraceSummaryRequest,
    GetTraceStageBreakdownRequest,
    HypothesisProposal,
    ListRelatedEntitiesRequest,
    RequestStopAction,
    SubmitHypothesisAction,
    ToolName,
)
from rca_agent.qwen_policy import create_qwen_discovery_pipeline
from rca_agent.runner import CoreDecisionProvider, EpisodeRunner, EventType, RecordedReplay


class HoldoutAcquisitionPolicy(Runnable[dict[str, Any], PolicyEnvelope]):
    """Acquire bounded trace context, then force a genuinely unmatched hypothesis."""

    SCREEN_SIGNALS = (
        "thread_pool_active",
        "thread_pool_rejected_tasks",
        "jvm_thread_cpu_time_ns",
        "jvm_gc_pause_seconds_delta",
        "jvm_heap_used_bytes",
    )

    def __init__(self, selector=None, inquiry=None, *, inquiry_query_limit: int = 10) -> None:
        if not 1 <= inquiry_query_limit <= 10:
            raise ValueError("inquiry query limit must be between one and ten")
        self.selector = selector
        self.inquiry = inquiry
        self.inquiry_query_limit = inquiry_query_limit
        self.selection_calls = 0
        self.selection_errors: list[str] = []
        self.selection_diagnostics: list[dict[str, Any]] = []
        self.selected_actions: list[CallToolAction] = []
        self.selection_queue: list[CallToolAction] = []
        self.inquiry_drafts: list[dict[str, Any]] = []
        self._step = 0
        self._tried_trace_refs: set[str] = set()
        self._summarized_entities: set[str] = set()
        self._expanded_entities: set[str] = set()
        self._acquisition_actions = 0
        self._pending_expand: str | None = None
        self._screened: set[tuple[str, str]] = set()
        self._screen_target_refs: tuple[str, ...] | None = None

    def invoke(self, input: dict[str, Any], config: Any = None, **kwargs: Any) -> PolicyEnvelope:
        del config, kwargs
        context = input["context"]
        if self._step == 0:
            self._step += 1
            self._acquisition_actions += 1
            bootstrap = context["bootstrap"]
            self._summarized_entities.add(bootstrap["alert_entity"]["entity_ref"])
            return PolicyEnvelope(action=CallToolAction(
                tool_name=ToolName.GET_SERVICE_TRACE_SUMMARY,
                request=GetServiceTraceSummaryRequest(
                    entity_ref=bootstrap["alert_entity"]["entity_ref"],
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
        trace_refs = [
            item.get("details", {}).get("trace_ref") for item in observations
            if item.get("signal") == "operation_trace_duration_ms"
        ]
        next_trace = next(
            (trace_ref for trace_ref in trace_refs
             if trace_ref and trace_ref not in self._tried_trace_refs),
            None,
        )
        participant_seen = any(
            item.get("signal") == "trace_participant" for item in observations
        )
        if not has_stage_context and not participant_seen and next_trace is not None:
            self._tried_trace_refs.add(next_trace)
            self._acquisition_actions += 1
            return PolicyEnvelope(action=CallToolAction(
                tool_name=ToolName.GET_TRACE_STAGE_BREAKDOWN,
                request=GetTraceStageBreakdownRequest(trace_ref=next_trace, max_stages=12),
            ))
        if participant_seen or has_stage_context:
            if self.selector is not None or self.inquiry is not None:
                action = (self._inquiry_screen(context) if self.inquiry is not None
                          else self._ai_screen(context))
                if action is not None:
                    return PolicyEnvelope(action=action)
                # Signal that acquisition has completed without entering the fixed screen.
                self._screen_target_refs = ("completed",)
            if self._screen_target_refs is None:
                self._screen_target_refs = tuple(
                    entity["entity_ref"]
                    for entity in _history_entities(context)
                    if entity["entity_type"] == "apm.service"
                )[:2]
            for entity_ref in self._screen_target_refs:
                if self.selector is not None or self.inquiry is not None:
                    break
                for signal in self.SCREEN_SIGNALS:
                    key = (entity_ref, signal)
                    if key in self._screened:
                        continue
                    self._screened.add(key)
                    self._acquisition_actions += 1
                    return PolicyEnvelope(action=CallToolAction(
                        tool_name=ToolName.GET_METRIC_SERIES,
                        request=GetMetricSeriesRequest(
                            entity_ref=entity_ref,
                            signal=signal,
                            window=context["bootstrap"]["observation_horizon"],
                            max_points=100,
                        ),
                    ))
        if (
            not has_stage_context
            and not self._screen_target_refs
            and self._acquisition_actions < 16
        ):
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
                return PolicyEnvelope(action=CallToolAction(
                    tool_name=ToolName.LIST_RELATED_ENTITIES,
                    request=ListRelatedEntitiesRequest(
                        entity_ref=pending["entity_ref"], max_results=50,
                    ),
                ))
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
                bootstrap = context["bootstrap"]
                return PolicyEnvelope(action=CallToolAction(
                    tool_name=ToolName.GET_SERVICE_TRACE_SUMMARY,
                    request=GetServiceTraceSummaryRequest(
                        entity_ref=service["entity_ref"],
                        window=bootstrap["observation_horizon"],
                        min_duration_ms=100,
                        max_traces=4,
                    ),
                ))
            expandable = next(
                (entity for entity in entities
                 if entity["entity_type"] in {"apm.operation", "apm.service"}
                 and entity["entity_ref"] not in self._expanded_entities),
                None,
            )
            if expandable is not None:
                self._expanded_entities.add(expandable["entity_ref"])
                self._acquisition_actions += 1
                return PolicyEnvelope(action=CallToolAction(
                    tool_name=ToolName.LIST_RELATED_ENTITIES,
                    request=ListRelatedEntitiesRequest(
                        entity_ref=expandable["entity_ref"], max_results=50,
                    ),
                ))
        if self._step == 1:
            self._step += 1
            bootstrap = context["bootstrap"]
            return PolicyEnvelope(action=SubmitHypothesisAction(
                proposal=HypothesisProposal(
                    hypothesis_id="h-catalog-holdout",
                    candidate_id="catalogHoldoutUnknown",
                    mechanism="unexplained operation latency",
                    causal_entity_ref=bootstrap["alert_entity"]["entity_ref"],
                    rationale=(
                        "The bounded slow-operation alert is not assigned to an available Catalog "
                        "mechanism. Discover causal mechanisms testable by the issued trace."
                    ),
                )
            ))
        return PolicyEnvelope(action=RequestStopAction(
            summary="The bounded holdout investigation has completed available verification."
        ))

    def _ai_screen(self, context):
        if not self.selection_queue and self.selection_calls < 2:
            self.selection_calls += 1
            options = query_options(context, _history_entities(context))
            used = {a.model_dump_json() for a in self.selected_actions}
            options = {ref: item for ref, item in options.items()
                       if item["action"].model_dump_json() not in used}
            try:
                self.selection_queue = list(self.selector.select(context, options))
            except Exception as exc:  # noqa: BLE001 - bounded provider failure, no secrets
                self.selection_errors.append(type(exc).__name__)
        if self.selection_queue:
            action = self.selection_queue.pop(0)
            self.selected_actions.append(action)
            self._acquisition_actions += 1
            return action
        return None

    def _inquiry_screen(self, context):
        if len(self.selected_actions) >= self.inquiry_query_limit:
            self.selection_queue.clear()
            return None
        if not self.selection_queue and self.selection_calls < 5:
            self.selection_calls += 1
            options = query_options(context, _history_entities(context)[:8])
            used = {a.model_dump_json() for a in self.selected_actions}
            # The inquiry plans baseline comparisons or entity exploration, not raw series dumps.
            options = {ref: item for ref, item in options.items()
                       if item["action"].model_dump_json() not in used
                       and item["action"].tool_name != ToolName.GET_METRIC_SERIES}
            observations = _history_observations(context.get("history", ()))
            compact = [{
                "signal": item.get("signal"),
                "entity_ref": item.get("entity_ref"),
                "data_status": item.get("data_status"),
                "value": str(item.get("value"))[:500],
            } for item in observations[-60:]]
            incident = {
                "alert": context["bootstrap"]["alert_summary"],
                "issued_entities": _history_entities(context)[:8],
                "recent_observations": compact,
                "previous_inquiry": {
                    "hypotheses": self.inquiry_drafts[-1]["draft"]["hypotheses"],
                    "needs": self.inquiry_drafts[-1]["draft"]["needs"],
                } if self.inquiry_drafts else None,
                "carried_candidates": [
                    {"hypothesis_id": item.hypothesis.hypothesis_id,
                     "mechanism": item.hypothesis.mechanism,
                     "alternative": item.hypothesis.alternative}
                    for item in inquiry_candidate_records(self.inquiry_drafts)
                ],
            }
            for attempt in range(2):
                try:
                    draft, actions = self.inquiry.plan(incident, options)
                    remaining = self.inquiry_query_limit - len(self.selected_actions)
                    self.selection_queue = list(actions[:remaining])
                    self.inquiry_drafts.append({
                        "round": self.selection_calls,
                        "attempt": attempt + 1,
                        "draft": draft.model_dump(mode="json"),
                        "selected_actions": [
                            {"need_id": choice.need_id,
                             "query_ref": choice.query_ref,
                             "signal": getattr(action.request, "signal", "related_entity"),
                             "entity_ref": options[choice.query_ref]["entity"]["entity_ref"],
                             "action": action.model_dump(mode="json")}
                            for choice, action in zip(
                                draft.choices, actions[:remaining], strict=False
                            )
                        ],
                    })
                    break
                except Exception as exc:  # noqa: BLE001 - bounded provider failure, no secrets
                    if isinstance(exc, InquiryOutputError):
                        self.selection_diagnostics.append({
                            "round": self.selection_calls, "attempt": attempt + 1,
                            **exc.diagnostic(),
                        })
                    # Only pass a host-authored schema reminder; never echo raw provider text.
                    safe_validation_errors = {
                        "duplicate inquiry identifiers",
                        "inquiry need has no hypothesis",
                        "invalid inquiry choices",
                        "inquiry choice is not offered",
                    }
                    code = (str(exc) if type(exc) is ValueError
                            and str(exc) in safe_validation_errors else type(exc).__name__)
                    self.selection_errors.append(code)
                    if getattr(exc, "status_code", None) is not None or attempt == 1:
                        break
                    incident["validation_error"] = (
                        "Return valid JSON in the requested schema. Choose unique offered "
                        "query_refs, each linked to one declared evidence need."
                    )
        if self.selection_queue:
            action = self.selection_queue.pop(0)
            self.selected_actions.append(action)
            self._acquisition_actions += 1
            return action
        return None

    def discovery_handoff(self, history) -> dict[str, Any] | None:
        if self.inquiry is None or not self.inquiry_drafts:
            return None
        candidates = inquiry_candidate_records(self.inquiry_drafts)
        candidate_by_key = {
            (" ".join(item.hypothesis.mechanism.casefold().split()),
             " ".join(item.hypothesis.alternative.casefold().split())):
            item.hypothesis.hypothesis_id for item in candidates
        }
        responses = {
            json.dumps(record["action"], sort_keys=True): record["result"]["value"]
            for record in history
            if record.get("result", {}).get("kind") == "tool_response"
        }
        links = []
        for round_record in self.inquiry_drafts:
            questions = {need["need_id"]: need
                         for need in round_record["draft"]["needs"]}
            round_candidates = {
                item["hypothesis_id"]: candidate_by_key[(
                    " ".join(item["mechanism"].casefold().split()),
                    " ".join(item["alternative"].casefold().split()),
                )]
                for item in round_record["draft"]["hypotheses"]
            }
            for selected in round_record["selected_actions"]:
                response = responses.get(json.dumps(selected["action"], sort_keys=True))
                if response is None:
                    continue
                need = questions[selected["need_id"]]
                links.append({
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


def _history_observations(history) -> list[dict[str, Any]]:
    observations = []
    for record in history:
        result = record.get("result", {})
        value = result.get("value", {}) if isinstance(result, dict) else {}
        observations.extend(value.get("observations", ()))
    return observations


def _history_entities(context: dict[str, Any]) -> list[dict[str, Any]]:
    bootstrap = context["bootstrap"]
    ranked: dict[str, tuple[int, dict[str, Any]]] = {
        bootstrap["alert_entity"]["entity_ref"]: (0, bootstrap["alert_entity"]),
    }
    for observation in _history_observations(context.get("history", ())):
        if observation.get("signal") not in {"related_entity", "trace_participant"} or not isinstance(
            observation.get("value"), dict
        ):
            continue
        entity = observation["value"]
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
        ranked.values(), key=lambda item: (item[0], item[1]["display_name"])
    )]


def run_case(args: argparse.Namespace) -> dict[str, Any]:
    provider = getattr(args, "provider", "deepseek")
    model_name = args.model or (DEFAULT_LOCAL_MODEL if provider == "qwen-local"
                                else "deepseek-v4-pro")
    thinking_enabled = provider == "qwen-local" or not args.disable_thinking

    def chat_model(*, max_retries: int = 0):
        if provider == "qwen-local":
            return create_local_chat_model(
                model=model_name, base_url=getattr(args, "base_url", None),
                timeout_seconds=args.model_timeout_seconds, max_retries=max_retries,
                allow_insecure_http=getattr(args, "allow_insecure_http", False),
            )
        return create_deepseek_chat_model(
            model=model_name, timeout_seconds=args.model_timeout_seconds,
            max_retries=max_retries, thinking_enabled=thinking_enabled,
        )

    environment = Rca100Environment.from_case(
        args.cases_root, args.case_id, query_budget=args.query_budget,
    )
    if provider == "qwen-local":
        discovery, verification_planner = create_qwen_discovery_pipeline(
            model=model_name,
            base_url=getattr(args, "base_url", None),
            timeout_seconds=args.model_timeout_seconds,
            allow_insecure_http=getattr(args, "allow_insecure_http", False),
        )
    else:
        discovery, verification_planner = create_deepseek_discovery_pipeline(
            model=model_name,
            timeout_seconds=args.model_timeout_seconds,
            thinking_enabled=thinking_enabled,
        )
    runner = EpisodeRunner(
        environment,
        max_rounds=args.max_rounds,
        decision_provider=CoreDecisionProvider(frozenset({args.holdout_candidate})),
        discovery=discovery,
        verification_planner=verification_planner,
        evidence_registry=rca100_evidence_registry,
        discovery_timeout_seconds=args.discovery_timeout_seconds,
    )
    selector = None
    inquiry = None
    if getattr(args, "acquisition", "fixed") == "ai":
        selector = LangChainAcquisitionSelector.from_chat_model(
            chat_model(), structured_method="json_schema" if provider == "qwen-local"
            else "json_mode",
        )
    elif getattr(args, "acquisition", "fixed") == "inquiry":
        inquiry = LangChainInquiryAdapter.from_chat_model(
            chat_model(), structured_method="json_schema" if provider == "qwen-local"
            else "json_mode",
        )
    acquisition = HoldoutAcquisitionPolicy(
        selector, inquiry,
        inquiry_query_limit=getattr(args, "inquiry_query_limit", 10),
    )
    policy = LangChainPolicyAdapter(acquisition)
    report = LangGraphInvestigation(
        runner, policy,
        discovery_notes_provider=acquisition.discovery_handoff if inquiry is not None else None,
        inquiry_rounds_provider=(lambda: tuple(acquisition.inquiry_drafts))
        if inquiry is not None else None,
        enable_core_feedback=inquiry is not None,
    ).run(recursion_limit=args.recursion_limit)
    replay = RecordedReplay.verify(runner.transcript.events)

    # Ground truth is loaded only after AnswerArtifact freeze and replay verification.
    truth = FlatGroundTruthRepository(args.answer_key_root).load(args.case_id)
    decision = report.answer.authoritative_decision
    winner = next(
        (p for p in report.answer.policy_proposals
         if p.hypothesis_id == decision.winning_hypothesis_id),
        None,
    )
    supported_candidates = [
        proposal.model_dump(mode="json")
        for proposal in report.answer.policy_proposals
        if proposal.hypothesis_id in decision.supported_hypothesis_ids
    ]
    discovery = [event.payload for event in runner.transcript.events
                 if event.event_type == EventType.DISCOVERY_COMPLETED]
    discovery_started = [event.payload for event in runner.transcript.events
                         if event.event_type == EventType.DISCOVERY_STARTED]
    registered = [event.payload for event in runner.transcript.events
                  if event.event_type == EventType.DYNAMIC_CANDIDATE_REGISTERED]
    verification = [event.payload for event in runner.transcript.events
                    if event.event_type == EventType.DYNAMIC_VERIFICATION]
    draft_evaluations = [event.payload for event in runner.transcript.events
                         if event.event_type == EventType.DISCOVERY_DRAFT_EVALUATED]
    rejected_payloads = [event.payload for event in runner.transcript.events
                         if event.event_type == EventType.DISCOVERY_PAYLOAD_REJECTED]
    mechanism_generated = [event.payload for event in runner.transcript.events
                           if event.event_type == EventType.MECHANISM_DRAFT_GENERATED]
    mechanism_frozen = [event.payload for event in runner.transcript.events
                        if event.event_type == EventType.MECHANISM_DRAFT_FROZEN]
    mechanism_rejected = [event.payload for event in runner.transcript.events
                          if event.event_type == EventType.MECHANISM_DRAFT_REJECTED]
    plan_generated = [event.payload for event in runner.transcript.events
                      if event.event_type == EventType.VERIFICATION_PLAN_GENERATED]
    plan_rejected = [event.payload for event in runner.transcript.events
                     if event.event_type == EventType.VERIFICATION_PLAN_REJECTED]
    plan_compiled = [event.payload for event in runner.transcript.events
                     if event.event_type == EventType.VERIFICATION_PLAN_COMPILED]
    frozen_by_index = {
        item["draft_index"]: item["mechanism"] for item in mechanism_frozen
    }
    mechanism_errors = {
        item["draft_index"]: item["error_code"] for item in mechanism_rejected
    }
    two_stage_drafts = []
    for item in mechanism_generated:
        frozen = frozen_by_index.get(item["draft_index"])
        mechanism_id = frozen.get("mechanism_id") if frozen else None
        first_rejection = next((event["error_code"] for event in plan_rejected
                                if event["mechanism_id"] == mechanism_id
                                and event["attempt"] == 1), None)
        final_rejection = next((event["error_code"] for event in reversed(plan_rejected)
                                if event["mechanism_id"] == mechanism_id), None)
        initial_compile = next((event for event in plan_compiled
                                if event["mechanism_id"] == mechanism_id
                                and event["attempt"] == 1), None)
        final_compile = next((event for event in plan_compiled
                              if event["mechanism_id"] == mechanism_id), None)
        mechanism_error = mechanism_errors.get(item["draft_index"])
        two_stage_drafts.append({
            "draft_index": item["draft_index"],
            "mechanism_id": mechanism_id,
            "raw_draft": item["draft"],
            "final_draft": item["draft"],
            "initial_status": "accepted" if initial_compile else "rejected",
            "initial_error": mechanism_error or first_rejection,
            "repair_actions": (),
            "final_status": "accepted" if final_compile else "rejected",
            "final_error": None if final_compile else mechanism_error or final_rejection,
            "registration": final_compile.get("registration", "none")
            if final_compile else "none",
        })
    invalid_drafts = []
    for rejection in rejected_payloads:
        raw_payload = rejection.get("raw_payload")
        hypotheses = (
            raw_payload.get("hypotheses", raw_payload.get("mechanisms", ()))
            if isinstance(raw_payload, dict) else ()
        )
        for draft_index, raw_draft in enumerate(hypotheses):
            if isinstance(raw_draft, dict) and isinstance(raw_draft.get("mechanism"), str):
                invalid_drafts.append({
                    "draft_index": draft_index,
                    "raw_draft": raw_draft,
                    "final_draft": raw_draft,
                    "initial_status": "rejected",
                    "initial_error": "invalid_discovery_payload",
                    "repair_actions": (),
                    "final_status": "rejected",
                    "final_error": "invalid_discovery_payload",
                    "registration": "none",
                })
    tool_responses = [event.payload for event in runner.transcript.events
                      if event.event_type == EventType.TOOL_RESPONSE]
    issued_services = set()
    missing_observations = 0
    for response in tool_responses:
        for observation in response.get("observations", ()):
            if observation.get("data_status") == "missing":
                missing_observations += 1
            value = observation.get("value")
            if isinstance(value, dict) and value.get("entity_type") == "apm.service":
                issued_services.add(value.get("display_name", ""))
            for key in ("resolved_service",):
                entity = observation.get("details", {}).get(key)
                if isinstance(entity, dict) and entity.get("entity_type") == "apm.service":
                    issued_services.add(entity.get("display_name", ""))
    target_services = {
        re.sub(r"-[a-f0-9]{9,10}-[a-z0-9]{5}$", "", entity)
        for entity in truth.root_cause_entities
    }
    return {
        "case_id": args.case_id,
        "acquisition_mode": getattr(args, "acquisition", "fixed"),
        "inquiry_query_limit": acquisition.inquiry_query_limit,
        "acquisition_selection_calls": acquisition.selection_calls,
        "acquisition_selection_errors": acquisition.selection_errors,
        "acquisition_selection_diagnostics": acquisition.selection_diagnostics,
        "acquisition_selected_actions": [a.model_dump(mode="json")
                                         for a in acquisition.selected_actions],
        "acquisition_inquiry_drafts": acquisition.inquiry_drafts,
        "transcript_inquiry_rounds": sum(
            event.event_type == EventType.INQUIRY_ROUND_RECORDED
            for event in runner.transcript.events
        ),
        "acquisition_handoff": (discovery_started[-1].get("inquiry_handoff")
                                if discovery_started else None),
        "inquiry_candidates": [item.model_dump(mode="json")
                               for item in report.answer.inquiry_candidates],
        "tool_responses": tool_responses,
        "registered_candidates": registered,
        "policy_proposals": [p.model_dump(mode="json") for p in report.answer.policy_proposals],
        "holdout_candidate": args.holdout_candidate,
        "provider": provider,
        "model": model_name,
        "thinking_enabled": thinking_enabled,
        "catalog_candidate_visible": args.holdout_candidate not in runner.excluded_catalog_candidate_ids,
        "status": decision.status.value,
        "supported_hypothesis_ids": list(decision.supported_hypothesis_ids),
        "supported_candidates": supported_candidates,
        "winner": winner.model_dump(mode="json") if winner else None,
        "dynamic_mechanisms": [item["compiled"]["candidate"]["fault_type"] for item in registered],
        "dynamic_plans": [{
            "mechanism": item["compiled"]["candidate"]["fault_type"],
            "alternatives": item["compiled"]["alternatives"],
            "predictions": item["compiled"]["predictions"],
        } for item in registered],
        "raw_discovery_drafts": [*draft_evaluations, *two_stage_drafts, *invalid_drafts],
        "raw_mechanism_drafts": mechanism_generated,
        "verification_plan_events": {
            "generated": plan_generated,
            "rejected": plan_rejected,
            "compiled": plan_compiled,
        },
        "discovery_payload_rejections": rejected_payloads,
        "verification": verification[-1] if verification else None,
        "discovery": discovery[-1] if discovery else None,
        "ground_truth": {
            "root_cause_types": truth.root_cause_types,
            "root_cause_entities": truth.root_cause_entities,
            "expected_fault_id": truth.expected_fault_id,
        },
        "graph_steps": report.graph_steps,
        "core_feedback_queries": [record for item in report.history
                                  if item.get("kind") == "core_feedback"
                                  for record in item["queries"]],
        "core_feedback_plans": [item["result"] for item in report.history
                                if item.get("kind") == "core_feedback_plan"],
        "core_feedback_plan_events": [
            {"kind": event.event_type.value, **event.payload}
            for event in runner.transcript.events
            if event.event_type in {
                EventType.VERIFICATION_PLAN_EXTENSION_STARTED,
                EventType.VERIFICATION_PLAN_EXTENSION_REJECTED,
                EventType.VERIFICATION_PLAN_EXTENDED,
            }
        ],
        "core_feedback_gaps": [{
            "candidate_id": item.candidate_id,
            "missing_signals": item.missing_signals,
            "unavailable_signals": item.unavailable_signals,
        } for item in report.final_preview.candidate_assessments
            if item.verification_pending and (
                item.missing_signals or item.unavailable_signals
            )],
        "tool_calls": replay.tool_calls,
        "diagnostics": {
            "tool_status_counts": dict(sorted(Counter(
                str(item.get("status")) for item in tool_responses
            ).items())),
            "missing_observations": missing_observations,
            "issued_services": sorted(issued_services),
            "target_service_issued": bool(target_services & issued_services),
            "discovery_rejections": discovery[-1].get("rejections", ()) if discovery else (),
            "verification_blockers": sorted({
                blocker
                for candidate in (verification[-1].get("candidates", {}).values()
                                  if verification else ())
                for blocker in candidate.get("blockers", ())
            }),
        },
        "replay_valid": replay.valid,
    }


def parser() -> argparse.ArgumentParser:
    default_dataset = Path.home() / "study/AgenticOpsEval/RCA100"
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--case-id", default="t025")
    value.add_argument("--acquisition", choices=("fixed", "ai", "inquiry"), default="fixed")
    value.add_argument("--inquiry-query-limit", type=int, default=10)
    value.add_argument("--holdout-candidate", default="dbNetworkLatency")
    value.add_argument("--cases-root", type=Path,
                       default=Path(os.getenv("RCA100_CASES_ROOT", default_dataset / "cases")))
    value.add_argument("--answer-key-root", type=Path,
                       default=Path(os.getenv("RCA100_ANSWER_KEY_ROOT",
                                               default_dataset / "answer_key")))
    value.add_argument("--provider", choices=("qwen-local", "deepseek"), default="qwen-local")
    value.add_argument("--model")
    value.add_argument("--base-url", help="OpenAI-compatible /v1 root; local provider only")
    value.add_argument("--allow-insecure-http", action="store_true")
    value.add_argument("--disable-thinking", action="store_true")
    value.add_argument("--query-budget", type=int, default=20)
    value.add_argument("--max-rounds", type=int, default=30)
    value.add_argument("--recursion-limit", type=int, default=100)
    value.add_argument("--model-timeout-seconds", type=float, default=45)
    value.add_argument("--discovery-timeout-seconds", type=float, default=50)
    value.add_argument("--output", type=Path)
    return value


def main() -> None:
    args = parser().parse_args()
    result = run_case(args)
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
