from __future__ import annotations

from collections.abc import Iterable

from pydantic import TypeAdapter

from evaluation.models import (
    ContractEvaluation,
    EfficiencyEvaluation,
    EvaluationOutcome,
    EvaluationProvenance,
    EvaluationReport,
    EvidenceEvaluation,
    FailureCategory,
    FailureTaxonomyEntry,
    OutcomeEvaluation,
    ReplayEvaluation,
)
from rca_agent.models import DataStatus, EpisodeStatus, StopReason, ToolStatus
from rca_agent.runner import EpisodeRunner, ReplayReport
from rca_agent.runner.transcript import EventType, TranscriptEvent
from rca_agent.tools.fingerprint import sha256_digest


def failure_taxonomy(
    events: Iterable[TranscriptEvent],
    replay: ReplayReport,
) -> tuple[FailureTaxonomyEntry, ...]:
    entries: list[FailureTaxonomyEntry] = []
    for event in events:
        kind = _failure_for_event(event)
        if kind is None:
            continue
        category, failure_kind, phase, severity, retryable, user_visible, hint = kind
        entries.append(FailureTaxonomyEntry(
            category=category,
            failure_kind=failure_kind,
            phase=phase,
            triggering_event_id=event.event_id,
            severity=severity,
            retryable=retryable,
            user_visible=user_visible,
            evidence=(event.event_type.value,),
            remediation_hint=hint,
        ))
    if not replay.valid:
        entries.append(FailureTaxonomyEntry(
            category=FailureCategory.REPLAY,
            failure_kind="replay_validation_failure",
            phase="replay",
            severity="error",
            retryable=False,
            user_visible=True,
            remediation_hint="Inspect the first divergent event and its digest chain.",
        ))
    return tuple(entries)


def build_evaluation_report(
    runner: EpisodeRunner,
    replay: ReplayReport,
    *,
    evaluator_version: str,
    evaluator_record_digest: str,
    outcome: EvaluationOutcome = EvaluationOutcome.NOT_EVALUABLE,
    outcome_detail: str = "ground_truth_not_loaded",
) -> EvaluationReport:
    answer = runner.answer
    if runner.status not in {EpisodeStatus.ANSWER_FROZEN, EpisodeStatus.EVALUATED} or answer is None:
        raise ValueError("EvaluationReport requires a frozen AnswerArtifact")
    events = runner.transcript.events
    observations = {
        item["observation_id"]: item
        for event in events
        if event.event_type == EventType.TOOL_RESPONSE
        for item in event.payload.get("observations", ())
    }
    referenced = set(answer.observation_refs)
    invalid_refs = referenced - observations.keys()
    proposal_refs = [set(proposal.observation_refs) for proposal in answer.policy_proposals]
    explicitly_admitted = set().union(*proposal_refs) if proposal_refs else set()
    unowned_refs = referenced - explicitly_admitted
    ineligible_statuses = {
        DataStatus.UNKNOWN.value,
        DataStatus.MISSING.value,
        DataStatus.IMPUTED.value,
        DataStatus.MODEL_INFERRED.value,
    }
    ineligible_positive = sum(
        observations[ref].get("data_status") in ineligible_statuses
        for ref in referenced & observations.keys()
    )
    evidence_result = "supported" if referenced and not invalid_refs else (
        "insufficient" if referenced else "not_evaluable"
    )
    invalid_queries = sum(
        event.payload.get("status") in {
            ToolStatus.REJECTED.value,
            ToolStatus.QUERY_FAILED.value,
        }
        for event in events
        if event.event_type == EventType.TOOL_RESPONSE
    )
    query_refs = [
        event.payload.get("query_ref")
        for event in events
        if event.event_type == EventType.TOOL_REQUEST
    ]
    budget_consumed = sum(
        event.payload.get("units", 0)
        for event in events
        if event.event_type == EventType.QUERY_COST
    )
    stop_quality = {
        StopReason.BUDGET_EXHAUSTED: "budget_exhausted",
        StopReason.TOOL_FAILURE: "failure",
        StopReason.POLICY_REQUEST: "manual",
        StopReason.MAX_ROUNDS: "clean",
    }.get(answer.termination_reason, "unknown")
    failures = failure_taxonomy(events, replay)
    contract_failures = tuple(
        item for item in failures
        if item.category in {FailureCategory.CONTRACT, FailureCategory.PROTOCOL, FailureCategory.STATE}
    )
    fields = {
        "episode_id": runner.episode_id,
        "answer_artifact_digest": answer.artifact_digest,
        "outcome": OutcomeEvaluation(result=outcome, detail_code=outcome_detail),
        "evidence": EvidenceEvaluation(
            result=evidence_result,
            valid_observation_count=len(referenced & observations.keys()),
            unissued_observation_count=len(invalid_refs),
            explicit_admission_count=len(referenced & explicitly_admitted),
            cross_candidate_leak_count=len(unowned_refs),
            ineligible_positive_count=ineligible_positive,
            same_episode=answer.episode_id == runner.episode_id,
        ),
        "efficiency": EfficiencyEvaluation(
            query_count=replay.tool_calls,
            budget_consumed=budget_consumed,
            invalid_query_count=invalid_queries,
            rejected_or_failed_query_count=invalid_queries,
            duplicate_query_count=len(query_refs) - len(set(query_refs)),
            stop_quality=stop_quality,
        ),
        "contract": ContractEvaluation(
            compliant=not contract_failures,
            violation_count=len(contract_failures),
        ),
        "replay": ReplayEvaluation(
            valid=replay.valid,
            head_digest=replay.head_digest,
            answer_digest=replay.answer_digest,
            workspace_digest=replay.workspace_digest,
            candidate_frontier_digest=replay.candidate_frontier_digest,
            stop_reason=replay.stop_reason.value if replay.stop_reason is not None else None,
            budget_remaining=replay.budget_remaining,
        ),
        "failure_taxonomy": failures,
        "provenance": EvaluationProvenance(
            evaluator_version=evaluator_version,
            record_digest=evaluator_record_digest,
        ),
    }
    return EvaluationReport(
        **fields,
        report_digest=sha256_digest(
            TypeAdapter(dict[str, object]).dump_python(fields, mode="json")
        ),
    )


def _failure_for_event(event: TranscriptEvent):
    explicit_category = event.payload.get("failure_category")
    if explicit_category in {item.value for item in FailureCategory}:
        return (
            FailureCategory(explicit_category),
            event.payload.get("failure_kind", "classified_failure"),
            event.payload.get("phase", "runtime"),
            event.payload.get("severity", "error"),
            bool(event.payload.get("retryable", False)),
            bool(event.payload.get("user_visible", True)),
            event.payload.get("remediation_hint", "Inspect the triggering event and preserve Runner authority."),
        )
    if event.event_type == EventType.TOOL_EXCEPTION and any(
        token in str(event.payload.get("error_type", "")).casefold()
        for token in ("worker", "factory", "adapter")
    ):
        return (
            FailureCategory.ADAPTER_ORCHESTRATION,
            "adapter_orchestration_failure",
            "adapter",
            "error",
            True,
            True,
            "Compare the worker-backed and legacy adapter contract before retrying.",
        )
    if event.event_type == EventType.TOOL_RESPONSE:
        status = event.payload.get("status")
        if status in {ToolStatus.QUERY_FAILED.value, ToolStatus.REJECTED.value}:
            category = (FailureCategory.EXECUTION
                        if status == ToolStatus.QUERY_FAILED.value
                        else FailureCategory.CONTRACT)
            return (
                category,
                "tool_failure" if status == ToolStatus.QUERY_FAILED.value else "query_rejected",
                "tool",
                "error" if status == ToolStatus.QUERY_FAILED.value else "warning",
                True,
                True,
                "Use the Runner capability and budget boundary before retrying.",
            )
    if event.event_type == EventType.STOP_REQUESTED:
        reason = event.payload.get("reason")
        if reason == StopReason.BUDGET_EXHAUSTED.value:
            return (
                FailureCategory.INVESTIGATION,
                "query_budget_exhaustion",
                "investigation",
                "info",
                False,
                True,
                "Stop and hand off with the bounded evidence already issued.",
            )
    if event.event_type == EventType.CANDIDATE_EVIDENCE_UPDATED:
        return (
            FailureCategory.GROUNDING,
            "candidate_evidence_update",
            "core_admission",
            "info",
            True,
            False,
            "Keep evidence references candidate-scoped and episode-issued.",
        )
    mapping = {
        EventType.POLICY_TIMEOUT: (FailureCategory.EXECUTION, "policy_timeout", "policy", "error", True, True,
                                   "Retry with a bounded policy invocation."),
        EventType.POLICY_EXCEPTION: (FailureCategory.EXECUTION, "policy_exception", "policy", "error", True, True,
                                     "Inspect the sanitized policy failure and retry if allowed."),
        EventType.TOOL_EXCEPTION: (FailureCategory.EXECUTION, "tool_failure", "tool", "error", True, True,
                                   "Retry only through the Runner budget and admission path."),
        EventType.ACTION_REJECTED: (FailureCategory.CONTRACT, "action_rejected", "policy", "warning", True, True,
                                    "Repair the typed action before resubmission."),
        EventType.DISCOVERY_PAYLOAD_REJECTED: (FailureCategory.PROTOCOL, "schema_violation", "discovery", "warning", True, True,
                                               "Return only the advertised typed schema."),
        EventType.MECHANISM_FORMALIZATION_REJECTED: (FailureCategory.PROTOCOL, "formalizer_contract_violation", "formalizer", "warning", True, True,
                                                    "Preserve candidate identity and formalization semantics."),
        EventType.VERIFICATION_PLAN_REJECTED: (FailureCategory.PROTOCOL, "planner_contract_violation", "planner", "warning", True, True,
                                               "Use only host-advertised capabilities and frozen mechanisms."),
        EventType.INQUIRY_QUERY_CANCELLED: (FailureCategory.INVESTIGATION, "query_cancelled", "inquiry", "info", True, True,
                                            "Recompute the next bounded query from the current host view."),
    }
    return mapping.get(event.event_type)
