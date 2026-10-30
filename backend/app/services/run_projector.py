"""Project safe recorded facts into the existing investigation-console shape.

This module is intentionally a presenter. It never evaluates evidence, calculates a
Core status, or turns an Observation into a root-cause conclusion.
"""
from __future__ import annotations

import json
from typing import Any

from backend.app.schemas import (
    API_SCHEMA_VERSION,
    AuditRecordView,
    CandidateView,
    CaseDetail,
    EvidenceView,
    FinalDecisionView,
    InvestigationRunView,
    RecoveryMetadata,
    RecoveryStatus,
    ReplayMetadata,
    ReplayStatus,
    RunSummary,
    TimelineStepView,
)
from backend.app.services.run_store import StoredRun
from rca_agent.models import AnswerArtifact
from rca_agent.runner.recovery import RecoveryState
from rca_agent.runner.transcript import EventType, TranscriptEvent


class RunProjector:
    def project(self, stored: StoredRun, case: CaseDetail) -> InvestigationRunView:
        summary = _run_summary(stored)
        evidence, evidence_by_observation = _evidence(stored.events, case, stored.answer)
        candidates = _candidates(stored.answer, stored.events, evidence_by_observation)
        timeline = _timeline(stored.events, evidence_by_observation, candidates)
        final = _final_decision(stored.answer, stored)
        raw = {
            "schema_version": API_SCHEMA_VERSION,
            "run_id": summary.run_id,
            "episode_id": summary.episode_id,
            "replay_status": stored.replay_status.value,
            "recovery_status": stored.recovery.state.value,
            "answer_digest": summary.answer_digest,
            "stop_reason": summary.stop_reason,
            "transcript_event_count": len(stored.events),
            "replay_error": stored.replay_error,
        }
        return InvestigationRunView(
            schema_version=API_SCHEMA_VERSION,
            episode_id=summary.episode_id,
            answer_digest=summary.answer_digest,
            run=summary,
            case=case,
            candidates=tuple(candidates),
            timeline=tuple(timeline),
            evidence=tuple(evidence),
            final_decision=final,
            replay=_replay_metadata(stored),
            recovery=_recovery_metadata(stored),
            audit=tuple(_audit_view(item) for item in stored.audit),
            raw=_clean(raw),
        )


def _run_summary(stored: StoredRun) -> RunSummary:
    from backend.app.services.run_store import RunStore

    return RunStore._summary(stored)


def _evidence(
    events: tuple[TranscriptEvent, ...], case: CaseDetail, answer: AnswerArtifact | None,
) -> tuple[list[EvidenceView], dict[str, str]]:
    result: list[EvidenceView] = []
    index: dict[str, str] = {}
    admissions = _candidate_observation_refs(answer)
    for event in events:
        if event.event_type != EventType.TOOL_RESPONSE:
            continue
        payload = event.payload
        for raw_observation in payload.get("observations", ()):  # Agent-visible public facts only.
            observation_id = str(raw_observation.get("observation_id", ""))
            if not observation_id:
                continue
            evidence_id = f"ev-{observation_id}"
            index[observation_id] = evidence_id
            status = str(payload.get("status", "UNKNOWN")).upper()
            signal = str(raw_observation.get("signal", "observation"))
            candidate_ids = admissions.get(observation_id, ())
            observation_note = (
                "Neutral Observation returned by the host EvidenceProvider."
            )
            admission_context = (
                "Explicitly admitted by Candidate refs: " + ", ".join(candidate_ids)
                if candidate_ids else None
            )
            result.append(EvidenceView(
                id=evidence_id,
                title=_human_signal(signal),
                kind=str(raw_observation.get("modality", "query")),
                role="admission" if candidate_ids else "observation",
                value=_value_text(raw_observation.get("value")),
                unit=raw_observation.get("unit"),
                source=str(payload.get("tool_name", "RCA100 tool")),
                time_range=f"{case.window_start.isoformat()} — {case.window_end.isoformat()}",
                query_status=status,
                data_status=str(raw_observation.get("data_status", "UNKNOWN")).upper(),
                observation=(f"{signal}: {_value_text(raw_observation.get('value'))}"),
                observation_note=observation_note,
                admission_context=admission_context,
                interpretation=observation_note,
                candidate_ids=candidate_ids,
                query_ref=payload.get("query_ref"),
                observation_ref=observation_id,
                raw=_observation_projection(raw_observation),
            ))
    return result, index


def _candidate_observation_refs(
    answer: AnswerArtifact | None,
) -> dict[str, tuple[str, ...]]:
    bindings: dict[str, set[str]] = {}
    if answer is None:
        return {}
    for item in answer.inquiry_candidates:
        for observation_ref in item.observation_refs:
            bindings.setdefault(observation_ref, set()).add(item.hypothesis_id)
    for proposal in answer.policy_proposals:
        for observation_ref in proposal.observation_refs:
            bindings.setdefault(observation_ref, set()).add(proposal.hypothesis_id)
    return {ref: tuple(sorted(candidate_ids)) for ref, candidate_ids in bindings.items()}


def _candidates(
    answer: AnswerArtifact | None,
    events: tuple[TranscriptEvent, ...],
    evidence_by_observation: dict[str, str],
) -> list[CandidateView]:
    records: dict[str, dict[str, Any]] = {}

    frozen: dict[str, dict[str, Any]] = {}
    compiled: dict[str, str] = {}
    verification_states: dict[str, str] = {}
    for event in events:
        if event.event_type == EventType.MECHANISM_DRAFT_FROZEN:
            mechanism = event.payload.get("mechanism", {})
            mechanism_id = mechanism.get("mechanism_id")
            if isinstance(mechanism_id, str):
                frozen[mechanism_id] = mechanism
        elif event.event_type in {
            EventType.DYNAMIC_CANDIDATE_REGISTERED,
            EventType.VERIFICATION_PLAN_EXTENDED,
        }:
            mechanism_id = event.payload.get("mechanism_id")
            candidate = event.payload.get("compiled", {}).get("candidate", {})
            candidate_id = candidate.get("candidate_id")
            if isinstance(mechanism_id, str) and isinstance(candidate_id, str):
                compiled[mechanism_id] = candidate_id
        elif event.event_type == EventType.DYNAMIC_VERIFICATION:
            for candidate_id, verification in event.payload.get("candidates", {}).items():
                if isinstance(candidate_id, str) and isinstance(verification, dict):
                    verification_states[candidate_id] = str(
                        verification.get("state", "unresolved")
                    )

    mechanism_by_candidate = {
        candidate_id: mechanism_id for mechanism_id, candidate_id in compiled.items()
    }

    if answer is not None:
        for item in answer.inquiry_candidates:
            records[item.hypothesis_id] = {
                "id": item.hypothesis_id,
                "name": _short_name(item.mechanism),
                "mechanism": item.mechanism,
                "status": _inquiry_status(item.discovery_disposition, item.compiled_candidate_ids,
                                           item.core_evidence_statuses),
                "mentioned": True,
                "formalized": bool(item.formal_mechanism_ids),
                "compiled": bool(item.compiled_candidate_ids),
                "hypothesis_id": item.hypothesis_id,
                "mechanism_id": item.formal_mechanism_ids[0] if item.formal_mechanism_ids else None,
                "compiled_candidate_id": (item.compiled_candidate_ids[0]
                                          if item.compiled_candidate_ids else None),
                "summary": item.discovery_reason,
                "evidence_ids": tuple(evidence_by_observation[ref] for ref in item.observation_refs
                                        if ref in evidence_by_observation),
            }
        for proposal in answer.policy_proposals:
            if proposal.hypothesis_id not in records:
                mechanism_id = mechanism_by_candidate.get(proposal.candidate_id)
                dynamic_status = verification_states.get(proposal.candidate_id)
                records[proposal.hypothesis_id] = {
                    "id": proposal.hypothesis_id,
                    "name": _short_name(proposal.mechanism),
                    "mechanism": proposal.mechanism,
                    "status": (
                        _verification_status(dynamic_status)
                        if dynamic_status is not None
                        else _decision_status(answer.authoritative_decision.status.value)
                    ),
                    "mentioned": True,
                    "formalized": mechanism_id is not None,
                    "compiled": mechanism_id is not None,
                    "hypothesis_id": proposal.hypothesis_id,
                    "mechanism_id": mechanism_id,
                    "compiled_candidate_id": proposal.candidate_id,
                    "summary": proposal.rationale,
                    "evidence_ids": tuple(evidence_by_observation[ref] for ref in proposal.observation_refs
                                            if ref in evidence_by_observation),
                }

    for mechanism_id, mechanism in frozen.items():
        if any(item.get("mechanism_id") == mechanism_id for item in records.values()):
            continue
        mechanism_text = str(mechanism.get("mechanism", "Formalized mechanism"))
        records[mechanism_id] = {
            "id": mechanism_id,
            "name": _short_name(mechanism_text),
            "mechanism": mechanism_text,
            "status": _verification_status(
                verification_states.get(compiled.get(mechanism_id, ""), "unresolved")
            ),
            "mentioned": False,
            "formalized": True,
            "compiled": mechanism_id in compiled,
            "hypothesis_id": None,
            "mechanism_id": mechanism_id,
            "compiled_candidate_id": compiled.get(mechanism_id),
            "summary": "Formalized by Discovery; see the recorded verification events for evidence state.",
            "evidence_ids": (),
        }
    return [CandidateView(**item) for item in records.values()]


def _timeline(
    events: tuple[TranscriptEvent, ...], evidence_by_observation: dict[str, str],
    candidates: list[CandidateView],
) -> list[TimelineStepView]:
    candidate_ids = {item.id for item in candidates}
    result = []
    for event in events:
        descriptor = _EVENT_DISPLAY.get(event.event_type)
        if descriptor is None:
            continue
        actor, title, status = descriptor
        payload = event.payload
        evidence_id = None
        if event.event_type == EventType.TOOL_RESPONSE:
            observations = payload.get("observations", ())
            if observations:
                evidence_id = evidence_by_observation.get(observations[0].get("observation_id", ""))
        related = tuple(item for item in _event_candidate_refs(payload) if item in candidate_ids)
        result.append(TimelineStepView(
            id=f"tl-{event.sequence}",
            sequence=event.sequence,
            time=event.recorded_at,
            actor=actor,
            title=title,
            summary=_event_summary(event),
            status=status,
            query=_query_text(payload) if event.event_type == EventType.TOOL_REQUEST else None,
            observation=_observation_text(payload) if event.event_type == EventType.TOOL_RESPONSE else None,
            decision=(payload.get("answer", {}).get("authoritative_decision", {}).get("status")
                      if event.event_type == EventType.ANSWER_FROZEN else None),
            evidence_id=evidence_id,
            candidate_ids=related,
            developer=_developer_projection(payload),
        ))
    return result


def _final_decision(answer: AnswerArtifact | None, stored: StoredRun) -> FinalDecisionView | None:
    if (
        answer is None
        or stored.replay_status != ReplayStatus.VALID
        or stored.recovery.state != RecoveryState.FROZEN
    ):
        return None
    decision = answer.authoritative_decision
    missing = answer.missing_evidence
    alternatives = tuple(item.alternative for item in answer.inquiry_candidates if item.alternative)
    status = _decision_status(decision.status.value)
    why = (
        "Recorded Replay is valid, but the Core did not issue a verified root-cause decision."
        if status != "VERIFIED"
        else "The Core issued an authoritative verified decision from the recorded evidence."
    )
    return FinalDecisionView(
        status=status,
        explanation=decision.reason,
        supporting=tuple(f"Observation {ref}" for ref in answer.observation_refs),
        missing=missing,
        alternatives=alternatives,
        why_not_verified=why,
        authoritative=True,
        answer_frozen=True,
        answer_digest=answer.artifact_digest,
        replay_status=stored.replay_status,
    )


def _replay_metadata(stored: StoredRun) -> ReplayMetadata:
    report = stored.replay_report
    return ReplayMetadata(
        status=stored.replay_status,
        valid=stored.replay_status == ReplayStatus.VALID,
        event_count=len(stored.events),
        head_digest=(report.head_digest if report else (stored.events[-1].digest if stored.events else None)),
        answer_digest=(report.answer_digest if report else stored.recovery.answer_digest),
        error_code=stored.replay_error,
    )


def _recovery_metadata(stored: StoredRun) -> RecoveryMetadata:
    snapshot = stored.recovery
    return RecoveryMetadata(
        status=RecoveryStatus(snapshot.state.value),
        episode_id=snapshot.episode_id,
        event_count=snapshot.event_count,
        next_sequence=snapshot.next_sequence,
        head_digest=snapshot.head_digest,
        budget_remaining=snapshot.budget_remaining,
        answer_digest=snapshot.answer_digest,
        stop_reason=snapshot.stop_reason.value if snapshot.stop_reason else None,
        pending_operations=tuple(item.model_dump(mode="json") for item in snapshot.pending_operations),
        replay_valid=snapshot.replay_valid,
        failure_category=snapshot.failure_category,
        failure_kind=snapshot.failure_kind,
    )


def _audit_view(record: Any) -> AuditRecordView:
    return AuditRecordView.model_validate(record.model_dump(mode="json"))


_EVENT_DISPLAY = {
    EventType.ENVIRONMENT_RESET: ("SYSTEM", "告警进入", "neutral"),
    EventType.INQUIRY_ROUND_RECORDED: ("AI", "Inquiry 更新候选", "neutral"),
    EventType.TOOL_REQUEST: ("AI", "选择查询", "running"),
    EventType.TOOL_RESPONSE: ("RUNNER", "获得 Observation", "success"),
    EventType.MECHANISM_DRAFT_FROZEN: ("AI", "Discovery 固化 Mechanism", "success"),
    EventType.VERIFICATION_PLAN_GENERATED: ("AI", "生成验证计划", "neutral"),
    EventType.VERIFICATION_PLAN_COMPILED: ("RUNNER", "验证计划已编译", "success"),
    EventType.VERIFICATION_PLAN_EXTENDED: ("AI", "追加验证计划", "running"),
    EventType.DYNAMIC_VERIFICATION: ("CORE", "Core 重新评估证据", "warning"),
    EventType.ANSWER_FROZEN: ("CORE", "调查结论已冻结", "success"),
    EventType.POLICY_EXCEPTION: ("SYSTEM", "Policy 执行失败", "failed"),
    EventType.POLICY_TIMEOUT: ("SYSTEM", "Policy 调用超时", "failed"),
    EventType.TOOL_EXCEPTION: ("RUNNER", "查询执行失败", "failed"),
}


def _event_summary(event: TranscriptEvent) -> str:
    payload = event.payload
    if event.event_type == EventType.TOOL_RESPONSE:
        return str(payload.get("message", "Tool response recorded."))
    if event.event_type == EventType.INQUIRY_ROUND_RECORDED:
        hypotheses = payload.get("draft", {}).get("hypotheses", ())
        return f"Recorded {len(hypotheses)} tentative hypothesis/hypotheses before querying."
    if event.event_type == EventType.ANSWER_FROZEN:
        return "The Runner froze the answer after recomputing the authoritative Core decision."
    return event.event_type.value.replace("_", " ")


def _query_text(payload: dict[str, Any]) -> str | None:
    action = payload.get("action") or payload
    try:
        return json.dumps(action, ensure_ascii=False, sort_keys=True)
    except TypeError:
        return None


def _observation_text(payload: dict[str, Any]) -> str | None:
    observations = payload.get("observations", ())
    if not observations:
        return payload.get("message")
    first = observations[0]
    return f"{first.get('signal', 'observation')}: {_value_text(first.get('value'))}"


def _event_candidate_refs(payload: dict[str, Any]) -> tuple[str, ...]:
    values = []
    for key in ("hypothesis_id", "candidate_id", "mechanism_id"):
        value = payload.get(key)
        if isinstance(value, str):
            values.append(value)
    return tuple(values)


def _inquiry_status(disposition: str, compiled: tuple[str, ...], statuses: tuple[str, ...]) -> str:
    if statuses:
        return _decision_status(statuses[-1])
    if compiled:
        return "UNRESOLVED"
    if disposition == "discarded":
        return "NOT_PROPOSED"
    return "UNRESOLVED"


def _decision_status(value: str) -> str:
    normalized = value.lower()
    if normalized in {"confirmed", "confirmed_root_cause", "discovered_verified"}:
        return "VERIFIED"
    if normalized in {"supported", "supported_hypothesis", "multiple_supported"}:
        return "SUPPORTED"
    if normalized == "contradicted":
        return "FAILED"
    if normalized == "out_of_scope":
        return "OUT_OF_SCOPE"
    if normalized == "query_failed":
        return "QUERY_FAILED"
    if normalized == "inconclusive":
        return "INCONCLUSIVE"
    if normalized in {"unknown", "missing"}:
        return normalized.upper()
    return "UNRESOLVED"


def _verification_status(value: str) -> str:
    normalized = value.casefold()
    if normalized in {"verified", "passed"}:
        return "VERIFIED"
    if normalized == "failed":
        return "FAILED"
    if normalized == "supported":
        return "SUPPORTED"
    if normalized == "out_of_scope":
        return "OUT_OF_SCOPE"
    if normalized == "query_failed":
        return "QUERY_FAILED"
    if normalized == "inconclusive":
        return "INCONCLUSIVE"
    if normalized in {"unknown", "missing"}:
        return normalized.upper()
    return "UNRESOLVED"


def _short_name(mechanism: str) -> str:
    return mechanism[:80] if mechanism else "Unspecified mechanism"


def _human_signal(signal: str) -> str:
    return signal.replace("_", " ").title()


def _value_text(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)[:800]
    return str(value)


_PRIVATE_KEYS = {
    "ground_truth", "answer_key", "cases_root", "workspace", "region_id",
    "path", "filesystem", "secret", "token", "prompt", "trace_id",
}

_OBSERVATION_KEYS = {
    "observation_id", "modality", "entity_ref", "signal", "value", "unit",
    "data_status", "observed_at", "details",
}
_DETAIL_KEYS = {
    "trace_ref", "stage", "duration_ms", "ratio", "sample_count", "status",
    "baseline", "incident_max", "baseline_mean", "baseline_ratio", "incident_delta",
    "baseline_delta", "connection_acquisition_ms", "trust", "data_status",
}
_DEVELOPER_KEYS = {
    "query_ref", "observation_ref", "capability_ref", "evidence_capability",
    "evidence_provider", "evidence_scope_digest", "evidence_time_range",
    "evidence_cost_units", "status", "reason", "termination_reason", "stop_reason",
    "failure_category", "failure_kind", "error_type", "candidate_id", "hypothesis_id",
    "mechanism_id", "plan_version", "core_revision", "replay", "query_refs",
    "operation_status", "operation_kind", "operation_id", "execution_source",
}


def _clean(value: Any) -> Any:
    """Defence in depth: presentation responses never disclose host/evaluator fields."""
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()
                if str(key).lower() not in _PRIVATE_KEYS}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    return value


def _observation_projection(value: dict[str, Any]) -> dict[str, Any]:
    projection = {
        key: _safe_value(value[key])
        for key in _OBSERVATION_KEYS
        if key in value and key != "details"
    }
    details = value.get("details")
    if isinstance(details, dict):
        safe_details = {
            key: _safe_value(details[key])
            for key in _DETAIL_KEYS
            if key in details
        }
        if safe_details:
            projection["details"] = safe_details
    return projection


def _developer_projection(value: dict[str, Any]) -> dict[str, Any]:
    projection: dict[str, Any] = {}
    for key in _DEVELOPER_KEYS:
        if key not in value:
            continue
        item = value[key]
        if key == "authoritative_decision":
            continue
        projection[key] = _safe_value(item)
    decision = value.get("authoritative_decision")
    if isinstance(decision, dict):
        projection["authoritative_decision"] = {
            key: decision[key]
            for key in ("status", "source")
            if key in decision
        }
    return projection


def _safe_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _safe_value(item)
            for key, item in value.items()
            if str(key).casefold() not in _PRIVATE_KEYS
        }
    if isinstance(value, (list, tuple)):
        return [_safe_value(item) for item in value[:32]]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)[:200]
