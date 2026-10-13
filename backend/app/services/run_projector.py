"""Project safe recorded facts into the existing investigation-console shape.

This module is intentionally a presenter. It never evaluates evidence, calculates a
Core status, or turns an Observation into a root-cause conclusion.
"""
from __future__ import annotations

import json
from typing import Any

from backend.app.schemas import (
    CandidateView,
    CaseDetail,
    EvidenceView,
    FinalDecisionView,
    InvestigationRunView,
    ReplayStatus,
    RunSummary,
    TimelineStepView,
)
from backend.app.services.run_store import StoredRun
from rca_agent.models import AnswerArtifact
from rca_agent.runner.transcript import EventType, TranscriptEvent


class RunProjector:
    def project(self, stored: StoredRun, case: CaseDetail) -> InvestigationRunView:
        summary = _run_summary(stored)
        evidence, evidence_by_observation = _evidence(stored.events, case)
        candidates = _candidates(stored.answer, stored.events, evidence_by_observation)
        timeline = _timeline(stored.events, evidence_by_observation, candidates)
        final = _final_decision(stored.answer, stored.replay_status)
        raw = {
            "run_id": summary.run_id,
            "episode_id": stored.answer.episode_id if stored.answer else None,
            "replay_status": stored.replay_status.value,
            "replay_error": stored.replay_error,
            "transcript_event_count": len(stored.events),
            "answer_artifact_digest": stored.answer.artifact_digest if stored.answer else None,
        }
        return InvestigationRunView(
            run=summary,
            case=case,
            candidates=tuple(candidates),
            timeline=tuple(timeline),
            evidence=tuple(evidence),
            final_decision=final,
            raw=_clean(raw),
        )


def _run_summary(stored: StoredRun) -> RunSummary:
    from backend.app.services.run_store import RunStore

    return RunStore._summary(stored)


def _evidence(
    events: tuple[TranscriptEvent, ...], case: CaseDetail
) -> tuple[list[EvidenceView], dict[str, str]]:
    result: list[EvidenceView] = []
    index: dict[str, str] = {}
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
            status = str(payload.get("status", "unknown")).upper()
            signal = str(raw_observation.get("signal", "observation"))
            result.append(EvidenceView(
                id=evidence_id,
                title=_human_signal(signal),
                kind=str(raw_observation.get("modality", "query")),
                role="context",
                value=_value_text(raw_observation.get("value")),
                unit=raw_observation.get("unit"),
                source=str(payload.get("tool_name", "RCA100 tool")),
                time_range=f"{case.window_start.isoformat()} — {case.window_end.isoformat()}",
                query_status=status,
                observation=(f"{signal}: {_value_text(raw_observation.get('value'))}"),
                interpretation="Recorded observation. Its evidence role is assigned only by a recorded verification plan.",
                query_ref=payload.get("query_ref"),
                observation_ref=observation_id,
                raw=_clean(raw_observation),
            ))
    return result, index


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
            developer=_clean(payload),
        ))
    return result


def _final_decision(
    answer: AnswerArtifact | None, replay_status: ReplayStatus
) -> FinalDecisionView | None:
    if answer is None or replay_status != ReplayStatus.VALID:
        return None
    decision = answer.authoritative_decision
    missing = answer.missing_evidence
    alternatives = tuple(item.alternative for item in answer.inquiry_candidates if item.alternative)
    status = decision.status.value.upper()
    why = (
        "Recorded Replay is valid, but the Core did not issue a verified root-cause decision."
        if status not in {"CONFIRMED_ROOT_CAUSE", "DISCOVERED_VERIFIED"}
        else "The Core issued an authoritative verified decision from the recorded evidence."
    )
    return FinalDecisionView(
        status=status,
        explanation=decision.reason,
        supporting=tuple(
            proposal.rationale for proposal in answer.policy_proposals
            if proposal.hypothesis_id in decision.supported_hypothesis_ids
        ),
        missing=missing,
        alternatives=alternatives,
        why_not_verified=why,
    )


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
    if normalized == "supported":
        return "SUPPORTED"
    if normalized == "contradicted":
        return "FAILED"
    if normalized in {"inconclusive", "out_of_scope", "query_failed"}:
        return "INCONCLUSIVE"
    return "UNRESOLVED"


def _verification_status(value: str) -> str:
    normalized = value.casefold()
    if normalized in {"verified", "passed"}:
        return "VERIFIED"
    if normalized == "failed":
        return "FAILED"
    if normalized == "supported":
        return "SUPPORTED"
    if normalized == "inconclusive":
        return "INCONCLUSIVE"
    return "UNRESOLVED"


def _short_name(mechanism: str) -> str:
    return mechanism[:80] if mechanism else "Unspecified mechanism"


def _human_signal(signal: str) -> str:
    return signal.replace("_", " ").title()


def _value_text(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)[:800]
    return str(value)


_PRIVATE_KEYS = {"ground_truth", "answer_key", "cases_root", "workspace", "region_id"}


def _clean(value: Any) -> Any:
    """Defence in depth: presentation responses never disclose host/evaluator fields."""
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()
                if str(key).lower() not in _PRIVATE_KEYS}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    return value
