from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from rca_agent.core.dynamic import (
    CompiledCandidate,
    DynamicVerification,
    assess_dynamic,
    verify_dynamic,
)
from rca_agent.core.evidence_control import DecisionStatus as CoreDecisionStatus
from rca_agent.core.evidence_control import EvidenceController, FamilyRoute, RouteStatus
from rca_agent.core.v3.catalog import mechanism_catalog
from rca_agent.core.v3.engine import OpenRcaEngine
from rca_agent.core.v3.models import Candidate, ConditionPolarity
from rca_agent.core.v3.models import Observation as CoreObservation
from rca_agent.models import (
    AuthoritativeDecision,
    CandidateAssessmentPreview,
    CandidateEvidenceStatus,
    CapabilityDescriptor,
    CoreAssessmentPreview,
    DataStatus,
    DecisionSource,
    DecisionStatus,
    DiscriminatingSignalView,
    HypothesisProposal,
    Observation,
    StopReason,
)

TRUSTED_DATA_STATUSES = frozenset({DataStatus.OBSERVED, DataStatus.DERIVED})
WINDOW_RATIO_SIGNALS = {
    "request_count_window_comparison": "service_workload_ratio",
    "request_latency_seconds_window_comparison": "service_latency_ratio",
    "error_count_window_comparison": "service_error_ratio",
    "error_rate_window_comparison": "service_error_ratio",
    "cpu_usage_window_comparison": "service_cpu_ratio",
}
DIRECT_CORE_SIGNALS = {
    "node_cpu_peak",
    "node_cpu_high_fraction",
    "node_memory_peak",
    "node_disk_peak",
    "network_stage_ms",
    "network_error_ratio",
    "service_error_ratio",
    "service_latency_ratio",
    "service_workload_ratio",
    "service_cpu_ratio",
    "application_error_marker",
    "rate_limit_marker",
    "sql_duration_share",
    "sql_shape",
    "connection_duration_ms",
    "dependency_error_marker",
}
CAPABILITY_TO_TOOL = {
    "node_resource_metrics": "get_metric_series",
    "trace_stage_decomposition": "get_trace_stage_breakdown",
    "network_metrics": "get_metric_series",
    "service_metrics": "compare_metric_windows",
    "bounded_log_markers": "search_log_markers",
}


@dataclass(frozen=True, slots=True)
class IssuedObservation:
    observation: Observation
    query_ref: str
    complete: bool = True


@dataclass(frozen=True, slots=True)
class _CoreEvaluation:
    candidates: tuple[Candidate, ...]
    proposals_by_candidate: dict[str, tuple[HypothesisProposal, ...]]
    assessments_by_candidate: dict[str, tuple[Any, ...]]
    records_by_candidate: dict[str, tuple[IssuedObservation, ...]]
    decision: Any
    unresolved: tuple[str, ...]
    unavailable: tuple[str, ...]
    dynamic_verifications: dict[str, DynamicVerification]


class CoreDecisionProvider:
    """Translate issued Agent facts into the internal evidence-controlled decision kernel."""

    def __init__(self, excluded_catalog_candidate_ids: frozenset[str] = frozenset()) -> None:
        self._dynamic: dict[str, CompiledCandidate] = {}
        self._excluded_catalog_candidate_ids = frozenset(excluded_catalog_candidate_ids)

    @property
    def excluded_catalog_candidate_ids(self) -> frozenset[str]:
        return self._excluded_catalog_candidate_ids

    def catalog(self, entity_ref: str) -> tuple[Candidate, ...]:
        return tuple(
            candidate for candidate in mechanism_catalog(entity_ref)
            if candidate.candidate_id not in self._excluded_catalog_candidate_ids
        )

    def register_dynamic(self, compiled: CompiledCandidate) -> None:
        key = compiled.candidate.candidate_id
        if key in self._dynamic:
            raise ValueError("dynamic candidate already registered")
        self._dynamic[key] = compiled.model_copy(deep=True)

    @property
    def has_dynamic_candidates(self) -> bool:
        return bool(self._dynamic)

    def verification_reports(self, proposals, observations) -> dict[str, DynamicVerification]:
        evaluation = _evaluate(
            proposals, observations, self._dynamic, self._excluded_catalog_candidate_ids,
        )
        return evaluation.dynamic_verifications if evaluation else {}

    def preview(
        self,
        proposals: tuple[HypothesisProposal, ...],
        observations: tuple[IssuedObservation, ...],
        capabilities: tuple[CapabilityDescriptor, ...] | None = None,
    ) -> CoreAssessmentPreview:
        """Assess current evidence without freezing or mutating an episode."""

        overall = self.decide(proposals, observations, StopReason.POLICY_REQUEST)
        if not proposals:
            return CoreAssessmentPreview(overall_decision=overall)
        evaluation = _evaluate(
            proposals, observations, self._dynamic, self._excluded_catalog_candidate_ids,
        )
        if evaluation is None:
            return CoreAssessmentPreview(overall_decision=overall)
        candidate_assessments = _candidate_previews(evaluation)
        return CoreAssessmentPreview(
            overall_decision=overall,
            candidate_assessments=candidate_assessments,
            discriminating_signals=_discriminating_signals(
                evaluation,
                candidate_assessments,
                observations,
                capabilities,
            ),
        )

    def decide(
        self,
        proposals: tuple[HypothesisProposal, ...],
        observations: tuple[IssuedObservation, ...],
        termination_reason: StopReason,
    ) -> AuthoritativeDecision:
        if termination_reason == StopReason.TOOL_FAILURE:
            return AuthoritativeDecision(
                status=DecisionStatus.QUERY_FAILED,
                source=DecisionSource.CORE,
                reason="the bounded observation query failed before an authoritative decision",
            )
        if not proposals:
            return AuthoritativeDecision(
                status=DecisionStatus.INCONCLUSIVE,
                source=DecisionSource.CORE,
                reason="no policy hypothesis was submitted for domain-core assessment",
            )

        evaluation = _evaluate(
            proposals, observations, self._dynamic, self._excluded_catalog_candidate_ids,
        )
        if evaluation is None:
            _, _, unresolved = _resolve_candidates(
                proposals, excluded_catalog_candidate_ids=self._excluded_catalog_candidate_ids,
            )
            return AuthoritativeDecision(
                status=DecisionStatus.OUT_OF_SCOPE,
                source=DecisionSource.CORE,
                reason=_reason("no submitted hypothesis maps to the trusted candidate catalog", unresolved),
            )
        decision = evaluation.decision
        status = _decision_status(decision.status, decision.candidate_states)
        if decision.winner in evaluation.dynamic_verifications:
            status = DecisionStatus.DISCOVERED_VERIFIED
        winning_hypothesis_id = (
            evaluation.proposals_by_candidate[decision.winner][0].hypothesis_id
            if decision.winner is not None
            else None
        )
        notes = [*evaluation.unresolved, *evaluation.unavailable]
        return AuthoritativeDecision(
            status=status,
            source=DecisionSource.CORE,
            winning_hypothesis_id=winning_hypothesis_id,
            reason=_reason(
                "dynamic mechanism satisfies generic-root-cause-v1 and is returned as this "
                "investigation's root cause; not promoted to Catalog; alternatives are bounded "
                "by the recorded verification plan"
                if status == DecisionStatus.DISCOVERED_VERIFIED else decision.reason, notes,
            ),
        )


def _resolve_candidates(
    proposals: tuple[HypothesisProposal, ...],
    dynamic: dict[str, CompiledCandidate] | None = None,
    excluded_catalog_candidate_ids: frozenset[str] = frozenset(),
) -> tuple[list[Candidate], dict[str, list[HypothesisProposal]], list[str]]:
    selected: dict[str, Candidate] = {}
    proposals_by_candidate: dict[str, list[HypothesisProposal]] = {}
    unresolved: list[str] = []
    for proposal in proposals:
        registered = (dynamic or {}).get(proposal.candidate_id or "")
        if registered is not None:
            if (registered.source_hypothesis_id != proposal.hypothesis_id
                    or registered.candidate.causal_entity != proposal.causal_entity_ref):
                unresolved.append(f"{proposal.hypothesis_id}: dynamic candidate scope mismatch")
                continue
            candidate = registered.candidate
            selected[candidate.candidate_id] = candidate
            proposals_by_candidate.setdefault(candidate.candidate_id, []).append(proposal)
            continue
        catalog = tuple(
            candidate for candidate in mechanism_catalog(proposal.causal_entity_ref)
            if candidate.candidate_id not in excluded_catalog_candidate_ids
        )
        by_key = {
            _candidate_key(value): candidate
            for candidate in catalog
            for value in (candidate.candidate_id, candidate.fault_type)
        }
        requested = proposal.candidate_id or proposal.mechanism
        candidate = by_key.get(_candidate_key(requested))
        if candidate is None and proposal.candidate_id is None:
            candidate = by_key.get(_candidate_key(proposal.hypothesis_id.removeprefix("h-")))
        if candidate is None:
            unresolved.append(f"{proposal.hypothesis_id}: candidate is outside trusted catalog")
            continue
        selected.setdefault(candidate.candidate_id, candidate)
        proposals_by_candidate.setdefault(candidate.candidate_id, []).append(proposal)
    return list(selected.values()), proposals_by_candidate, unresolved


def _evaluate(
    proposals: tuple[HypothesisProposal, ...],
    observations: tuple[IssuedObservation, ...],
    dynamic: dict[str, CompiledCandidate] | None = None,
    excluded_catalog_candidate_ids: frozenset[str] = frozenset(),
) -> _CoreEvaluation | None:
    dynamic = dynamic or {}
    candidates, proposal_groups, unresolved = _resolve_candidates(
        proposals, dynamic, excluded_catalog_candidate_ids,
    )
    if not candidates:
        return None
    records_by_id = {item.observation.observation_id: item for item in observations}
    assessments_by_candidate: dict[str, tuple[Any, ...]] = {}
    records_by_candidate: dict[str, tuple[IssuedObservation, ...]] = {}
    assertions = []
    unavailable: list[str] = []
    for candidate in candidates:
        candidate_proposals = proposal_groups[candidate.candidate_id]
        scoped_refs = {
            observation_ref
            for proposal in candidate_proposals
            for observation_ref in proposal.observation_refs
        }
        scoped_records = tuple(
            records_by_id[item]
            for item in sorted(scoped_refs)
            if item in records_by_id
        )
        records_by_candidate[candidate.candidate_id] = scoped_records
        core_observations, candidate_unavailable = _normalize_observations(scoped_records)
        unavailable.extend(
            f"{candidate.candidate_id}: {item}" for item in candidate_unavailable
        )
        assessments = tuple(assess_dynamic(dynamic[candidate.candidate_id], scoped_records)) if (
            candidate.candidate_id in dynamic
        ) else tuple(
            OpenRcaEngine._assess(
                [candidate],
                core_observations,
                {condition.condition_id for condition in candidate.conditions},
            )
        )
        assessments_by_candidate[candidate.candidate_id] = assessments
        assertions.extend(
            item.assertion for item in assessments if item.assertion is not None
        )
    route = OpenRcaEngine._route(candidates, assertions)
    verifications = {
        key: verify_dynamic(value, assessments_by_candidate[key], records_by_candidate[key])
        for key, value in dynamic.items() if key in assessments_by_candidate
    }
    eligible = []
    for key in route.eligible_candidates:
        compiled = dynamic.get(key)
        if compiled is not None and not verifications[key].passed:
            continue
        eligible.append(key)
    if route.status == RouteStatus.ELIGIBLE:
        route = FamilyRoute(status=RouteStatus.ELIGIBLE, eligible_candidates=eligible) if eligible else (
            FamilyRoute(status=RouteStatus.UNKNOWN,
                        reasons=["dynamic verification is incomplete or contradicted"])
        )
    decision = EvidenceController().decide(
        [item.candidate_id for item in candidates],
        route,
        assertions,
    )
    return _CoreEvaluation(
        candidates=tuple(candidates),
        proposals_by_candidate={
            key: tuple(value) for key, value in proposal_groups.items()
        },
        assessments_by_candidate=assessments_by_candidate,
        records_by_candidate=records_by_candidate,
        decision=decision,
        unresolved=tuple(unresolved),
        unavailable=tuple(unavailable),
        dynamic_verifications=verifications,
    )


def _candidate_previews(
    evaluation: _CoreEvaluation,
) -> tuple[CandidateAssessmentPreview, ...]:
    state_by_candidate = {
        item.candidate: item for item in evaluation.decision.candidate_states
    }
    result = []
    for candidate in evaluation.candidates:
        condition_by_id = {item.condition_id: item for item in candidate.conditions}
        required_ids = set(candidate.positive_evidence_requirements)
        satisfied: set[str] = set()
        missing: set[str] = set()
        unavailable: set[str] = set()
        contradicting: set[str] = set()
        assessments = evaluation.assessments_by_candidate[candidate.candidate_id]
        scoped_signals = {
            signal
            for record in evaluation.records_by_candidate[candidate.candidate_id]
            for signal in _core_signals_for_record(record)
        }
        for assessment in assessments:
            condition = condition_by_id[assessment.condition_id]
            if condition.condition_id in required_ids:
                if assessment.observation is None:
                    target = unavailable if condition.signal in scoped_signals else missing
                    target.add(condition.signal)
                elif assessment.matched:
                    satisfied.add(condition.signal)
                else:
                    contradicting.add(condition.signal)
            elif condition.polarity == ConditionPolarity.CONTRADICT:
                if assessment.observation is None:
                    target = unavailable if condition.signal in scoped_signals else missing
                    target.add(condition.signal)
                elif assessment.matched:
                    contradicting.add(condition.signal)
        proposals = evaluation.proposals_by_candidate[candidate.candidate_id]
        state = state_by_candidate[candidate.candidate_id]
        if evaluation.decision.winner == candidate.candidate_id:
            evidence_status = (CandidateEvidenceStatus.DISCOVERED_VERIFIED
                               if candidate.candidate_id in evaluation.dynamic_verifications
                               else CandidateEvidenceStatus.CONFIRMED)
        elif contradicting or state.contradict_strength > 0:
            evidence_status = CandidateEvidenceStatus.CONTRADICTED
        elif state.support_strength > 0:
            evidence_status = CandidateEvidenceStatus.SUPPORTED
        elif not any(item.observation_refs for item in proposals):
            evidence_status = CandidateEvidenceStatus.UNASSESSED
        else:
            evidence_status = CandidateEvidenceStatus.INCONCLUSIVE
        result.append(
            CandidateAssessmentPreview(
                candidate_id=candidate.candidate_id,
                hypothesis_ids=tuple(item.hypothesis_id for item in proposals),
                evidence_status=evidence_status,
                satisfied_signals=tuple(sorted(satisfied)),
                missing_signals=tuple(sorted(missing)),
                contradicting_signals=tuple(sorted(contradicting)),
                unavailable_signals=tuple(sorted(unavailable)),
                required_capabilities=tuple(sorted(candidate.required_capabilities)),
                verification_pending=(candidate.candidate_id in evaluation.dynamic_verifications
                                      and evidence_status != CandidateEvidenceStatus.DISCOVERED_VERIFIED),
            )
        )
    return tuple(sorted(result, key=lambda item: item.candidate_id))


def _discriminating_signals(
    evaluation: _CoreEvaluation,
    previews: tuple[CandidateAssessmentPreview, ...],
    observations: tuple[IssuedObservation, ...],
    capabilities: tuple[CapabilityDescriptor, ...] | None,
) -> tuple[DiscriminatingSignalView, ...]:
    unresolved = tuple(
        item
        for item in previews
        if item.evidence_status
        not in {CandidateEvidenceStatus.CONFIRMED, CandidateEvidenceStatus.DISCOVERED_VERIFIED,
                CandidateEvidenceStatus.CONTRADICTED}
    )
    if len(unresolved) < 2:
        return ()
    unresolved_ids = tuple(item.candidate_id for item in unresolved)
    missing_by_candidate = {
        item.candidate_id: {*item.missing_signals, *item.unavailable_signals}
        for item in unresolved
    }
    condition_owners: dict[str, set[str]] = {}
    capability_by_signal: dict[str, str] = {}
    for candidate in evaluation.candidates:
        if candidate.candidate_id not in missing_by_candidate:
            continue
        required_ids = set(candidate.positive_evidence_requirements)
        for condition in candidate.conditions:
            if (
                condition.condition_id in required_ids
                and condition.signal in missing_by_candidate[candidate.candidate_id]
            ):
                condition_owners.setdefault(condition.signal, set()).add(
                    candidate.candidate_id
                )
                capability_by_signal.setdefault(condition.signal, condition.capability_id)
    issued_refs_by_signal: dict[str, set[str]] = {}
    for record in observations:
        for signal in _core_signals_for_record(record):
            issued_refs_by_signal.setdefault(signal, set()).add(
                record.observation.observation_id
            )
    available_tools = (
        None
        if capabilities is None
        else {item.tool_name for item in capabilities if item.available}
    )
    result = []
    for signal, owners in condition_owners.items():
        capability = capability_by_signal[signal]
        required_tool = CAPABILITY_TO_TOOL.get(capability, capability if capability in {
            "get_metric_series", "compare_metric_windows", "get_trace_stage_breakdown"
        } else None)
        available = required_tool is not None and (
            available_tools is None or required_tool in available_tools
        )
        result.append(
            DiscriminatingSignalView(
                signal=signal,
                candidate_ids=tuple(sorted(owners)),
                compared_candidate_ids=unresolved_ids,
                required_capability=capability,
                required_tool=required_tool,
                issued_observation_refs=tuple(
                    sorted(issued_refs_by_signal.get(signal, ()))
                ),
                available=available,
            )
        )
    return tuple(sorted(result, key=lambda item: (not item.available, item.signal)))


def _core_signals_for_record(record: IssuedObservation) -> tuple[str, ...]:
    signal = record.observation.signal
    if signal in DIRECT_CORE_SIGNALS:
        return (signal,)
    if signal in WINDOW_RATIO_SIGNALS:
        return (WINDOW_RATIO_SIGNALS[signal],)
    if signal == "cpu_usage_rate":
        return ("node_cpu_high_fraction", "node_cpu_peak")
    if signal == "memory_usage_rate":
        return ("node_memory_peak",)
    if signal == "disk_usage_rate":
        return ("node_disk_peak",)
    return (signal,)


def _normalize_observations(
    records: tuple[IssuedObservation, ...],
) -> tuple[list[CoreObservation], list[str]]:
    normalized: dict[str, CoreObservation] = {}
    unavailable: list[str] = []
    numeric_series: dict[str, list[tuple[float, IssuedObservation]]] = {}
    for record in records:
        observation = record.observation
        if observation.data_status not in TRUSTED_DATA_STATUSES or observation.value is None:
            unavailable.append(f"{observation.observation_id}: observation is not trusted evidence")
            continue
        if observation.signal in DIRECT_CORE_SIGNALS and isinstance(
            observation.value, (int, float, str)
        ):
            normalized[observation.signal] = _core_observation(
                observation.signal, observation.value, record
            )
            continue
        mapped_signal = WINDOW_RATIO_SIGNALS.get(observation.signal)
        if mapped_signal is not None and isinstance(observation.value, dict):
            ratio = observation.value.get("ratio")
            if isinstance(ratio, (int, float)):
                normalized[mapped_signal] = _core_observation(mapped_signal, ratio, record)
            else:
                unavailable.append(f"{observation.observation_id}: comparison ratio is unavailable")
            continue
        if observation.signal in {
            "cpu_usage_rate",
            "memory_usage_rate",
            "disk_usage_rate",
        } and isinstance(observation.value, (int, float)):
            numeric_series.setdefault(observation.signal, []).append((float(observation.value), record))

    _add_node_summaries(normalized, numeric_series)
    return list(normalized.values()), unavailable


def _add_node_summaries(
    normalized: dict[str, CoreObservation],
    series: dict[str, list[tuple[float, IssuedObservation]]],
) -> None:
    mappings = {
        "memory_usage_rate": "node_memory_peak",
        "disk_usage_rate": "node_disk_peak",
    }
    for source, target in mappings.items():
        values = series.get(source, [])
        if values:
            normalized[target] = _core_observation(target, max(item[0] for item in values), values[0][1])
    cpu = series.get("cpu_usage_rate", [])
    if cpu:
        normalized["node_cpu_peak"] = _core_observation(
            "node_cpu_peak", max(item[0] for item in cpu), cpu[0][1]
        )
        normalized["node_cpu_high_fraction"] = _core_observation(
            "node_cpu_high_fraction",
            sum(item[0] >= 80.0 for item in cpu) / len(cpu),
            cpu[0][1],
        )


def _core_observation(
    signal: str,
    value: float | str,
    record: IssuedObservation,
) -> CoreObservation:
    return CoreObservation(
        signal=signal,
        value=value,
        query_ref=record.query_ref,
        observed_entity=record.observation.entity_ref,
    )


def _decision_status(core_status: CoreDecisionStatus, states: list[Any]) -> DecisionStatus:
    if core_status == CoreDecisionStatus.CONFIRMED:
        return DecisionStatus.CONFIRMED_ROOT_CAUSE
    if core_status == CoreDecisionStatus.OUT_OF_SCOPE:
        return DecisionStatus.OUT_OF_SCOPE
    if core_status == CoreDecisionStatus.QUERY_FAILED:
        return DecisionStatus.QUERY_FAILED
    if any(item.support_strength > 0 for item in states):
        return DecisionStatus.SUPPORTED_HYPOTHESIS
    return DecisionStatus.INCONCLUSIVE


def _candidate_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def _reason(base: str, notes: list[str]) -> str:
    suffix = f"; {'; '.join(notes)}" if notes else ""
    return f"{base}{suffix}"[:1000]
