from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from rca_agent.core.evidence_control import DecisionStatus as CoreDecisionStatus
from rca_agent.core.evidence_control import EvidenceController
from rca_agent.core.v3.catalog import mechanism_catalog
from rca_agent.core.v3.engine import OpenRcaEngine
from rca_agent.core.v3.models import Candidate
from rca_agent.core.v3.models import Observation as CoreObservation
from rca_agent.models import (
    AuthoritativeDecision,
    DataStatus,
    DecisionSource,
    DecisionStatus,
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


@dataclass(frozen=True, slots=True)
class IssuedObservation:
    observation: Observation
    query_ref: str


class CoreDecisionProvider:
    """Translate issued Agent facts into the internal evidence-controlled decision kernel."""

    def preview(
        self,
        proposals: tuple[HypothesisProposal, ...],
        observations: tuple[IssuedObservation, ...],
    ) -> AuthoritativeDecision:
        """Assess current evidence without freezing or mutating an episode."""

        return self.decide(proposals, observations, StopReason.POLICY_REQUEST)

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

        candidates, proposal_by_candidate, unresolved = _resolve_candidates(proposals)
        if not candidates:
            return AuthoritativeDecision(
                status=DecisionStatus.OUT_OF_SCOPE,
                source=DecisionSource.CORE,
                reason=_reason("no submitted hypothesis maps to the trusted candidate catalog", unresolved),
            )

        core_observations, unavailable = _normalize_observations(observations)
        executable = {
            condition.condition_id
            for candidate in candidates
            for condition in candidate.conditions
        }
        assessments = OpenRcaEngine._assess(candidates, core_observations, executable)
        assertions = [item.assertion for item in assessments if item.assertion is not None]
        decision = EvidenceController().decide(
            [item.candidate_id for item in candidates],
            OpenRcaEngine._route(candidates, assertions),
            assertions,
        )
        status = _decision_status(decision.status, decision.candidate_states)
        winning_hypothesis_id = (
            proposal_by_candidate[decision.winner].hypothesis_id
            if decision.winner is not None
            else None
        )
        notes = [*unresolved, *unavailable]
        return AuthoritativeDecision(
            status=status,
            source=DecisionSource.CORE,
            winning_hypothesis_id=winning_hypothesis_id,
            reason=_reason(decision.reason, notes),
        )


def _resolve_candidates(
    proposals: tuple[HypothesisProposal, ...],
) -> tuple[list[Candidate], dict[str, HypothesisProposal], list[str]]:
    selected: dict[str, Candidate] = {}
    proposal_by_candidate: dict[str, HypothesisProposal] = {}
    unresolved: list[str] = []
    for proposal in proposals:
        catalog = mechanism_catalog(proposal.causal_entity_ref)
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
        proposal_by_candidate.setdefault(candidate.candidate_id, proposal)
    return list(selected.values()), proposal_by_candidate, unresolved


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
