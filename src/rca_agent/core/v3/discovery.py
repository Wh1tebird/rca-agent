from __future__ import annotations

import json
import os
import re
from collections.abc import Callable
from enum import StrEnum
from typing import Protocol
from urllib.request import Request, urlopen

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rca_agent.core.v3.catalog import mechanism_catalog
from rca_agent.core.v3.execution import ObservationProvider
from rca_agent.core.v3.models import (
    Candidate,
    CandidateProposal,
    Comparison,
    ConditionPolarity,
    EvidenceCondition,
    Incident,
    MechanismFamily,
    Observation,
    ObservationCapability,
    OpenRcaResult,
    TypedQueryPlan,
)


class PredictedObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    signal: str
    comparison: Comparison
    threshold: float | str
    independent_group: str = Field(min_length=1)
    rationale: str = Field(min_length=1)


class MechanismHypothesis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hypothesis_id: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_]{1,63}$")
    mechanism: str = Field(min_length=3)
    family: MechanismFamily
    causal_explanation: str = Field(min_length=10)
    explained_observations: list[str] = Field(min_length=1)
    unexplained_observations: list[str] = Field(min_length=1)
    predicted_observations: list[PredictedObservation] = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def unique_prediction_groups(self) -> MechanismHypothesis:
        signals = [item.signal for item in self.predicted_observations]
        if len(signals) != len(set(signals)):
            raise ValueError("predicted observation signals must be unique")
        return self


class DiscoveryPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hypotheses: list[MechanismHypothesis] = Field(min_length=1, max_length=4)


class MechanismDiscoveryResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    generator: str
    hypotheses: list[MechanismHypothesis]
    rejected_hypotheses: list[str] = Field(default_factory=list)


class AlignmentStatus(StrEnum):
    MATCH_EXISTING = "match_existing"
    CREATE_NEW = "create_new"
    UNCERTAIN = "uncertain"


class AlignmentDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hypothesis_id: str
    status: AlignmentStatus
    candidate_id: str | None = None
    rationale: str = Field(min_length=1)

    @model_validator(mode="after")
    def status_candidate_contract(self) -> AlignmentDecision:
        if self.status == AlignmentStatus.MATCH_EXISTING and not self.candidate_id:
            raise ValueError("matched alignment requires candidate_id")
        if self.status != AlignmentStatus.MATCH_EXISTING and self.candidate_id is not None:
            raise ValueError("only matched alignment may name candidate_id")
        return self


class AlignmentPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decisions: list[AlignmentDecision] = Field(min_length=1)


class CandidateAlignmentResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    generator: str
    decisions: list[AlignmentDecision]
    rejected_decisions: list[str] = Field(default_factory=list)


class OpenDiscoveryRun(BaseModel):
    model_config = ConfigDict(extra="forbid")

    preflight_observations: list[Observation]
    preflight_missing: list[str]
    discovery: MechanismDiscoveryResult
    alignment: CandidateAlignmentResult
    proposal: CandidateProposal
    rca: OpenRcaResult


class DiscoveryGenerator(Protocol):
    def discover(
        self,
        incident: Incident,
        capabilities: list[ObservationCapability],
        observations: list[Observation],
    ) -> MechanismDiscoveryResult: ...


class CandidateAligner(Protocol):
    def align(
        self,
        hypotheses: list[MechanismHypothesis],
        candidates: list[Candidate],
    ) -> CandidateAlignmentResult: ...


class DeepSeekJsonClient:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout_seconds: float = 30.0,
        transport: Callable[[Request, float], bytes] | None = None,
    ) -> None:
        self._api_key = api_key or os.getenv("DEEPSEEK_API_KEY", "")
        self._base_url = (
            base_url or os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        ).rstrip("/")
        self._model = model or os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        self._timeout = timeout_seconds
        self._transport = transport or self._urlopen

    def complete(self, system: str, payload: dict[str, object]) -> object:
        if not self._api_key:
            raise RuntimeError("missing_api_key")
        body = {
            "model": self._model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
        }
        request = Request(
            f"{self._base_url}/chat/completions",
            data=json.dumps(body).encode(),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        response = json.loads(self._transport(request, self._timeout))
        return json.loads(response["choices"][0]["message"]["content"])

    @staticmethod
    def _urlopen(request: Request, timeout: float) -> bytes:
        with urlopen(request, timeout=timeout) as response:
            return response.read()


class DeepSeekMechanismDiscovery:
    def __init__(self, client: DeepSeekJsonClient | None = None) -> None:
        self._client = client or DeepSeekJsonClient()

    def discover(
        self,
        incident: Incident,
        capabilities: list[ObservationCapability],
        observations: list[Observation],
    ) -> MechanismDiscoveryResult:
        try:
            raw = self._client.complete(
                _DISCOVERY_PROMPT,
                _discovery_context(incident, capabilities, observations),
            )
            payload = DiscoveryPayload.model_validate(raw)
            allowed_signals = {
                signal for item in capabilities if item.available for signal in item.signals
            }
            observed_signals = {item.signal for item in observations}
            accepted = []
            rejected = []
            for hypothesis in payload.hypotheses:
                bad_explained = set(hypothesis.explained_observations) - observed_signals
                bad_predictions = {
                    item.signal for item in hypothesis.predicted_observations
                } - allowed_signals
                if bad_explained or bad_predictions:
                    rejected.append(f"{hypothesis.hypothesis_id}: ungrounded signals")
                else:
                    accepted.append(hypothesis)
            return MechanismDiscoveryResult(
                generator="deepseek_open_discovery_v1",
                hypotheses=accepted,
                rejected_hypotheses=rejected,
            )
        except Exception as exc:  # noqa: BLE001 - discovery fails closed
            return MechanismDiscoveryResult(
                generator="deepseek_open_discovery_failed",
                hypotheses=[],
                rejected_hypotheses=[f"discovery unavailable: {type(exc).__name__}"],
            )


class DeepSeekCandidateAligner:
    def __init__(self, client: DeepSeekJsonClient | None = None) -> None:
        self._client = client or DeepSeekJsonClient()

    def align(
        self,
        hypotheses: list[MechanismHypothesis],
        candidates: list[Candidate],
    ) -> CandidateAlignmentResult:
        if not hypotheses:
            return CandidateAlignmentResult(generator="alignment_skipped", decisions=[])
        try:
            raw = self._client.complete(
                _ALIGNMENT_PROMPT,
                {
                    "hypotheses": [item.model_dump(mode="json") for item in hypotheses],
                    "candidate_catalog": [
                        {
                            "candidate_id": item.candidate_id,
                            "fault_type": item.fault_type,
                            "family": item.family,
                            "proposal_reason": item.proposal_reason,
                        }
                        for item in candidates
                    ],
                },
            )
            payload = AlignmentPayload.model_validate(raw)
            known_hypotheses = {item.hypothesis_id for item in hypotheses}
            known_candidates = {item.candidate_id for item in candidates}
            accepted = []
            rejected = []
            seen = set()
            for decision in payload.decisions:
                invalid = (
                    decision.hypothesis_id not in known_hypotheses
                    or decision.hypothesis_id in seen
                    or (
                        decision.status == AlignmentStatus.MATCH_EXISTING
                        and decision.candidate_id not in known_candidates
                    )
                )
                if invalid:
                    rejected.append(f"{decision.hypothesis_id}: invalid alignment")
                    continue
                seen.add(decision.hypothesis_id)
                accepted.append(decision)
            return CandidateAlignmentResult(
                generator="deepseek_candidate_alignment_v1",
                decisions=accepted,
                rejected_decisions=rejected,
            )
        except Exception as exc:  # noqa: BLE001 - alignment fails closed
            return CandidateAlignmentResult(
                generator="deepseek_candidate_alignment_failed",
                decisions=[],
                rejected_decisions=[f"alignment unavailable: {type(exc).__name__}"],
            )


class StaticProposalGenerator:
    def __init__(self, proposal: CandidateProposal) -> None:
        self._proposal = proposal

    def propose(self, incident, capabilities, scout_observations) -> CandidateProposal:
        return self._proposal


class OpenDiscoveryPipeline:
    def __init__(
        self,
        capabilities: list[ObservationCapability],
        discoverer: DiscoveryGenerator | None = None,
        aligner: CandidateAligner | None = None,
    ) -> None:
        self._capabilities = capabilities
        self._discoverer = discoverer or DeepSeekMechanismDiscovery()
        self._aligner = aligner or DeepSeekCandidateAligner()

    def investigate(
        self,
        incident: Incident,
        provider: ObservationProvider,
        excluded_candidate_ids: set[str] | None = None,
    ) -> OpenDiscoveryRun:
        from rca_agent.core.v3.engine import OpenRcaEngine

        observations, missing = mechanism_independent_preflight(
            incident, provider, self._capabilities
        )
        discovery = self._discoverer.discover(
            incident, self._capabilities, observations
        )
        catalog = [
            item
            for item in mechanism_catalog(incident.alert_entity.split("::", 1)[0])
            if item.candidate_id not in (excluded_candidate_ids or set())
        ]
        alignment = self._aligner.align(discovery.hypotheses, catalog)
        proposal = compile_aligned_proposal(discovery, alignment, catalog, self._capabilities)
        result = OpenRcaEngine(
            self._capabilities,
            proposal_generator=StaticProposalGenerator(proposal),
        ).investigate(incident, provider)
        return OpenDiscoveryRun(
            preflight_observations=observations,
            preflight_missing=missing,
            discovery=discovery,
            alignment=alignment,
            proposal=proposal,
            rca=result,
        )


def mechanism_independent_preflight(
    incident: Incident,
    provider: ObservationProvider,
    capabilities: list[ObservationCapability],
) -> tuple[list[Observation], list[str]]:
    plans = [
        TypedQueryPlan(
            condition_id=f"discovery-{signal}",
            capability_id=capability.capability_id,
            query_type=capability.typed_query,
            signal=signal,
            bounded_window_start=incident.window_start,
            bounded_window_end=incident.window_end,
        )
        for capability in capabilities
        if capability.available
        for signal in capability.signals
    ]
    observations, missing = provider.execute(plans)
    unique = {item.signal: item for item in [*provider.scout(), *observations]}
    return list(unique.values()), missing


def compile_aligned_proposal(
    discovery: MechanismDiscoveryResult,
    alignment: CandidateAlignmentResult,
    catalog: list[Candidate],
    capabilities: list[ObservationCapability],
) -> CandidateProposal:
    hypotheses = {item.hypothesis_id: item for item in discovery.hypotheses}
    candidates = {item.candidate_id: item for item in catalog}
    compiled = []
    rejected = [*discovery.rejected_hypotheses, *alignment.rejected_decisions]
    for decision in alignment.decisions:
        hypothesis = hypotheses.get(decision.hypothesis_id)
        if hypothesis is None or decision.status == AlignmentStatus.UNCERTAIN:
            continue
        if decision.status == AlignmentStatus.MATCH_EXISTING:
            candidate = _compatible_catalog_candidate(
                hypothesis,
                candidates,
                decision.candidate_id,
            )
        else:
            candidate = compile_new_candidate(hypothesis, capabilities)
        if candidate is None:
            rejected.append(f"{decision.hypothesis_id}: evidence compilation rejected")
        else:
            compiled.append(candidate)
    return CandidateProposal(
        generator="open_discovery_alignment_compilation_v1",
        candidates=_deduplicate(compiled),
        rejected_candidates=rejected,
    )


def compile_new_candidate(
    hypothesis: MechanismHypothesis,
    capabilities: list[ObservationCapability],
) -> Candidate | None:
    capability_by_signal = {
        signal: capability.capability_id
        for capability in capabilities
        if capability.available
        for signal in capability.signals
    }
    predictions = {item.signal: item for item in hypothesis.predicted_observations}
    mentioned_signals = {
        *hypothesis.explained_observations,
        *(
            signal
            for signal, prediction in predictions.items()
            if any(
                template.family == hypothesis.family
                and template.signal == signal
                and template.comparison == prediction.comparison
                for template in _TRUSTED_EVIDENCE_TEMPLATES
            )
        ),
    }
    templates = [
        template
        for template in _TRUSTED_EVIDENCE_TEMPLATES
        if template.family == hypothesis.family
        and template.signal in mentioned_signals
        and template.signal in capability_by_signal
    ]
    if len({item.independent_group for item in templates}) < 2:
        return None
    candidate_id = re.sub(r"[^A-Za-z0-9_]", "_", hypothesis.hypothesis_id)[:64]
    conditions = [
        EvidenceCondition(
            condition_id=f"{candidate_id}-{template.signal}",
            candidate_id=candidate_id,
            polarity=ConditionPolarity.SUPPORT,
            signal=template.signal,
            comparison=template.comparison,
            threshold=template.threshold,
            capability_id=capability_by_signal[template.signal],
            independent_group=template.independent_group,
            strength=2.0,
            reason=template.reason,
        )
        for template in templates
    ]
    return Candidate(
        candidate_id=candidate_id,
        fault_type=hypothesis.mechanism,
        family=hypothesis.family,
        causal_entity="incident alert entity",
        proposal_reason=hypothesis.causal_explanation,
        positive_evidence_requirements=[item.condition_id for item in conditions],
        falsification_conditions=[
            f"positive condition {item.condition_id} is observed and does not match"
            for item in conditions
        ],
        confounders=hypothesis.unexplained_observations,
        required_capabilities=sorted({item.capability_id for item in conditions}),
        conditions=conditions,
        knowledge_provenance="AI open discovery plus deterministic evidence compiler",
    )


def _discovery_context(incident, capabilities, observations) -> dict[str, object]:
    return {
        "incident": {
            "alert_title": incident.alert_title,
            "alert_entity": incident.alert_entity,
            "window_start": incident.window_start,
            "window_end": incident.window_end,
        },
        "observations": [
            {
                "signal": item.signal,
                "value": item.value,
                "observed_entity": item.observed_entity,
            }
            for item in observations
        ],
        "available_observation_signals": sorted(
            {
                signal
                for item in capabilities
                if item.available
                for signal in item.signals
            }
        ),
    }


def _deduplicate(candidates: list[Candidate]) -> list[Candidate]:
    return list({item.candidate_id: item for item in candidates}.values())


class TrustedEvidenceTemplate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    family: MechanismFamily
    signal: str
    comparison: Comparison
    threshold: float | str
    independent_group: str
    reason: str


_TRUSTED_EVIDENCE_TEMPLATES = (
    TrustedEvidenceTemplate(
        family=MechanismFamily.RESOURCE,
        signal="node_cpu_peak",
        comparison=Comparison.GTE,
        threshold=80.0,
        independent_group="node_cpu_saturation",
        reason="Trusted resource template requires high node CPU, not merely a changed ratio.",
    ),
    TrustedEvidenceTemplate(
        family=MechanismFamily.RESOURCE,
        signal="node_cpu_high_fraction",
        comparison=Comparison.GTE,
        threshold=0.2,
        independent_group="node_cpu_sustained",
        reason="Trusted resource template requires sustained node CPU saturation.",
    ),
    TrustedEvidenceTemplate(
        family=MechanismFamily.APPLICATION,
        signal="service_workload_ratio",
        comparison=Comparison.GTE,
        threshold=2.0,
        independent_group="application_demand",
        reason="Trusted application template requires a material workload increase.",
    ),
    TrustedEvidenceTemplate(
        family=MechanismFamily.APPLICATION,
        signal="service_cpu_ratio",
        comparison=Comparison.GTE,
        threshold=1.5,
        independent_group="application_cpu_response",
        reason="Trusted application template requires an independent CPU response.",
    ),
    TrustedEvidenceTemplate(
        family=MechanismFamily.NETWORK,
        signal="network_stage_ms",
        comparison=Comparison.GTE,
        threshold=100.0,
        independent_group="network_delay",
        reason="Trusted network template requires latency concentrated in a network stage.",
    ),
    TrustedEvidenceTemplate(
        family=MechanismFamily.DATABASE_DEPENDENCY,
        signal="connection_duration_ms",
        comparison=Comparison.GTE,
        threshold=100.0,
        independent_group="database_connection_delay",
        reason="Trusted database template measures connection-stage delay.",
    ),
    TrustedEvidenceTemplate(
        family=MechanismFamily.DATABASE_DEPENDENCY,
        signal="sql_duration_share",
        comparison=Comparison.GTE,
        threshold=0.8,
        independent_group="database_query_duration",
        reason="Trusted database template requires SQL execution to dominate request time.",
    ),
    TrustedEvidenceTemplate(
        family=MechanismFamily.DATABASE_DEPENDENCY,
        signal="sql_shape",
        comparison=Comparison.EQ,
        threshold="complex_cross_join",
        independent_group="database_query_structure",
        reason="Trusted database template independently checks complex SQL structure.",
    ),
)


def _compatible_catalog_candidate(
    hypothesis: MechanismHypothesis,
    candidates: dict[str, Candidate],
    requested_candidate_id: str | None,
) -> Candidate | None:
    requested = candidates.get(requested_candidate_id or "")
    if requested is None or requested.family != hypothesis.family:
        return None
    explained = set(hypothesis.explained_observations)
    predictions = {item.signal: item for item in hypothesis.predicted_observations}

    def score(candidate: Candidate) -> tuple[int, int, int]:
        signals = {
            item.signal
            for item in candidate.conditions
            if item.polarity == ConditionPolarity.SUPPORT
        }
        strong_predictions = sum(
            _prediction_reaches_condition(predictions[condition.signal], condition)
            for condition in candidate.conditions
            if condition.polarity == ConditionPolarity.SUPPORT
            and condition.signal in predictions
        )
        return (
            strong_predictions,
            2 * len(signals & explained) + len(signals & set(predictions)),
            len(signals & explained),
        )

    same_family = [item for item in candidates.values() if item.family == hypothesis.family]
    best = max(same_family, key=lambda item: (score(item), item.candidate_id), default=requested)
    return best if score(best) > score(requested) else requested


def _prediction_reaches_condition(
    prediction: PredictedObservation,
    condition: EvidenceCondition,
) -> int:
    if prediction.comparison != condition.comparison:
        return 0
    if prediction.comparison == Comparison.GTE:
        return int(
            isinstance(prediction.threshold, (int, float))
            and isinstance(condition.threshold, (int, float))
            and prediction.threshold >= condition.threshold
        )
    if prediction.comparison == Comparison.LTE:
        return int(
            isinstance(prediction.threshold, (int, float))
            and isinstance(condition.threshold, (int, float))
            and prediction.threshold <= condition.threshold
        )
    return int(str(prediction.threshold) == str(condition.threshold))


_DISCOVERY_PROMPT = """Perform open mechanism discovery for an incident.
You MUST NOT assume or name a dataset label, candidate catalog, or known candidate ID.
Infer causal mechanisms only from the incident and observations. Return JSON with
exactly {"hypotheses": [...]} matching the supplied schema concepts. For each
hypothesis provide: hypothesis_id, mechanism, family, causal_explanation,
explained_observations (signal names actually observed), unexplained_observations,
predicted_observations, and confidence. Each predicted observation needs signal,
comparison (gte/lte/eq/contains), threshold, independent_group, and rationale.
Prefer mechanisms that explain where latency or failure concentrates. Use at least
two different causal dimensions when the observations support them. Do not output
queries, SQL text, file paths, credentials, candidate IDs, prose, or markdown.
family must be exactly one of: resource, network, application, database_dependency.
independent_group names a causal evidence dimension, not a service or entity. Include
the strongest direct mechanism observations in predicted_observations even when they
have already been observed, so the verification layer can reacquire them.
"""


_ALIGNMENT_PROMPT = """Align independently discovered mechanisms to a candidate catalog.
Return JSON with exactly {"decisions": [...]}. Provide one decision per hypothesis:
hypothesis_id, status (match_existing/create_new/uncertain), candidate_id (only for
match_existing), and rationale. Match only when the causal mechanism is semantically
equivalent, not merely because symptoms overlap. Use create_new when no catalog
mechanism is equivalent. Use uncertain rather than forcing a match. Do not generate
evidence conditions, queries, prose, or markdown.
"""
