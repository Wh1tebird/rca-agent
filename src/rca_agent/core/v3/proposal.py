from __future__ import annotations

import json
import os
from collections.abc import Callable
from typing import Protocol
from urllib.request import Request, urlopen

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from rca_agent.core.v3.catalog import mechanism_catalog
from rca_agent.core.v3.models import (
    Candidate,
    CandidateProposal,
    Incident,
    MechanismFamily,
    Observation,
    ObservationCapability,
)


class ProposalGenerator(Protocol):
    def propose(
        self,
        incident: Incident,
        capabilities: list[ObservationCapability],
        scout_observations: list[Observation],
    ) -> CandidateProposal: ...


class RuleProposalGenerator:
    """Deterministic v3 baseline; selects catalog candidates from incident observations."""

    def propose(
        self,
        incident: Incident,
        capabilities: list[ObservationCapability],
        scout_observations: list[Observation],
    ) -> CandidateProposal:
        values = {item.signal: item.value for item in scout_observations}
        families: set[MechanismFamily] = set()
        if any(signal.startswith("node_") for signal in values):
            families.add(MechanismFamily.RESOURCE)
        if _numeric(values.get("service_error_ratio"), minimum=1.5):
            families.add(MechanismFamily.APPLICATION)
        if _numeric(values.get("service_workload_ratio"), minimum=1.5):
            families.add(MechanismFamily.APPLICATION)
        text = f"{incident.alert_title} {incident.alert_entity}".casefold()
        if any(word in text for word in ("latency", "timeout", "响应", "慢", "延迟")):
            families.update({MechanismFamily.NETWORK, MechanismFamily.DATABASE_DEPENDENCY})
        if any(word in text for word in ("error", "错误", "异常", "5xx")):
            families.add(MechanismFamily.APPLICATION)
        if not families:
            families = set(MechanismFamily)
        selected = [
            item
            for item in mechanism_catalog(incident.alert_entity.split("::", 1)[0])
            if item.family in families
        ]
        return CandidateProposal(generator="rule_catalog_v1", candidates=selected)


class FilteredRuleProposalGenerator:
    """Experimental rule baseline with named candidates absent from the seed set."""

    def __init__(self, excluded_candidate_ids: set[str]) -> None:
        self._excluded = frozenset(excluded_candidate_ids)
        self._rules = RuleProposalGenerator()

    def propose(
        self,
        incident: Incident,
        capabilities: list[ObservationCapability],
        scout_observations: list[Observation],
    ) -> CandidateProposal:
        proposal = self._rules.propose(incident, capabilities, scout_observations)
        return proposal.model_copy(
            update={
                "generator": "filtered_rule_catalog_v1",
                "candidates": [
                    item
                    for item in proposal.candidates
                    if item.candidate_id not in self._excluded
                ],
            }
        )


class AiProposalPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    selected_candidate_ids: list[str] = Field(default_factory=list)
    candidates: list[Candidate] = Field(default_factory=list)


class ConstrainedAiProposalGenerator:
    """Schema gate for AI proposals; no model call or final-decision authority lives here."""

    def __init__(self, fallback: ProposalGenerator | None = None) -> None:
        self._fallback = fallback or RuleProposalGenerator()

    def validate_or_fallback(
        self,
        raw_payload: object,
        incident: Incident,
        capabilities: list[ObservationCapability],
        scout_observations: list[Observation],
    ) -> CandidateProposal:
        safe_rejections: list[str] = []
        try:
            payload = AiProposalPayload.model_validate(raw_payload)
            candidates, rejected = self._capability_gate(payload.candidates, capabilities)
            safe_rejections = rejected
            if not candidates:
                raise ValueError("AI proposal has no capability-grounded candidate")
            return CandidateProposal(
                generator="ai_schema_gated_v1",
                candidates=candidates,
                rejected_candidates=rejected,
            )
        except (ValidationError, ValueError, TypeError) as exc:
            fallback = self._fallback.propose(incident, capabilities, scout_observations)
            return fallback.model_copy(
                update={
                    "generator": "rule_catalog_v1_after_ai_rejection",
                    "rejected_candidates": safe_rejections
                    or [f"AI proposal rejected: {type(exc).__name__}"],
                }
            )

    @staticmethod
    def _capability_gate(
        candidates: list[Candidate], capabilities: list[ObservationCapability]
    ) -> tuple[list[Candidate], list[str]]:
        known = {item.capability_id: set(item.signals) for item in capabilities}
        accepted = []
        rejected = []
        for candidate in candidates:
            unknown = sorted(set(candidate.required_capabilities) - set(known))
            if unknown:
                rejected.append(
                    f"{candidate.candidate_id}: unknown capabilities {unknown}"
                )
                continue
            invalid_conditions = [
                item.condition_id
                for item in candidate.conditions
                if item.capability_id not in candidate.required_capabilities
                or item.signal not in known.get(item.capability_id, set())
            ]
            if invalid_conditions:
                rejected.append(
                    f"{candidate.candidate_id}: invalid condition signals "
                    f"{sorted(invalid_conditions)}"
                )
                continue
            accepted.append(candidate)
        return accepted, rejected


class DeepSeekProposalGenerator:
    """DeepSeek-backed proposer; the frozen safety controller remains authoritative."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout_seconds: float = 30.0,
        fallback: ProposalGenerator | None = None,
        transport: Callable[[Request, float], bytes] | None = None,
        excluded_seed_candidate_ids: set[str] | None = None,
    ) -> None:
        self._api_key = api_key or os.getenv("DEEPSEEK_API_KEY", "")
        self._base_url = (base_url or os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")).rstrip("/")
        self._model = model or os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        self._timeout_seconds = timeout_seconds
        self._fallback = fallback or RuleProposalGenerator()
        self._gate = ConstrainedAiProposalGenerator(self._fallback)
        self._transport = transport or self._urlopen
        self._excluded_seed_candidate_ids = frozenset(
            excluded_seed_candidate_ids or set()
        )

    def propose(
        self,
        incident: Incident,
        capabilities: list[ObservationCapability],
        scout_observations: list[Observation],
    ) -> CandidateProposal:
        if not self._api_key:
            return self._fallback_after_error(
                incident, capabilities, scout_observations, "missing_api_key"
            )
        catalog = [
            item
            for item in mechanism_catalog(incident.alert_entity.split("::", 1)[0])
            if item.candidate_id not in self._excluded_seed_candidate_ids
        ]
        request_payload = {
            "model": self._model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(
                        _sanitized_context(
                            incident, capabilities, scout_observations, catalog
                        ),
                        ensure_ascii=False,
                    ),
                },
            ],
        }
        request = Request(
            f"{self._base_url}/chat/completions",
            data=json.dumps(request_payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            response = json.loads(self._transport(request, self._timeout_seconds))
            content = response["choices"][0]["message"]["content"]
            raw = json.loads(content)
            payload = AiProposalPayload.model_validate(raw)
            by_id = {item.candidate_id: item for item in catalog}
            selected = [
                by_id[candidate_id]
                for candidate_id in payload.selected_candidate_ids
                if candidate_id in by_id
            ]
            unknown_ids = sorted(set(payload.selected_candidate_ids) - set(by_id))
            combined = _deduplicate_candidates([*selected, *payload.candidates])
            proposal = self._gate.validate_or_fallback(
                {"candidates": [item.model_dump(mode="json") for item in combined]},
                incident,
                capabilities,
                scout_observations,
            )
            if proposal.generator.startswith("rule_catalog_v1_after_ai"):
                return proposal
            rejected = [*proposal.rejected_candidates]
            rejected.extend(f"unknown seed candidate: {item}" for item in unknown_ids)
            return proposal.model_copy(
                update={
                    "generator": "deepseek_schema_gated_v1",
                    "rejected_candidates": rejected,
                }
            )
        except Exception as exc:  # noqa: BLE001 - model failure must fail closed
            return self._fallback_after_error(
                incident, capabilities, scout_observations, type(exc).__name__
            )

    @staticmethod
    def _urlopen(request: Request, timeout: float) -> bytes:
        with urlopen(request, timeout=timeout) as response:
            return response.read()

    def _fallback_after_error(
        self,
        incident: Incident,
        capabilities: list[ObservationCapability],
        scout_observations: list[Observation],
        reason: str,
    ) -> CandidateProposal:
        fallback = self._fallback.propose(incident, capabilities, scout_observations)
        return fallback.model_copy(
            update={
                "generator": "rule_catalog_v1_after_ai_error",
                "rejected_candidates": [f"AI proposal unavailable: {reason}"],
            }
        )


class HybridProposalGenerator:
    """Conservative union: AI may add valid candidates but cannot delete rule seeds."""

    def __init__(
        self,
        ai: ProposalGenerator | None = None,
        rules: ProposalGenerator | None = None,
    ) -> None:
        self._rules = rules or RuleProposalGenerator()
        self._ai = ai or DeepSeekProposalGenerator(fallback=self._rules)

    def propose(
        self,
        incident: Incident,
        capabilities: list[ObservationCapability],
        scout_observations: list[Observation],
    ) -> CandidateProposal:
        rule_proposal = self._rules.propose(
            incident, capabilities, scout_observations
        )
        ai_proposal = self._ai.propose(incident, capabilities, scout_observations)
        if ai_proposal.generator == "rule_catalog_v1_after_ai_error":
            return rule_proposal.model_copy(
                update={
                    "generator": "hybrid_rule_only_after_ai_error",
                    "rejected_candidates": ai_proposal.rejected_candidates,
                }
            )
        return CandidateProposal(
            generator="hybrid_deepseek_plus_rule_v1",
            candidates=_deduplicate_candidates(
                [*rule_proposal.candidates, *ai_proposal.candidates]
            ),
            rejected_candidates=ai_proposal.rejected_candidates,
        )


def open_discovery_generator(excluded_candidate_ids: set[str]) -> HybridProposalGenerator:
    """Build a live hybrid where excluded mechanisms are absent from both seed paths."""
    rules = FilteredRuleProposalGenerator(excluded_candidate_ids)
    ai = DeepSeekProposalGenerator(
        fallback=rules,
        excluded_seed_candidate_ids=excluded_candidate_ids,
    )
    return HybridProposalGenerator(ai=ai, rules=rules)


_SYSTEM_PROMPT = """You propose RCA hypotheses; you never decide the root cause.
Return one JSON object with exactly these keys:
1. selected_candidate_ids: IDs selected from seed_candidates.
2. candidates: optional fully specified new Candidate objects, or an empty list.
The seed catalog is deliberately incomplete. Actively inspect the incident and
available signal names for a plausible mechanism missing from the seeds. When one is
plausible, create it in candidates rather than mapping it to a broad existing seed.
Use only listed capability IDs and their listed signal names. Every new candidate
needs its own positive evidence, falsification conditions, confounders, and
candidate-owned EvidenceConditions. When the inventory permits it, require at least
two positive conditions from distinct independent_group values and different causal
dimensions (for example duration dominance and structural shape).
Support for A is not contradiction of B. Prefer unknown over forced classification.
Do not output SQL, file paths, credentials, prose, or markdown.
"""


def _sanitized_context(
    incident: Incident,
    capabilities: list[ObservationCapability],
    scout_observations: list[Observation],
    catalog: list[Candidate],
) -> dict[str, object]:
    return {
        "incident": {
            "alert_title": incident.alert_title,
            "alert_entity": incident.alert_entity,
            "window_start": incident.window_start,
            "window_end": incident.window_end,
        },
        "capabilities": [
            {
                "capability_id": item.capability_id,
                "modality": item.modality,
                "signals": item.signals,
                "available": item.available,
            }
            for item in capabilities
        ],
        "scout_observations": [
            {
                "signal": item.signal,
                "value": item.value,
                "observed_entity": item.observed_entity,
            }
            for item in scout_observations
        ],
        "seed_candidates": [item.model_dump(mode="json") for item in catalog],
    }


def _deduplicate_candidates(candidates: list[Candidate]) -> list[Candidate]:
    unique: dict[str, Candidate] = {}
    for candidate in candidates:
        unique.setdefault(candidate.candidate_id, candidate)
    return list(unique.values())


def _numeric(value: object, minimum: float) -> bool:
    return isinstance(value, (int, float)) and float(value) >= minimum
