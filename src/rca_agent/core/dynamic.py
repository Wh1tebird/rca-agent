"""Host-owned signal contracts and a bounded compiler for open hypotheses.

These contracts describe observations and predicates, not a catalog of fault names.
The model selects predicates; it cannot author thresholds or evidence weights.
"""

from __future__ import annotations

import math
from itertools import pairwise
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rca_agent.core.evidence_control import EvidenceAssertion, EvidencePolarity
from rca_agent.core.v3.models import (
    Candidate,
    Comparison,
    ConditionPolarity,
    EvidenceAssessment,
    EvidenceCondition,
    MechanismFamily,
)
from rca_agent.core.v3.models import (
    Observation as CoreObservation,
)
from rca_agent.models import CallToolAction, DataStatus, Observation
from rca_agent.tools.fingerprint import sha256_digest


class SignalSpec(BaseModel):
    """Host supplies an executable, bounded recipe for one entity and signal."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    signal: str = Field(min_length=1, max_length=120)
    entity_ref: str = Field(pattern=r"^ent-[a-z0-9-]{3,40}$")
    data_type: Literal["number", "string"]
    unit: str | None = None
    description: str = Field(min_length=1, max_length=500)
    independent_group: str = Field(min_length=1)
    query: CallToolAction
    available: bool = True
    unavailable_reason: str | None = None
    minimum: float | None = None
    maximum: float | None = None

    @model_validator(mode="after")
    def valid_range(self):
        if self.minimum is not None and not math.isfinite(self.minimum):
            raise ValueError("non-finite minimum")
        if self.maximum is not None and not math.isfinite(self.maximum):
            raise ValueError("non-finite maximum")
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ValueError("inverted range")
        return self

    def accepts(self, observation: Observation) -> bool:
        request = self.query.request
        window = getattr(request, "window", getattr(request, "incident_window", None))
        if (
            observation.observed_at is not None
            and window is not None
            and (
                observation.observed_at.tzinfo is None
                or not (window.start <= observation.observed_at <= window.end)
            )
        ):
            return False
        if (
            observation.entity_ref != self.entity_ref
            or observation.signal != self.signal
            or observation.unit != self.unit
            or observation.data_status not in {DataStatus.OBSERVED, DataStatus.DERIVED}
        ):
            return False
        value = observation.value
        if self.data_type == "string":
            return isinstance(value, str)
        return (
            type(value) in (int, float)
            and math.isfinite(value)
            and (self.minimum is None or value >= self.minimum)
            and (self.maximum is None or value <= self.maximum)
        )


class RuleSpec(BaseModel):
    """An operator with a host-reviewed threshold source and explicit sample semantics."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    rule_id: str = Field(pattern=r"^[a-z][a-z0-9_]{1,63}$")
    signal: str
    comparison: Comparison
    threshold: float | str | None = None
    source: Literal["configuration", "slo", "baseline", "capacity"]
    provenance: str = Field(min_length=1, max_length=500)
    description: str = Field(
        default="Host-defined observable predicate", min_length=1, max_length=500
    )
    reference_signal: str | None = None
    factor: float = Field(default=1.0, gt=0, allow_inf_nan=False)
    aggregation: Literal["single", "max", "sustained"] = "single"
    min_samples: int = Field(default=1, ge=1)
    min_duration_seconds: float = Field(default=0, ge=0, allow_inf_nan=False)
    max_gap_seconds: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    minimum_fraction: float = Field(default=1, gt=0, le=1)

    @model_validator(mode="after")
    def threshold_source(self):
        if (self.threshold is None) == (self.reference_signal is None):
            raise ValueError("exactly one threshold source is required")
        if self.reference_signal == self.signal:
            raise ValueError("a signal cannot serve as its own baseline or capacity")
        if self.source in {"baseline", "capacity"} and self.reference_signal is None:
            raise ValueError("baseline and capacity require an observed reference")
        if isinstance(self.threshold, float) and not math.isfinite(self.threshold):
            raise ValueError("non-finite threshold")
        if self.aggregation == "sustained" and (
            self.min_samples < 2 or self.min_duration_seconds <= 0 or self.max_gap_seconds is None
        ):
            raise ValueError("sustained evidence requires timestamped coverage")
        return self


class Prediction(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    rule_id: str
    polarity: ConditionPolarity
    rationale: str = Field(min_length=1, max_length=500)


class DynamicHypothesis(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    source_hypothesis_id: str = Field(pattern=r"^h-[a-z0-9-]{1,40}$")
    mechanism: str = Field(min_length=3, max_length=500)
    family: MechanismFamily
    causal_explanation: str = Field(min_length=10, max_length=2000)
    predictions: tuple[Prediction, ...] = Field(min_length=2, max_length=12)
    alternatives: tuple[str, ...] = Field(min_length=1, max_length=8)


class DiscoveryPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    hypotheses: tuple[DynamicHypothesis, ...] = Field(default=(), max_length=4)


class ConfirmationContract(BaseModel):
    """Optional host-owned sufficiency contract; generic anomalies alone cannot confirm."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    mechanism: str
    required_support: frozenset[str]
    required_refutations: frozenset[str] = frozenset()
    provenance: str = Field(min_length=1)


class CompiledCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    candidate: Candidate
    source_hypothesis_id: str
    rules: tuple[RuleSpec, ...]
    signals: tuple[SignalSpec, ...]
    confirmation: ConfirmationContract | None = None
    version: str


class EvidenceRegistry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    signals: tuple[SignalSpec, ...]
    rules: tuple[RuleSpec, ...]
    confirmations: tuple[ConfirmationContract, ...] = ()

    @model_validator(mode="after")
    def validate_contracts(self):
        if len({(s.entity_ref, s.signal) for s in self.signals}) != len(self.signals):
            raise ValueError("duplicate signal contract")
        if len({r.rule_id for r in self.rules}) != len(self.rules):
            raise ValueError("duplicate rule ID")
        for rule in self.rules:
            targets = [s for s in self.signals if s.signal == rule.signal]
            if not targets:
                raise ValueError("rule refers to an unknown signal")
            for spec in targets:
                if spec.data_type == "string":
                    if (
                        rule.comparison not in {Comparison.EQ, Comparison.CONTAINS}
                        or not isinstance(rule.threshold, str)
                        or rule.aggregation != "single"
                    ):
                        raise ValueError("string rule type mismatch")
                elif rule.threshold is not None:
                    if not isinstance(rule.threshold, (int, float)):
                        raise ValueError("numeric rule type mismatch")
                    if (spec.minimum is not None and rule.threshold < spec.minimum) or (
                        spec.maximum is not None and rule.threshold > spec.maximum
                    ):
                        raise ValueError("threshold outside signal range")
                if rule.comparison == Comparison.CONTAINS and spec.data_type != "string":
                    raise ValueError("contains requires a string")
                if rule.reference_signal:
                    reference = next(
                        (
                            s
                            for s in self.signals
                            if s.entity_ref == spec.entity_ref and s.signal == rule.reference_signal
                        ),
                        None,
                    )
                    if (
                        reference is None
                        or reference.data_type != "number"
                        or (reference.unit != spec.unit or spec.data_type != "number")
                    ):
                        raise ValueError("reference signal type or unit mismatch")
        return self

    def compile(self, draft: DynamicHypothesis, entity_ref: str) -> CompiledCandidate:
        by_rule = {r.rule_id: r for r in self.rules}
        by_signal = {s.signal: s for s in self.signals if s.entity_ref == entity_ref}
        selected: list[RuleSpec] = []
        specs: dict[str, SignalSpec] = {}
        groups: set[str] = set()
        if len({p.rule_id for p in draft.predictions}) != len(draft.predictions):
            raise ValueError("duplicate predictions")
        for prediction in draft.predictions:
            rule = by_rule.get(prediction.rule_id)
            if rule is None:
                raise ValueError("unknown rule")
            for signal in (rule.signal, rule.reference_signal):
                if signal is None:
                    continue
                spec = by_signal.get(signal)
                if spec is None or not spec.available:
                    raise ValueError("signal unavailable for target entity")
                specs[signal] = spec
            selected.append(rule)
            if prediction.polarity == ConditionPolarity.SUPPORT:
                groups.add(by_signal[rule.signal].independent_group)
        if len(groups) < 2:
            raise ValueError("two host-defined independent support groups are required")
        payload = {
            "draft": draft.model_dump(mode="json"),
            "entity": entity_ref,
            "rules": [r.model_dump(mode="json") for r in selected],
            "signals": [s.model_dump(mode="json") for s in specs.values()],
        }
        version = sha256_digest(payload)
        candidate_id = "dyn_" + version[:24]
        conditions = [
            EvidenceCondition(
                condition_id=f"{candidate_id}-{r.rule_id}",
                candidate_id=candidate_id,
                polarity=p.polarity,
                signal=r.signal,
                comparison=r.comparison,
                threshold=r.threshold if r.threshold is not None else 0,
                capability_id=by_signal[r.signal].query.tool_name.value,
                independent_group=by_signal[r.signal].independent_group,
                strength=2,
                reason=p.rationale,
            )
            for r, p in zip(selected, draft.predictions, strict=True)
        ]
        support = {p.rule_id for p in draft.predictions if p.polarity == ConditionPolarity.SUPPORT}
        refutations = {
            p.rule_id for p in draft.predictions if p.polarity == ConditionPolarity.CONTRADICT
        }
        confirmation = next(
            (
                c
                for c in self.confirmations
                if c.mechanism == draft.mechanism
                and len(c.required_support) >= 2
                and c.required_support <= support
                and c.required_refutations <= refutations
            ),
            None,
        )
        candidate = Candidate(
            candidate_id=candidate_id,
            fault_type=draft.mechanism,
            family=draft.family,
            causal_entity=entity_ref,
            proposal_reason=draft.causal_explanation,
            positive_evidence_requirements=[
                c.condition_id for c in conditions if c.polarity == ConditionPolarity.SUPPORT
            ],
            falsification_conditions=[
                p.rationale for p in draft.predictions if p.polarity == ConditionPolarity.CONTRADICT
            ]
            or ["Required prediction observed but not satisfied"],
            confounders=list(draft.alternatives),
            required_capabilities=sorted({s.query.tool_name.value for s in specs.values()}),
            conditions=conditions,
            knowledge_provenance=f"dynamic-compiler-v1:{version}",
        )
        return CompiledCandidate(
            candidate=candidate,
            source_hypothesis_id=draft.source_hypothesis_id,
            rules=tuple(selected),
            signals=tuple(specs.values()),
            confirmation=confirmation,
            version=version,
        )


def assess_dynamic(compiled: CompiledCandidate, records) -> list[EvidenceAssessment]:
    """Evaluate only explicitly attached, entity- and query-matched issued observations."""
    from rca_agent.tools.fingerprint import query_fingerprint

    specs = {s.signal: s for s in compiled.signals}

    def values(signal):
        spec = specs[signal]
        fingerprint = query_fingerprint(spec.query.tool_name, spec.query.request)
        return [
            r
            for r in records
            if r.query_ref == fingerprint
            and getattr(r, "complete", True)
            and spec.accepts(r.observation)
        ]

    assessments = []
    for condition, rule in zip(compiled.candidate.conditions, compiled.rules, strict=True):
        rows = values(rule.signal)
        threshold = rule.threshold
        references = values(rule.reference_signal) if rule.reference_signal else []
        if rule.reference_signal:
            threshold = (
                references[0].observation.value * rule.factor
                if len(references) == 1 and references[0].observation.value > 0
                else None
            )
        unavailable = not rows or threshold is None or len(rows) < rule.min_samples
        if rule.aggregation == "single" and len(rows) != 1:
            unavailable = True
        if rule.aggregation == "sustained" and rows:
            times = [r.observation.observed_at for r in rows]
            unavailable |= any(t is None for t in times) or len(set(times)) != len(times)
            if not unavailable:
                unavailable |= (max(times) - min(times)).total_seconds() < rule.min_duration_seconds
                ordered = sorted(times)
                unavailable |= any(
                    (b - a).total_seconds() > rule.max_gap_seconds for a, b in pairwise(ordered)
                )
        matched = False
        observation = None
        assertion = None
        if not unavailable:
            observed = [r.observation.value for r in rows]
            if rule.aggregation == "sustained":
                matched = (
                    sum(_compare(rule.comparison, v, threshold) for v in observed) / len(observed)
                    >= rule.minimum_fraction
                )
                value = observed[0]
            else:
                value = max(observed) if rule.aggregation == "max" else observed[0]
                matched = _compare(rule.comparison, value, threshold)
            observation = CoreObservation(
                signal=rule.signal,
                value=value,
                query_ref=rows[0].query_ref,
                observed_entity=compiled.candidate.causal_entity,
            )
            if matched:
                assertion = EvidenceAssertion(
                    evidence_id="v3-" + condition.condition_id,
                    candidate=condition.candidate_id,
                    polarity=EvidencePolarity(condition.polarity),
                    strength=2,
                    independent_group=condition.independent_group,
                    explanation=condition.reason,
                    query_ref=rows[0].query_ref,
                )
        assessments.append(
            EvidenceAssessment(
                condition_id=condition.condition_id,
                candidate_id=condition.candidate_id,
                matched=matched,
                assertion=assertion,
                observation=observation,
                unavailable_reason="missing, invalid or insufficient bounded observations"
                if unavailable
                else None,
            )
        )
    return assessments


def _compare(operator: Comparison, value, threshold) -> bool:
    if operator == Comparison.GTE:
        return value >= threshold
    if operator == Comparison.LTE:
        return value <= threshold
    if operator == Comparison.CONTAINS:
        return threshold in value
    return value == threshold
