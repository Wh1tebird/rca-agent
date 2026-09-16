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
    support_eligible: bool = True
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
    target_entity_ref: str | None = Field(default=None, pattern=r"^ent-[a-z0-9-]{3,40}$")
    polarity: ConditionPolarity
    rationale: str = Field(min_length=1, max_length=500)
    distinguishes: tuple[str, ...] = Field(default=(), max_length=8)


class DynamicHypothesis(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    source_hypothesis_id: str = Field(pattern=r"^h-[a-z0-9-]{1,40}$")
    causal_entity_ref: str | None = Field(default=None, pattern=r"^ent-[a-z0-9-]{3,40}$")
    mechanism: str = Field(min_length=3, max_length=500)
    family: MechanismFamily
    causal_explanation: str = Field(min_length=10, max_length=2000)
    predictions: tuple[Prediction, ...] = Field(min_length=2, max_length=12)
    alternatives: tuple[str, ...] = Field(min_length=1, max_length=8)

    @model_validator(mode="after")
    def nonempty_unique_alternatives(self):
        alternatives = [value.strip().casefold() for value in self.alternatives]
        if any(not value for value in alternatives) or len(set(alternatives)) != len(alternatives):
            raise ValueError("alternatives must be nonempty and distinct")
        return self


class DiscoveryPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    hypotheses: tuple[DynamicHypothesis, ...] = Field(default=(), max_length=4)


class AlternativeDraft(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    alternative_id: str = Field(pattern=r"^alt-[a-z0-9-]{1,40}$")
    mechanism: str = Field(min_length=3, max_length=500)


class MechanismDraft(BaseModel):
    """A causal proposal with no evidence-query or verification-plan fields."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    source_hypothesis_id: str = Field(pattern=r"^h-[a-z0-9-]{1,40}$")
    causal_entity_ref: str = Field(pattern=r"^ent-[a-z0-9-]{3,40}$")
    mechanism: str = Field(min_length=3, max_length=500)
    family: MechanismFamily
    causal_explanation: str = Field(min_length=10, max_length=2000)
    alternatives: tuple[AlternativeDraft, ...] = Field(min_length=1, max_length=8)

    @model_validator(mode="after")
    def unique_alternative_ids(self):
        identifiers = [item.alternative_id for item in self.alternatives]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("alternative IDs must be unique")
        return self


class MechanismDiscoveryPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    mechanisms: tuple[MechanismDraft, ...] = Field(default=(), max_length=4)


class FrozenMechanism(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    mechanism_id: str = Field(pattern=r"^mech-[a-f0-9]{20}$")
    source_hypothesis_id: str
    causal_entity_ref: str
    mechanism: str
    family: MechanismFamily
    causal_explanation: str
    alternatives: tuple[AlternativeDraft, ...]
    digest: str = Field(pattern=r"^[a-f0-9]{64}$")


def freeze_mechanism(draft: MechanismDraft) -> FrozenMechanism:
    payload = draft.model_dump(mode="json")
    digest = sha256_digest(payload)
    return FrozenMechanism(
        mechanism_id="mech-" + digest[:20],
        digest=digest,
        **payload,
    )


class CapabilityHandle(BaseModel):
    """One host-authorized Entity, Signal, Rule and query-recipe combination."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    capability_ref: str = Field(pattern=r"^cap-[a-f0-9]{20}$")
    entity_ref: str
    rule_id: str
    signal: str
    description: str
    independent_group: str
    support_eligible: bool


class PlannedPrediction(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    capability_ref: str = Field(pattern=r"^cap-[a-f0-9]{20}$")
    polarity: ConditionPolarity
    rationale: str = Field(min_length=1, max_length=500)
    distinguishes: tuple[str, ...] = Field(default=(), max_length=8)


class VerificationPlanDraft(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    mechanism_id: str = Field(pattern=r"^mech-[a-f0-9]{20}$")
    predictions: tuple[PlannedPrediction, ...] = Field(min_length=2, max_length=12)


class VerificationPlanPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    plans: tuple[VerificationPlanDraft, ...] = Field(default=(), max_length=4)


def normalize_verification_plan(
    plan: VerificationPlanDraft,
) -> tuple[VerificationPlanDraft, tuple[str, ...]]:
    """Merge repeated uses of the same capability with the same polarity."""
    grouped: dict[tuple[str, ConditionPolarity], PlannedPrediction] = {}
    changed = False
    for prediction in plan.predictions:
        key = (prediction.capability_ref, prediction.polarity)
        previous = grouped.get(key)
        if previous is None:
            grouped[key] = prediction
            continue
        changed = True
        grouped[key] = previous.model_copy(update={
            "distinguishes": tuple(dict.fromkeys((
                *previous.distinguishes, *prediction.distinguishes,
            ))),
        })
    if not changed:
        return plan, ()
    return VerificationPlanDraft(
        mechanism_id=plan.mechanism_id,
        predictions=tuple(grouped.values()),
    ), ("merged_duplicate_capability_predictions",)


def repair_mechanical_contract(
    draft: DynamicHypothesis,
) -> tuple[DynamicHypothesis, tuple[str, ...]]:
    """Canonicalize exact alternative references without changing hypothesis semantics."""

    def reference_key(value: str) -> str:
        return " ".join(value.strip().casefold().split())

    alternatives_by_key: dict[str, list[str]] = {}
    for alternative in draft.alternatives:
        alternatives_by_key.setdefault(reference_key(alternative), []).append(alternative)

    changed = False
    predictions = []
    for prediction in draft.predictions:
        repaired_references = []
        for reference in prediction.distinguishes:
            matches = alternatives_by_key.get(reference_key(reference), ())
            repaired = matches[0] if len(matches) == 1 else reference
            changed |= repaired != reference
            repaired_references.append(repaired)
        predictions.append(prediction.model_copy(update={
            "distinguishes": tuple(repaired_references),
        }))
    if not changed:
        return draft, ()
    return draft.model_copy(update={"predictions": tuple(predictions)}), (
        "canonicalized_alternative_references",
    )


class CompiledCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    candidate: Candidate
    source_hypothesis_id: str
    origin_hypothesis_id: str | None = None
    rules: tuple[RuleSpec, ...]
    signals: tuple[SignalSpec, ...]
    predictions: tuple[Prediction, ...] = ()
    alternatives: tuple[str, ...] = ()
    discovery_query_refs: tuple[str, ...] = ()
    discovery_observation_refs: tuple[str, ...] = ()
    discovery_fact_digests: tuple[str, ...] = ()
    version: str


class EvidenceRegistry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    signals: tuple[SignalSpec, ...]
    rules: tuple[RuleSpec, ...]

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

    def capability_handles(self) -> tuple[CapabilityHandle, ...]:
        handles = []
        for signal in self.signals:
            if not signal.available:
                continue
            for rule in self.rules:
                if rule.signal != signal.signal:
                    continue
                if rule.reference_signal is not None and not any(
                    reference.entity_ref == signal.entity_ref
                    and reference.signal == rule.reference_signal
                    and reference.available
                    for reference in self.signals
                ):
                    continue
                identity = {"entity_ref": signal.entity_ref, "rule_id": rule.rule_id}
                handles.append(CapabilityHandle(
                    capability_ref="cap-" + sha256_digest(identity)[:20],
                    entity_ref=signal.entity_ref,
                    rule_id=rule.rule_id,
                    signal=signal.signal,
                    description=rule.description,
                    independent_group=signal.independent_group,
                    support_eligible=rule.support_eligible,
                ))
        return tuple(handles)

    def compile_plan(
        self,
        mechanism: FrozenMechanism,
        plan: VerificationPlanDraft,
        **compile_kwargs,
    ) -> CompiledCandidate:
        if plan.mechanism_id != mechanism.mechanism_id:
            raise ValueError("verification plan references an unknown frozen mechanism")
        handles = {item.capability_ref: item for item in self.capability_handles()}
        alternatives = {
            item.alternative_id: item.mechanism for item in mechanism.alternatives
        }
        predictions = []
        distinguished: set[str] = set()
        for item in plan.predictions:
            handle = handles.get(item.capability_ref)
            if handle is None:
                raise ValueError("unknown capability reference")
            if not set(item.distinguishes) <= alternatives.keys():
                raise ValueError("unknown alternative reference")
            if item.polarity == ConditionPolarity.CONTRADICT:
                distinguished.update(item.distinguishes)
            predictions.append(Prediction(
                rule_id=handle.rule_id,
                target_entity_ref=handle.entity_ref,
                polarity=item.polarity,
                rationale=item.rationale,
                distinguishes=tuple(alternatives[value] for value in item.distinguishes),
            ))
        if not any(item.polarity == ConditionPolarity.CONTRADICT for item in plan.predictions):
            raise ValueError("verification plan requires a refutation prediction")
        if distinguished != set(alternatives):
            raise ValueError(
                "every alternative requires a dedicated refutation prediction"
            )
        draft = DynamicHypothesis(
            source_hypothesis_id=mechanism.source_hypothesis_id,
            causal_entity_ref=mechanism.causal_entity_ref,
            mechanism=mechanism.mechanism,
            family=mechanism.family,
            causal_explanation=mechanism.causal_explanation,
            predictions=tuple(predictions),
            alternatives=tuple(alternatives.values()),
        )
        return self.compile(draft, mechanism.causal_entity_ref, **compile_kwargs)

    def compile(
        self,
        draft: DynamicHypothesis,
        entity_ref: str,
        *,
        discovery_query_refs: tuple[str, ...] = (),
        discovery_observation_refs: tuple[str, ...] = (),
        discovery_fact_digests: tuple[str, ...] = (),
    ) -> CompiledCandidate:
        by_rule = {r.rule_id: r for r in self.rules}
        causal_entity_ref = draft.causal_entity_ref or entity_ref
        by_signal = {(s.entity_ref, s.signal): s for s in self.signals}
        selected: list[RuleSpec] = []
        specs: dict[tuple[str, str], SignalSpec] = {}
        prediction_entities: list[str] = []
        groups: set[str] = set()
        if len({(p.target_entity_ref or causal_entity_ref, p.rule_id)
                for p in draft.predictions}) != len(draft.predictions):
            raise ValueError("duplicate predictions")
        for prediction in draft.predictions:
            if not set(prediction.distinguishes) <= set(draft.alternatives):
                raise ValueError("discriminating prediction references an undeclared alternative")
            rule = by_rule.get(prediction.rule_id)
            if rule is None:
                raise ValueError("unknown rule")
            if prediction.polarity == ConditionPolarity.SUPPORT and not rule.support_eligible:
                raise ValueError("rule is not eligible as root-cause support")
            target_entity_ref = prediction.target_entity_ref or causal_entity_ref
            prediction_entities.append(target_entity_ref)
            for signal in (rule.signal, rule.reference_signal):
                if signal is None:
                    continue
                spec = by_signal.get((target_entity_ref, signal))
                if spec is None or not spec.available:
                    raise ValueError("signal unavailable for target entity")
                specs[(target_entity_ref, signal)] = spec
            selected.append(rule)
            if prediction.polarity == ConditionPolarity.SUPPORT:
                groups.add(by_signal[(target_entity_ref, rule.signal)].independent_group)
        if len(groups) < 2:
            raise ValueError("two host-defined independent support groups are required")
        normalized_mechanism = " ".join(
            draft.mechanism.casefold().replace("-", " ").split()
        )
        claims_full_gc = (
            "full gc" in normalized_mechanism
            or "full garbage collection" in normalized_mechanism
        )
        support_signals = {
            rule.signal
            for rule, prediction in zip(selected, draft.predictions, strict=True)
            if prediction.polarity == ConditionPolarity.SUPPORT
        }
        if claims_full_gc and not any(
            signal.startswith("full_gc_") for signal in support_signals
        ):
            raise ValueError(
                "generic GC observations cannot establish the specific Full GC subtype"
            )
        payload = {
            "draft": draft.model_dump(mode="json"),
            "entity": causal_entity_ref,
            "rules": [r.model_dump(mode="json") for r in selected],
            "signals": [s.model_dump(mode="json") for s in specs.values()],
            "discovery_query_refs": discovery_query_refs,
            "discovery_observation_refs": discovery_observation_refs,
            "discovery_fact_digests": discovery_fact_digests,
        }
        version = sha256_digest(payload)
        candidate_id = "dyn_" + version[:24]
        conditions = [
            EvidenceCondition(
                condition_id=f"{candidate_id}-{index}-{r.rule_id}",
                candidate_id=candidate_id,
                polarity=p.polarity,
                signal=r.signal,
                comparison=r.comparison,
                threshold=r.threshold if r.threshold is not None else 0,
                capability_id=by_signal[(target, r.signal)].query.tool_name.value,
                independent_group=by_signal[(target, r.signal)].independent_group,
                strength=2,
                reason=p.rationale,
                observed_entity=target,
            )
            for index, (r, p, target) in enumerate(
                zip(selected, draft.predictions, prediction_entities, strict=True)
            )
        ]
        candidate = Candidate(
            candidate_id=candidate_id,
            fault_type=draft.mechanism,
            family=draft.family,
            causal_entity=causal_entity_ref,
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
            origin_hypothesis_id=draft.source_hypothesis_id,
            rules=tuple(selected),
            signals=tuple(specs.values()),
            predictions=draft.predictions,
            alternatives=draft.alternatives,
            discovery_query_refs=discovery_query_refs,
            discovery_observation_refs=discovery_observation_refs,
            discovery_fact_digests=discovery_fact_digests,
            version=version,
        )


class DynamicVerification(BaseModel):
    """One mechanism-independent standard, with auditable test outcomes."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    standard: Literal["generic-root-cause-v1"] = "generic-root-cause-v1"
    passed: bool
    blockers: tuple[str, ...]
    support_signals: tuple[str, ...]
    refutation_signals: tuple[str, ...]
    distinguished_alternatives: tuple[str, ...]
    new_query_refs: tuple[str, ...]


def observation_fact_digest(observation: Observation) -> str:
    """Identify a fact independently of request formatting, result limits and issued IDs."""
    return sha256_digest({
        "entity_ref": observation.entity_ref, "signal": observation.signal,
        "value": observation.value, "unit": observation.unit,
        "observed_at": observation.observed_at.isoformat() if observation.observed_at else None,
        "trace_ref": observation.details.get("trace_ref"),
    })


def verify_dynamic(
    compiled: CompiledCandidate,
    assessments: tuple[EvidenceAssessment, ...],
    records,
) -> DynamicVerification:
    """Require executed support, refutation and alternative tests plus new verification evidence.

    No mechanism name is consulted. The controller still arbitrates eligible candidates.
    """
    blockers = []
    pairs = list(zip(compiled.candidate.conditions, assessments, strict=True))
    support = [(c, a) for c, a in pairs if c.polarity == ConditionPolarity.SUPPORT]
    refutations = [(c, a) for c, a in pairs if c.polarity == ConditionPolarity.CONTRADICT]
    if len({c.independent_group for c, _ in support}) < 2:
        blockers.append("insufficient_independent_support")
    if any(a.observation is None for _, a in pairs):
        blockers.append("required_observations_unavailable")
    if any(not a.matched for _, a in support):
        blockers.append("support_predictions_not_satisfied")
    if not refutations:
        blockers.append("refutation_test_absent")
    if any(a.matched for _, a in refutations):
        blockers.append("refutation_observed")
    distinguished = set()
    if len(compiled.predictions) == len(assessments):
        for prediction, (_, assessment) in zip(compiled.predictions, pairs, strict=True):
            satisfied = assessment.observation is not None and (
                assessment.matched
                if prediction.polarity == ConditionPolarity.SUPPORT
                else not assessment.matched
            )
            if satisfied:
                distinguished.update(prediction.distinguishes)
    if not compiled.alternatives or not set(compiled.alternatives) <= distinguished:
        blockers.append("alternative_explanations_unresolved")
    # A repeated query or repackaged old Observation is not a new verification fact.
    fresh_records = tuple(
        r
        for r in records
        if r.query_ref not in compiled.discovery_query_refs
        and r.observation.observation_id not in compiled.discovery_observation_refs
        and observation_fact_digest(r.observation) not in compiled.discovery_fact_digests
        and r.complete
    )
    new_queries = set()
    for index, (condition, assessment) in enumerate(pairs):
        if condition.polarity != ConditionPolarity.SUPPORT or not assessment.matched:
            continue
        reference_signal = compiled.rules[index].reference_signal
        reference_rows = tuple(r for r in records if r not in fresh_records
                               and r.observation.signal == reference_signal)
        fresh_assessment = assess_dynamic(compiled, (*fresh_records, *reference_rows))[index]
        if fresh_assessment.matched and fresh_assessment.observation is not None:
            new_queries.add(fresh_assessment.observation.query_ref)
    if not new_queries:
        blockers.append("no_new_support_verification")
    return DynamicVerification(
        passed=not blockers,
        blockers=tuple(blockers),
        support_signals=tuple(sorted(c.signal for c, a in support if a.matched)),
        refutation_signals=tuple(
            sorted(c.signal for c, a in refutations if a.observation is not None and not a.matched)
        ),
        distinguished_alternatives=tuple(sorted(distinguished)),
        new_query_refs=tuple(sorted(new_queries)),
    )


def assess_dynamic(compiled: CompiledCandidate, records) -> list[EvidenceAssessment]:
    """Evaluate only explicitly attached, entity- and query-matched issued observations."""
    from rca_agent.tools.fingerprint import query_fingerprint

    specs = {(s.entity_ref, s.signal): s for s in compiled.signals}

    def values(entity_ref, signal):
        spec = specs[(entity_ref, signal)]
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
        observed_entity = condition.observed_entity or compiled.candidate.causal_entity
        rows = values(observed_entity, rule.signal)
        threshold = rule.threshold
        references = (
            values(observed_entity, rule.reference_signal) if rule.reference_signal else []
        )
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
                observed_entity=observed_entity,
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
