"""Safe, deterministic plan diagnostics. No thresholds, secrets, or mechanism rewrites."""
from __future__ import annotations

from types import SimpleNamespace

from rca_agent.core.dynamic import EvidenceRegistry


def plan_feedback(mechanism, plan, handles, error_code: str) -> dict:
    registry = SimpleNamespace(capability_handles=lambda: handles)
    try:
        normalized, repairs = EvidenceRegistry.normalize_plan(registry, mechanism, plan)
    except ValueError:
        unique = {(p.capability_ref, p.polarity): p for p in plan.predictions}
        normalized = plan.model_copy(update={"predictions": tuple(unique.values())})
        repairs = ("merged_duplicate_capability_predictions",)
    by_ref = {handle.capability_ref: handle for handle in handles}
    errors = []
    for index, prediction in enumerate(plan.predictions):
        handle = by_ref.get(prediction.capability_ref)
        codes = []
        if handle is None:
            codes.append("UNKNOWN_CAPABILITY_REF")
        elif prediction.polarity.value == "support":
            if not handle.support_eligible:
                codes.append("SUPPORT_RULE_NOT_ELIGIBLE")
            if len(handle.refutes) == 1 and mechanism.mechanism_class not in handle.refutes:
                codes.append("CROSS_MECHANISM_SUPPORT")
            if prediction.distinguishes:
                codes.append("SUPPORT_CANNOT_REFUTE")
        else:
            if len(handle.refutes) == 1 and mechanism.mechanism_class in handle.refutes:
                codes.append("SELF_REFUTATION")
            alternatives = {item.alternative_id: item for item in mechanism.alternatives}
            if any(ref not in alternatives for ref in prediction.distinguishes):
                codes.append("UNKNOWN_ALTERNATIVE_REF")
            elif any(alternatives[ref].mechanism_class not in handle.refutes
                     for ref in prediction.distinguishes):
                codes.append("REFUTATION_SCOPE_MISMATCH")
        if codes:
            errors.append({"prediction_index": index, "capability_ref": prediction.capability_ref,
                           "error_codes": codes})
    supports = [p for p in normalized.predictions if p.polarity.value == "support"]
    if not supports:
        error_code = "NO_VALID_SUPPORT"
    elif len(normalized.predictions) < 2:
        error_code = "INSUFFICIENT_VALID_PREDICTIONS"
    valid_supports = [h for h in handles if h.entity_ref == mechanism.causal_entity_ref
                     and h.support_eligible
                     and (len(h.refutes) != 1 or mechanism.mechanism_class in h.refutes)]
    return {
        "version": 1, "error_code": error_code, "prediction_errors": errors,
        "normalization_repairs": list(repairs),
        "retained_prediction_count": len(normalized.predictions),
        "retained_support_count": len(supports),
        "eligible_support_capabilities": [
            {"capability_ref": h.capability_ref, "signal": h.signal,
             "independent_group": h.independent_group}
            for h in valid_supports[:16]
        ],
        "unobservable_alternative_ids": [a.alternative_id for a in mechanism.alternatives
                                        if not any(a.mechanism_class in h.refutes for h in handles)],
        "repair_instruction": (
            "Correct the listed roles using advertised capabilities. Do not resubmit the same "
            "invalid plan. Missing alternative telemetry must remain unresolved; never invent "
            "evidence, change the mechanism, or rewrite thresholds."
        ),
    }
