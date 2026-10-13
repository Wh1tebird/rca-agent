"""Queryability and intended information gain, never evidence or Core authority.

Version 2 ranks declared tasks, not inferred causes. An exact observable binding
is a planning hint; a successful query is still not a resolved discriminator.
"""
from __future__ import annotations

import re
from typing import Any

from rca_agent.core.dynamic import EvidenceRegistry
from rca_agent.investigation.models import ResolvedCapability
from rca_agent.langchain_acquisition import query_options
from rca_agent.models import CallToolAction, Observation, ResetResponse
from rca_agent.tools.fingerprint import query_fingerprint, sha256_digest


def issued_entities(bootstrap: ResetResponse, observations: tuple[Observation, ...]) -> list[dict]:
    entities = {bootstrap.alert_entity.entity_ref: bootstrap.alert_entity.model_dump(mode="json")}
    for observation in observations:
        for value in (
            observation.value if observation.signal in {"related_entity", "trace_participant"}
            else None,
            observation.details.get("neighbor"),
            observation.details.get("resolved_entity"),
            observation.details.get("resolved_service"),
        ):
            if isinstance(value, dict) and value.get("entity_ref") and value.get("entity_type"):
                entities[value["entity_ref"]] = value
    return list(entities.values())


def query_capabilities(
    bootstrap: ResetResponse, observations: tuple[Observation, ...],
    registry: EvidenceRegistry, selected_actions: list[dict],
) -> tuple[ResolvedCapability, ...]:
    """Publish only exact legal menu actions, independently of causal RuleSpec eligibility."""
    menu = query_options(
        {"bootstrap": bootstrap.model_dump(mode="json")}, issued_entities(bootstrap, observations),
    )
    tools = {str(item.tool_name): item for item in bootstrap.capabilities if item.available}
    specs = {(spec.entity_ref, spec.signal): spec for spec in registry.capability_handles()}
    result = []
    seen = set()
    for selected in selected_actions:
        action = CallToolAction.model_validate(selected["action"])
        ref = "aq-" + sha256_digest(action.model_dump(mode="json"))[:20]
        if ref not in menu or action != menu[ref]["action"]:
            continue
        fingerprint = query_fingerprint(action.tool_name, action.request)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        signal = getattr(action.request, "signal", "related_entity")
        entity_ref = action.request.entity_ref
        spec = specs.get((entity_ref, signal + "_baseline_ratio")) or specs.get((entity_ref, signal))
        group = spec.independent_group if spec is not None else "context_only"
        result.append(ResolvedCapability(
            capability_ref="cap-" + sha256_digest({"query_catalog_v2": fingerprint})[:20],
            entity_ref=entity_ref, rule_id="query_only", signal=signal,
            independent_group=group, support_eligible=False, refutes=(),
            query_fingerprint=fingerprint, tool_name=action.tool_name.value,
            query_cost=tools[action.tool_name.value].query_cost, executable=True,
            query_action=action.model_dump(mode="json"),
        ))
    return tuple(sorted(result, key=lambda item: item.capability_ref))


def scheduling_metadata(
    need: dict, capability: ResolvedCapability, bootstrap: ResetResponse,
    observations: tuple[Observation, ...], registry: EvidenceRegistry,
    binding: dict | None = None,
) -> dict[str, Any]:
    """Conservative exact signal/entity alignment. No semantic guessing from prose."""
    known_signals = {metric.signal for tool in bootstrap.capabilities for metric in tool.metrics}
    known_signals.update(spec.signal for spec in registry.signals)
    text = " ".join(need.get(key, "") for key in (
        "question", "expected_if_true", "expected_if_alternative",
    )).casefold()
    requested = need.get("observable_signals")
    signals = sorted(set(requested)) if requested else sorted(
        signal for signal in known_signals
        if re.search(r"(?<![a-z0-9_])" + re.escape(signal) + r"(?![a-z0-9_])", text)
    )
    entity = need.get("entity_ref")
    alignment = (
        "mismatch" if entity is not None and entity != capability.entity_ref
        or signals and capability.signal not in signals
        else "matched" if signals and capability.signal in signals
        else "unknown"
    )
    symptom = capability.independent_group in {"end_to_end_latency", "request_errors"}
    group_signals = {
        spec.signal for spec in registry.signals
        if spec.entity_ref == capability.entity_ref
        and spec.independent_group == capability.independent_group
    }
    observed_group = any(
        observation.entity_ref == capability.entity_ref and observation.signal in group_signals
        and observation.data_status.value in {"observed", "derived"}
        and observation.value is not None
        for observation in observations
    )
    missing = any(
        observation.entity_ref == capability.entity_ref
        and observation.signal in {capability.signal, capability.signal + "_window_comparison"}
        and observation.data_status.value in {"missing", "unknown"}
        for observation in observations
    )
    result = {
        "version": 2, "alignment": alignment, "requested_signals": signals,
        "requested_entity_ref": entity, "symptom_only": symptom,
        "entity_location_gain": capability.signal == "related_entity",
        "group_already_observed": observed_group, "missing_risk": missing,
    }
    if binding is not None and binding.get("mode"):
        result.update(version=3, mode=binding["mode"],
                      measured_signals=[capability.signal],
                      measurement_target=binding["measurement_target"])
        result["alignment"] = "matched" if binding["mode"] == "direct" else "exploratory"
    return result


def task_score(task, option, tasks) -> tuple[int, tuple[str, ...]]:
    """Fixed strategy coefficients are scheduling priorities, not evidence thresholds."""
    info = task.scheduling
    aligned = info["alignment"] == "matched"
    completed = [item for item in tasks if any(a.completed_sequence for a in item.attempts)]
    prior_groups = {
        (o.entity_ref, a.independent_group)
        for item in completed for a in item.attempts for o in item.capability_options
        if a.capability_ref == o.capability_ref and a.tool_status == "success"
    }
    seen_group = info["group_already_observed"] or (
        option.entity_ref, option.independent_group
    ) in prior_groups
    novel = not seen_group and option.independent_group != "context_only"
    pairs = {
        (item.candidate_track_id, item.alternative_id)
        for item in tasks if item.status.value == "open" and item.scheduling is not None
        and item.scheduling["alignment"] == "matched"
        and item.candidate_prediction.strip().casefold() != item.comparator_prediction.strip().casefold()
        and any(o.query_fingerprint == option.query_fingerprint for o in item.capability_options)
    } if aligned else set()
    score = (
        40 * len(pairs) + 20 * int(novel and aligned)
        + 10 * int(info["entity_location_gain"]) + 25 * int(aligned)
    ) // max(1, option.query_cost)
    score -= 25 * int(info["symptom_only"]) + 15 * int(seen_group)
    score -= 30 * int(info["alignment"] == "mismatch")
    score -= 10 * int(info["alignment"] == "unknown") + 30 * int(info["missing_risk"])
    reasons = ["QUERY_UTILITY_V2", "NEED_QUERY_" + info["alignment"].upper()]
    if pairs:
        reasons.append("DECLARED_ALTERNATIVE_COVERAGE_" + str(len(pairs)))
    if novel and aligned:
        reasons.append("NEW_INDEPENDENT_GROUP")
    if seen_group:
        reasons.append("SAME_GROUP_PENALTY")
    if info["symptom_only"]:
        reasons.append("SYMPTOM_PENALTY")
    if info["missing_risk"]:
        reasons.append("MISSING_DATA_RISK")
    if info["entity_location_gain"]:
        reasons.append("ENTITY_LOCATION_GAIN")
    return score, tuple(reasons)
