"""Bounded, semantics-preserving repair of a malformed Discovery envelope."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from rca_agent.core.dynamic import MechanismClass, MechanismDiscoveryPayload
from rca_agent.core.v3.models import MechanismFamily


class DiscoveryFormatRepairAnchor(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_hypothesis_id: str = Field(pattern=r"^h-[a-z0-9-]{1,40}$")
    causal_entity_ref: str = Field(pattern=r"^ent-[a-z0-9-]{3,40}$")
    mechanism: str = Field(min_length=3, max_length=500)
    mechanism_class: str | None = None
    family: str | None = None
    alternatives: tuple[str, ...] = ()


class DiscoveryFormatRepairContext(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    raw_payload: dict[str, Any]
    anchors: tuple[DiscoveryFormatRepairAnchor, ...] = Field(min_length=1, max_length=4)


def repair_context_from_raw(
    raw: Any,
    *,
    proposal_ids: set[str],
    issued_entities: set[str],
) -> DiscoveryFormatRepairContext | None:
    """Admit only a small structured source with unambiguous causal identity."""
    if not isinstance(raw, dict) or len(json.dumps(raw, default=str)) > 20_000:
        return None
    items = raw.get("mechanisms")
    if not isinstance(items, list) or not 1 <= len(items) <= 4:
        return None
    anchors = []
    identities: set[tuple[str, str, str]] = set()
    for item in items:
        if not isinstance(item, dict):
            return None
        source = item.get("source_hypothesis_id")
        entity = item.get("causal_entity_ref")
        mechanism = item.get("mechanism")
        if (not isinstance(source, str) or source not in proposal_ids
                or not isinstance(entity, str) or entity not in issued_entities
                or not isinstance(mechanism, str)):
            return None
        identity = (source, entity, mechanism)
        if identity in identities:
            return None
        identities.add(identity)
        original_alternatives = item.get("alternatives", ())
        if not isinstance(original_alternatives, (list, tuple)):
            return None
        alternative_texts = []
        for alternative in original_alternatives:
            if isinstance(alternative, dict) and isinstance(alternative.get("mechanism"), str):
                alternative_texts.append(alternative["mechanism"])
            else:
                return None
        try:
            anchors.append(DiscoveryFormatRepairAnchor(
                source_hypothesis_id=source,
                causal_entity_ref=entity,
                mechanism=mechanism,
                mechanism_class=(
                    item["mechanism_class"]
                    if isinstance(item.get("mechanism_class"), str)
                    and item["mechanism_class"] in MechanismClass._value2member_map_
                    else None
                ),
                family=(
                    item["family"] if isinstance(item.get("family"), str)
                    and item["family"] in MechanismFamily._value2member_map_
                    else None
                ),
                alternatives=tuple(alternative_texts),
            ))
        except ValueError:
            return None
    return DiscoveryFormatRepairContext(raw_payload=raw, anchors=tuple(anchors))


def validate_discovery_format_repair(
    payload: MechanismDiscoveryPayload,
    context: DiscoveryFormatRepairContext,
) -> None:
    """Reject dropped, added, reordered or causally changed mechanisms."""
    if len(payload.mechanisms) != len(context.anchors):
        raise ValueError("DISCOVERY_REPAIR_MECHANISM_SET_CHANGED")
    for item, anchor in zip(payload.mechanisms, context.anchors, strict=True):
        if (
            item.source_hypothesis_id != anchor.source_hypothesis_id
            or item.causal_entity_ref != anchor.causal_entity_ref
            or item.mechanism != anchor.mechanism
            or (anchor.mechanism_class is not None
                and item.mechanism_class.value != anchor.mechanism_class)
            or (anchor.family is not None and item.family.value != anchor.family)
            or not set(anchor.alternatives).issubset({
                alternative.mechanism for alternative in item.alternatives
            })
        ):
            raise ValueError("DISCOVERY_REPAIR_CAUSAL_SEMANTICS_CHANGED")
