"""Versioned, pure measurement contracts shared by admission, Runner and Replay."""
import re

from rca_agent.models import CallToolAction, TimeWindow
from rca_agent.tools.fingerprint import query_fingerprint, sha256_digest


def normalized(text):
    return " ".join(text.casefold().split())


def action_fingerprint(action):
    """Stable Action identity, distinct from a completed ``qry-*`` result."""
    return "act-" + sha256_digest(action.model_dump(mode="json"))[:20]


def canonicalize_need_signals(needs, advertised_signals):
    """Canonicalize one exact advertised signal embedded in explanatory text.

    This is intentionally narrower than text similarity: a value is changed only
    when exactly one host-advertised signal identifier appears as a token. Unknown
    or ambiguous prose remains unchanged and is rejected by the normal contract.
    """
    signals = tuple(dict.fromkeys(str(signal) for signal in advertised_signals if signal))
    canonical = []
    for raw_need in needs:
        need = dict(raw_need)
        values = need.get("observable_signals")
        if values:
            normalized = []
            for value in values:
                if value in signals:
                    normalized.append(value)
                    continue
                matches = [signal for signal in signals if re.search(
                    rf"(?<![A-Za-z0-9_]){re.escape(signal)}(?![A-Za-z0-9_])",
                    str(value),
                )]
                normalized.append(matches[0] if len(matches) == 1 else value)
            need["observable_signals"] = normalized
        canonical.append(need)
    return canonical


def original_alternative_id(candidate):
    return "alt-" + sha256_digest({"cause": normalized(candidate["mechanism"]),
                                   "alternative": normalized(candidate["alternative"])})[:20]


def binding_id(round_number, need_id, query_ref):
    return "binding-" + sha256_digest({"round": round_number, "need_id": need_id,
                                      "query_ref": query_ref})[:20]


def measurement_target(action):
    request = action.request
    tool = action.tool_name.value
    windows = {}
    if tool == "get_metric_series":
        kind = "series"
        windows["series"] = request.window.model_dump(mode="json")
    elif tool == "compare_metric_windows":
        kind = "window_comparison"
        incident = request.incident_window
        windows["incident"] = incident.model_dump(mode="json")
        windows["baseline"] = TimeWindow(start=incident.start - (incident.end - incident.start),
                                          end=incident.start).model_dump(mode="json")
    elif tool == "list_related_entities":
        kind = "entity_exploration"
    else:
        raise ValueError("unsupported Inquiry measurement")
    return {"entity_ref": request.entity_ref,
            "signal": getattr(request, "signal", "related_entity"),
            "measurement_kind": kind, "windows": windows}


def public_query(item):
    target = measurement_target(item["action"])
    request = item["action"].request
    bounds = {field: getattr(request, field) for field in ("max_points", "max_results")
              if hasattr(request, field)}
    if "capability_max_results" in item:
        bounds["capability_max_results"] = item["capability_max_results"]
    return {**target, "tool_name": item["action"].tool_name.value,
            "entity_type": item["entity"]["entity_type"],
            "entity_name": item["entity"].get("display_name"),
            "unit": item.get("unit") or "unknown",
            "description": item["description"],
            "limitations": item.get("limitations", {}),
            "result_limits": bounds,
            "truncation": "Results are bounded; a truncated response is partial, not complete coverage.",
            "return_shape": {"series": "samples", "window_comparison": "window statistics and ratio",
                             "entity_exploration": "related entities"}[target["measurement_kind"]]}


def _window_contains(target_windows, required_windows):
    """Return whether every required window is covered by a menu target."""
    for name, required in (required_windows or {}).items():
        target = target_windows.get(name)
        if target is None:
            return False
        target = TimeWindow.model_validate(target)
        required = TimeWindow.model_validate(required)
        if target.start < required.start or target.end > required.end:
            return False
    return True


def _matches_need(item, need):
    """Strictly match only host-owned Need fields; no text similarity is used."""
    target = measurement_target(CallToolAction.model_validate(item["action"]))
    if need.get("entity_ref") is not None and need["entity_ref"] != target["entity_ref"]:
        return False
    signals = need.get("observable_signals")
    if signals and target["signal"] not in signals:
        return False
    if (need.get("measurement_kind") is not None
            and need["measurement_kind"] != target["measurement_kind"]):
        return False
    return _window_contains(target["windows"], need.get("windows"))


def _exploration_rank(item, needs):
    """Stable partial-match rank for the bounded exploration layer."""
    action = CallToolAction.model_validate(item["action"])
    target = measurement_target(action)
    entity_refs = {need.get("entity_ref") for need in needs if need.get("entity_ref")}
    signals = {signal for need in needs for signal in (need.get("observable_signals") or ())}
    kinds = {need.get("measurement_kind") for need in needs if need.get("measurement_kind")}
    tool_rank = {"list_related_entities": 0, "compare_metric_windows": 1,
                 "get_metric_series": 2}.get(action.tool_name.value, 9)
    return (
        0 if target["entity_ref"] in entity_refs else 1,
        0 if target["signal"] in signals else 1,
        0 if target["measurement_kind"] in kinds else 1,
        tool_rank,
        query_fingerprint(action.tool_name, action.request),
    )


def project_inquiry_menu(
    full_menu: dict[str, dict],
    needs=(),
    *,
    existing_query_fingerprints=(),
    budget_remaining: int | None = None,
    min_focused: int = 8,
    max_focused: int = 20,
    exploration_limit: int = 8,
) -> dict[str, object]:
    """Project a deterministic, host-owned menu without creating any Action.

    ``focused`` contains only candidates that satisfy every non-empty Need field.
    ``exploration`` is a small fallback layer ranked by exact host fields.  The
    returned items are references to entries in ``full_menu`` and preserve their
    original actions and fingerprints byte-for-byte.
    """
    if not 1 <= min_focused <= max_focused or not 1 <= exploration_limit <= max_focused:
        raise ValueError("invalid Inquiry menu projection bounds")
    existing = set(existing_query_fingerprints)
    eligible: list[tuple[str, dict]] = []
    for ref, item in full_menu.items():
        action = CallToolAction.model_validate(item["action"])
        fingerprint = query_fingerprint(action.tool_name, action.request)
        if fingerprint in existing:
            continue
        if budget_remaining is not None and item.get("query_cost", 0) > budget_remaining:
            continue
        eligible.append((ref, item))

    open_needs = tuple(need for need in needs if need.get("disposition", "open") == "open")
    focused = [(ref, item) for ref, item in eligible
               if open_needs and any(_matches_need(item, need) for need in open_needs)]
    focused.sort(key=lambda pair: (
        _exploration_rank(pair[1], open_needs),
        pair[0],
    ))
    exploration = sorted(eligible, key=lambda pair: (_exploration_rank(pair[1], open_needs), pair[0]))

    focused_items = dict(focused[:max_focused])
    exploration_items = dict(exploration[:exploration_limit])
    if focused_items:
        exposed = focused_items
        mode = "focused" if len(focused_items) >= min_focused else "focused_with_exploration"
    else:
        exposed = exploration_items
        mode = "exploration"
    return {
        "focused": focused_items,
        "exploration": exploration_items,
        "exposed": exposed,
        "mode": mode,
        "eligible_count": len(eligible),
        "focused_count": len(focused),
        "exploration_count": len(exploration_items),
        "min_focused": min_focused,
        "max_focused": max_focused,
    }


def fail(code, field, *, supplied=None, expected_format=None, details=None):
    from rca_agent.langchain_acquisition import InquiryOutputError
    raise InquiryOutputError(code, fields=(field,), supplied=supplied,
                             expected_format=expected_format, details=details)


def validate_choice(need, choice, action):
    target = measurement_target(action)
    if choice.mode not in {"direct", "exploratory"}:
        fail("BINDING_MODE_REQUIRED", "choices.mode")
    if not choice.purpose or not choice.purpose.strip():
        fail("BINDING_PURPOSE_REQUIRED", "choices.purpose")
    if choice.measurement_target is None or choice.measurement_target.model_dump(mode="json") != target:
        fail("MEASUREMENT_TARGET_MISMATCH", "choices.measurement_target")
    if choice.mode == "exploratory":
        return
    if need.entity_ref is None or need.entity_ref != target["entity_ref"]:
        fail("DIRECT_ENTITY_MISMATCH", "needs.entity_ref")
    if not need.observable_signals or target["signal"] not in need.observable_signals:
        fail("DIRECT_SIGNAL_MISMATCH", "needs.observable_signals")
    if need.measurement_kind and need.measurement_kind != target["measurement_kind"]:
        fail("DIRECT_MEASUREMENT_MISMATCH", "needs.measurement_kind",
             supplied=need.measurement_kind, expected_format=target["measurement_kind"],
             details=("Need measurement_kind must exactly match the selected direct query's "
                      f"measurement_kind={target['measurement_kind']}"))
    if need.windows is not None:
        for name, required in need.windows.items():
            window = target["windows"].get(name)
            if window is None:
                fail("DIRECT_WINDOW_MISMATCH", "needs.windows")
            actual = TimeWindow.model_validate(window)
            if actual.start < required.start or actual.end > required.end:
                fail("DIRECT_WINDOW_MISMATCH", "needs.windows")


def validate_draft(draft):
    ids = [h.hypothesis_id for h in draft.hypotheses]
    needs = [n.need_id for n in draft.needs]
    if len(ids) != len(set(ids)) or len(needs) != len(set(needs)):
        fail("DUPLICATE_IDENTIFIERS", "needs.need_id")
    if any(n.hypothesis_id not in ids for n in draft.needs):
        fail("UNKNOWN_HYPOTHESIS", "needs.hypothesis_id")
    if any(normalized(h.mechanism) == normalized(h.alternative) for h in draft.hypotheses):
        fail("SELF_ALTERNATIVE", "hypotheses.alternative")


def selected_bindings(draft, options, actions, round_number):
    result = []
    for choice in draft.choices:
        action = options[choice.query_ref]["action"]
        if action not in actions:
            continue
        item = {"need_id": choice.need_id, "query_ref": choice.query_ref,
                "signal": getattr(action.request, "signal", "related_entity"),
                "entity_ref": action.request.entity_ref, "action": action.model_dump(mode="json")}
        if draft.contract_version == 2:
            item.update(binding_id=binding_id(round_number, choice.need_id, choice.query_ref),
                        mode=choice.mode, purpose=choice.purpose,
                        measurement_target=choice.measurement_target.model_dump(mode="json"))
        result.append(item)
    return result


def validate_bindings(draft, raw, round_number, menu, *, max_bindings=2):
    validate_draft(draft)
    if any(h.alternative_id != original_alternative_id(h.model_dump()) for h in draft.hypotheses):
        fail("ALTERNATIVE_ID_MISMATCH", "hypotheses.alternative_id")
    keys = [(c.need_id, c.query_ref) for c in draft.choices]
    if len(keys) != len(set(keys)):
        fail("INVALID_BINDINGS", "choices")
    choices = dict(zip(keys, draft.choices, strict=True))
    needs = {n.need_id: n for n in draft.needs}
    if not raw or len(raw) > max_bindings:
        fail("INVALID_BINDINGS", "choices")
    seen = set()
    for item in raw:
        key = (item.get("need_id"), item.get("query_ref"))
        choice = choices.get(key)
        if choice is None or key in seen or key[0] not in needs:
            fail("INVALID_BINDINGS", "choices.query_ref")
        seen.add(key)
        action = CallToolAction.model_validate(item["action"])
        if key[1] not in menu or action != menu[key[1]]["action"]:
            fail("ACTION_NOT_OFFERED", "choices.query_ref")
        expected = selected_bindings(draft, menu, [action], round_number)
        if item not in expected:
            fail("BINDING_FIELDS_MISMATCH", "choices.measurement_target")
        validate_choice(needs[key[0]], choice, action)
    # A budget-limited prefix may omit later physical queries, never a shared binding.
    refs = {item["query_ref"] for item in raw}
    if seen != {key for key in choices if key[1] in refs}:
        fail("MISSING_SHARED_BINDING", "choices")
    return raw


def validate_handoff_links(handoff, rounds, responses, *, binding_responses=None):
    from rca_agent.langchain_acquisition import inquiry_candidate_records
    from rca_agent.tools.fingerprint import query_fingerprint

    candidates = inquiry_candidate_records(rounds)
    ids = {(normalized(c.hypothesis.mechanism), normalized(c.hypothesis.alternative)):
           c.hypothesis.hypothesis_id for c in candidates}
    expected = {}
    for record in rounds:
        if record["draft"].get("contract_version") != 2:
            continue
        hypotheses = {h["hypothesis_id"]: h for h in record["draft"]["hypotheses"]}
        needs = {n["need_id"]: n for n in record["draft"]["needs"]}
        for selected in record["selected_actions"]:
            action = CallToolAction.model_validate(selected["action"])
            ref = query_fingerprint(action.tool_name, action.request)
            response = (responses.get(ref) if binding_responses is None else
                        binding_responses.get(selected["binding_id"]))
            if response is None:
                continue
            need = needs[selected["need_id"]]
            hypothesis = hypotheses[need["hypothesis_id"]]
            expected[selected["binding_id"]] = {
                "binding_id": selected["binding_id"], "source_round": record["round"],
                "source_need_id": need["need_id"], "mode": selected["mode"],
                "measurement_target": selected["measurement_target"], "query_ref": ref,
                "question": need["question"], "signal": selected["signal"],
                "entity_ref": selected["entity_ref"],
                "hypothesis_id": ids[(normalized(hypothesis["mechanism"]), normalized(hypothesis["alternative"]))]}
    actual = {}
    for link in handoff.query_links:
        if link.binding_id is None:
            continue
        raw = link.model_dump(mode="json", exclude={"status", "observation_refs"})
        if link.binding_id in actual or expected.get(link.binding_id) != raw:
            raise ValueError("handoff measurement binding changed")
        if binding_responses is not None:
            response = binding_responses[link.binding_id]
            if (link.status != response["status"] or set(link.observation_refs) != {
                    observation["observation_id"] for observation in response["observations"]}):
                raise ValueError("handoff execution response changed")
        actual[link.binding_id] = raw
    if actual != expected:
        raise ValueError("handoff omitted accepted measurement binding")
