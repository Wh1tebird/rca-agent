"""Versioned, pure measurement contracts shared by admission, Runner and Replay."""
from rca_agent.models import CallToolAction, TimeWindow
from rca_agent.tools.fingerprint import sha256_digest


def normalized(text):
    return " ".join(text.casefold().split())


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


def fail(code, field):
    from rca_agent.langchain_acquisition import InquiryOutputError
    raise InquiryOutputError(code, fields=(field,))


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
        fail("DIRECT_MEASUREMENT_MISMATCH", "needs.measurement_kind")
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
