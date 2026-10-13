"""Host enforcement revision 2; historical transcripts without a revision retain revision 1."""
import json

from rca_agent.core.dynamic import AlternativeDraft
from rca_agent.tools.fingerprint import sha256_digest


def validate_revision(revision):
    if type(revision) is not int or revision not in {1, 2}:
        raise ValueError("unsupported engineering contract revision")
    return revision


def validate_inquiry_host(version):
    if type(version) is not int or version not in {1, 2}:
        raise ValueError("unsupported Inquiry host contract")
    return version


def inquiry_host_contract(config, *, historical=False):
    """Resolve host authority once. Old format inference is exclusively a Replay rule."""
    revision = validate_revision(config.get("engineering_contract_revision", 1))
    if "inquiry_contract_version" in config:
        return validate_inquiry_host(config["inquiry_contract_version"])
    if historical and revision == 1:
        return None  # Original format: unversioned legacy or explicitly v2, locked on first round.
    # Existing explicit rollback selectors remain supported; neither comes from a Draft.
    return 1 if revision == 1 or config.get("bound_handoff_version") == 6 else 2


def resolve_inquiry_version(version, *, host_version, recorded=False, historical=False):
    """A missing model marker inherits host authority; it never grants legacy permission."""
    if version is not None and (type(version) is not int or version not in {1, 2}):
        raise ValueError("unsupported Inquiry contract")
    if host_version is None:
        if not historical:
            raise ValueError("live Inquiry requires a host contract")
        if version == 1:
            raise ValueError("unsupported historical Inquiry marker")
        return 2 if version == 2 else 1
    validate_inquiry_host(host_version)
    if version is not None and version != host_version:
        raise ValueError("Inquiry contract differs from host")
    if recorded and host_version == 2 and version != 2:
        raise ValueError("recorded Inquiry omitted the host contract")
    if recorded and host_version == 1 and version is not None:
        raise ValueError("recorded legacy Inquiry must use its canonical unversioned format")
    return host_version


def freeze_input(draft, valid_drafts):
    """Reproduce the existing host peer-alternative projection without changing semantics."""
    existing = {a.mechanism.strip().casefold() for a in draft.alternatives}
    peers = tuple(AlternativeDraft(
        alternative_id="alt-peer-" + sha256_digest({
            "source": other.source_hypothesis_id, "mechanism": other.mechanism})[:16],
        mechanism=other.mechanism, mechanism_class=other.mechanism_class,
    ) for _, other in valid_drafts
        if other.source_hypothesis_id == draft.source_hypothesis_id
        and other.mechanism.strip().casefold() != draft.mechanism.strip().casefold()
        and other.mechanism.strip().casefold() not in existing)
    return draft.model_copy(update={"alternatives": (*draft.alternatives, *peers)})


def draft_rejection(draft, proposals, entities, seen, *, reject_self):
    from rca_agent.investigation.inquiry_contract import normalized
    key = (draft.source_hypothesis_id, draft.mechanism.strip().casefold())
    if draft.source_hypothesis_id not in proposals:
        return "UNKNOWN_SOURCE_HYPOTHESIS"
    if key in seen:
        return "DUPLICATE_MECHANISM"
    if draft.causal_entity_ref not in entities:
        return "UNISSUED_CAUSAL_ENTITY"
    if reject_self and any(normalized(a.mechanism) == normalized(draft.mechanism)
                           for a in draft.alternatives):
        return "SELF_ALTERNATIVE"
    seen.add(key)
    return None


def execution_binding_responses(events):
    """Bind only declarations preceding the matching request and response, never fingerprint alone."""
    declared = {}
    pending = None
    result = {}
    for event in events:
        kind = event.event_type.name
        if kind == "INQUIRY_ROUND_RECORDED":
            from rca_agent.investigation.inquiry_contract import binding_id
            for selected in event.payload.get("selected_actions", ()):
                identifier = selected.get("binding_id") or binding_id(
                    event.payload["round"], selected["need_id"], selected["query_ref"])
                declared[identifier] = (event.sequence, selected["action"])
        elif kind == "TOOL_REQUEST":
            pending = (event.sequence, event.payload)
        elif kind == "TOOL_RESPONSE" and pending is not None:
            sequence, request = pending
            if request["query_ref"] == event.payload["query_ref"] and sequence < event.sequence:
                for identifier, (declared_at, action) in declared.items():
                    if (declared_at < sequence and action["tool_name"] == request["tool_name"]
                            and action["request"] == request["request"]):
                        result[identifier] = event.payload
            pending = None
        elif kind == "TOOL_EXCEPTION":
            pending = None
    return result


def history_binding_responses(rounds, history, offsets):
    """Use host history positions; callers lacking positions may only claim the first offer."""
    responses = {}
    for index, record in enumerate(history):
        if record.get("result", {}).get("kind") == "tool_response":
            key = json.dumps(record["action"], sort_keys=True)
            responses.setdefault(key, []).append((index, record["result"]["value"]))
    first_offer = {}
    result = {}
    for record in rounds:
        for selected in record["selected_actions"]:
            key = json.dumps(selected["action"], sort_keys=True)
            first_offer.setdefault(key, record["round"])
            offset = offsets.get(record["round"])
            if offset is None and first_offer[key] != record["round"]:
                continue
            eligible = [response for index, response in responses.get(key, ())
                        if offset is None or index >= offset]
            if eligible:
                result[(record["round"], selected["need_id"], selected["query_ref"])] = eligible[-1]
    return result
