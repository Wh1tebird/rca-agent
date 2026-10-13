"""Versioned semantic handoff integrity. References are provenance, not causal proof."""
from collections import Counter

from rca_agent.core.dynamic import InquiryDisposition, MechanismDiscoveryPayload


def normalized(text: str) -> str:
    return " ".join(text.casefold().split())


def validate_handoff(payload: MechanismDiscoveryPayload, handoff: dict, observations: list[dict]):
    """Never infer semantic equivalence or adjudicate a model's causal explanation."""
    candidates = {h["hypothesis_id"]: h for h in handoff.get("hypotheses", ())}
    observed = {o["observation_id"] for o in observations
                if o.get("data_status") in {"observed", "derived"} and o.get("value") is not None}
    names = Counter(normalized(m.mechanism) for m in payload.mechanisms)
    dispositions = Counter(d.hypothesis_id for d in payload.inquiry_dispositions)
    errors = []
    linked = set()

    def error(code, identifier=None, mechanism=None):
        errors.append({"code": code, "hypothesis_id": identifier, "mechanism": mechanism})

    def basis(refs, identifier, mechanism):
        if not refs:
            error("MISSING_OBSERVATION_BASIS", identifier, mechanism)
        elif len(set(refs)) != len(refs) or not set(refs).issubset(observed):
            error("INVALID_OBSERVATION_BASIS", identifier, mechanism)

    for identifier in sorted(candidates):
        if dispositions[identifier] != 1:
            error("CANDIDATE_NOT_ACCOUNTED_ONCE", identifier)
    for item in payload.inquiry_dispositions:
        identifier = item.hypothesis_id
        if identifier not in candidates:
            error("UNKNOWN_INQUIRY_CANDIDATE", identifier, item.formal_mechanism)
            continue
        original = normalized(candidates[identifier]["mechanism"])
        formal = normalized(item.formal_mechanism) if item.formal_mechanism else None
        if item.disposition in {"kept", "revised"}:
            if formal is None or names[formal] != 1:
                error("FORMAL_MECHANISM_NOT_UNIQUE", identifier, item.formal_mechanism)
            else:
                linked.add(formal)
        elif formal is not None:
            error("INACTIVE_CANDIDATE_HAS_FORMAL_LINK", identifier, item.formal_mechanism)
        if item.disposition == "kept":
            if formal != original:
                error("KEPT_CAUSAL_SEMANTICS_CHANGED", identifier, item.formal_mechanism)
            if item.original_disposition is not None:
                error("UNEXPECTED_ORIGINAL_DISPOSITION", identifier, item.formal_mechanism)
        elif item.disposition == "revised":
            basis(item.basis_observation_refs, identifier, item.formal_mechanism)
            if formal == original:
                error("REVISION_HAS_UNCHANGED_MECHANISM", identifier, item.formal_mechanism)
            if item.original_disposition not in {"retained", "deferred"}:
                error("REVISION_ORIGINAL_NOT_ACCOUNTED", identifier, item.formal_mechanism)
            elif item.original_disposition == "retained":
                if names[original] != 1:
                    error("REVISION_DROPPED_ORIGINAL", identifier, item.formal_mechanism)
                else:
                    linked.add(original)
            elif names[original]:
                error("DEFERRED_ORIGINAL_WAS_FORMALIZED", identifier, item.formal_mechanism)
        else:
            if names[original]:
                error("INACTIVE_CANDIDATE_WAS_FORMALIZED", identifier, candidates[identifier]["mechanism"])
            if item.disposition == "discarded":
                basis(item.basis_observation_refs, identifier, None)
            if item.original_disposition is not None:
                error("UNEXPECTED_ORIGINAL_DISPOSITION", identifier)
        if item.basis_observation_refs is not None:
            basis(item.basis_observation_refs, identifier, item.formal_mechanism)

    new_counts = Counter(normalized(n.mechanism) for n in payload.new_mechanisms or ())
    for new in payload.new_mechanisms or ():
        key = normalized(new.mechanism)
        matching = [m for m in payload.mechanisms if normalized(m.mechanism) == key
                    and m.source_hypothesis_id == new.source_hypothesis_id]
        if key in linked or len(matching) != 1 or new_counts[key] != 1:
            error("INVALID_NEW_MECHANISM_LINK", mechanism=new.mechanism)
        basis(new.basis_observation_refs, None, new.mechanism)
        linked.add(key)
    for mechanism in payload.mechanisms:
        if names[normalized(mechanism.mechanism)] != 1:
            error("DUPLICATE_CAUSAL_MECHANISM", mechanism=mechanism.mechanism)
        if normalized(mechanism.mechanism) not in linked:
            error("MECHANISM_WITHOUT_PROVENANCE", mechanism=mechanism.mechanism)
    # Repeated provenance checks produce one stable diagnostic per field/code.
    return tuple(dict.fromkeys((e["code"], e["hypothesis_id"], e["mechanism"]) for e in errors))


def diagnostic_errors(errors):
    return [{"code": code, "hypothesis_id": identifier, "mechanism": mechanism}
            for code, identifier, mechanism in errors]


def repair_errors(initial, repaired, handoff, observations, *, recover_exact=False):
    errors = list(validate_handoff(repaired, handoff, observations))
    preserved = quarantine_handoff(initial, handoff, observations, recover_exact=recover_exact)
    by_name = {normalized(m.mechanism): m for m in repaired.mechanisms}
    for mechanism in preserved.mechanisms:
        if by_name.get(normalized(mechanism.mechanism)) != mechanism:
            errors.append(("REPAIR_CHANGED_VALID_MECHANISM", None, mechanism.mechanism))
    initial_errors = validate_handoff(initial, handoff, observations)
    bad_ids = {identifier for _, identifier, _ in initial_errors if identifier is not None}
    by_id = {d.hypothesis_id: d for d in repaired.inquiry_dispositions}
    for disposition in initial.inquiry_dispositions:
        if disposition.hypothesis_id not in bad_ids and (
            by_id.get(disposition.hypothesis_id) != disposition
        ):
            errors.append(("REPAIR_CHANGED_VALID_DISPOSITION", disposition.hypothesis_id, None))
    return tuple(dict.fromkeys(errors))


def recover_exact_links(payload, handoff):
    """Repair bookkeeping only when one existing draft exactly matches one original idea.

    Never rewrite a draft, infer synonyms, override an explicit revision/discard,
    or use observations to guess a cause. This does not grant evidence authority.
    """
    candidates = handoff["hypotheses"]
    original_counts = Counter(normalized(c["mechanism"]) for c in candidates)
    draft_counts = Counter(normalized(m.mechanism) for m in payload.mechanisms)
    dispositions = Counter(d.hypothesis_id for d in payload.inquiry_dispositions)
    by_id = {c["hypothesis_id"]: c for c in candidates}
    recovered = []
    for item in payload.inquiry_dispositions:
        candidate = by_id.get(item.hypothesis_id)
        if (candidate is not None and dispositions[item.hypothesis_id] == 1
                and item.disposition in {"kept", "deferred"}
                and original_counts[normalized(candidate["mechanism"])] == 1
                and draft_counts[normalized(candidate["mechanism"])] == 1
                and not any(d.hypothesis_id != item.hypothesis_id
                            and d.formal_mechanism
                            and normalized(d.formal_mechanism) == normalized(candidate["mechanism"])
                            for d in payload.inquiry_dispositions)
                and not any(normalized(n.mechanism) == normalized(candidate["mechanism"])
                            for n in payload.new_mechanisms or ())):
            recovered.append(InquiryDisposition(
                hypothesis_id=item.hypothesis_id, disposition="kept",
                formal_mechanism=candidate["mechanism"],
                reason="Host restored the unique exact original-to-draft link; this is bookkeeping, not evidence support.",
            ))
        else:
            recovered.append(item)
    return payload.model_copy(update={"inquiry_dispositions": tuple(recovered)})


def quarantine_handoff(payload, handoff, observations, *, recover_exact=False):
    """Keep independently valid outputs; defer invalid ideas without overwriting their meaning."""
    if recover_exact:
        payload = recover_exact_links(payload, handoff)
    errors = validate_handoff(payload, handoff, observations)
    bad_ids = {identifier for _, identifier, _ in errors if identifier is not None}
    bad_names = {normalized(name) for _, _, name in errors if name}
    for disposition in payload.inquiry_dispositions:
        if disposition.hypothesis_id in bad_ids and disposition.formal_mechanism:
            bad_names.add(normalized(disposition.formal_mechanism))
    dispositions = []
    by_id = {d.hypothesis_id: d for d in payload.inquiry_dispositions}
    for candidate in handoff["hypotheses"]:
        identifier = candidate["hypothesis_id"]
        item = by_id.get(identifier)
        if identifier in bad_ids or item is None:
            item = InquiryDisposition(
                hypothesis_id=identifier, disposition="deferred",
                reason="Discovery handoff contract failed; original idea retained without formal or Core authority.",
            )
            bad_names.add(normalized(candidate["mechanism"]))
        dispositions.append(item)
    safe = payload.model_copy(update={
        "mechanisms": tuple(m for m in payload.mechanisms if normalized(m.mechanism) not in bad_names),
        "inquiry_dispositions": tuple(dispositions),
        "new_mechanisms": tuple(n for n in payload.new_mechanisms or ()
                                if normalized(n.mechanism) not in bad_names),
        "identified_but_unsubmitted_mechanisms": (),
    })
    if validate_handoff(safe, handoff, observations):
        # Cross-linked/cyclic invalid model claims never become a guessed host repair.
        safe = MechanismDiscoveryPayload(inquiry_dispositions=tuple(
            InquiryDisposition(hypothesis_id=c["hypothesis_id"], disposition="deferred",
                               reason="Discovery handoff rejected; original investigation candidate retained.")
            for c in handoff["hypotheses"]
        ))
    return safe


def merge_partial_handoff(initial, retry, handoff, observations):
    """Version 4: independently admit safe retry groups without replacing valid originals."""
    base = quarantine_handoff(initial, handoff, observations, recover_exact=True)
    if retry is None:
        return base
    addition = quarantine_handoff(retry, handoff, observations, recover_exact=True)
    bad_ids = {identifier for _, identifier, _ in validate_handoff(initial, handoff, observations)
               if identifier is not None}
    originals = {c["hypothesis_id"]: c["mechanism"] for c in handoff["hypotheses"]}
    raw_by_id = {d.hypothesis_id: d for d in initial.inquiry_dispositions}
    base_by_id = {d.hypothesis_id: d for d in base.inquiry_dispositions}

    def names(disposition):
        result = {normalized(disposition.formal_mechanism)} if disposition.formal_mechanism else set()
        if disposition.disposition == "revised" and disposition.original_disposition == "retained":
            result.add(normalized(originals[disposition.hypothesis_id]))
        return result

    def admissible(trial):
        try:
            MechanismDiscoveryPayload.model_validate(trial.model_dump(mode="json"))
        except ValueError:
            return False
        return not validate_handoff(trial, handoff, observations)

    locked = {identifier for identifier, d in base_by_id.items()
              if names(d) or (identifier not in bad_ids and identifier in raw_by_id
                              and raw_by_id[identifier] == d and d.disposition == "discarded")}
    for disposition in addition.inquiry_dispositions:
        if disposition.hypothesis_id in locked:
            continue
        group = names(disposition)
        if not group:
            # Do not replace the original failed/deferred record with another failure.
            continue
        drafts = tuple(m for m in addition.mechanisms if normalized(m.mechanism) in group)
        occupied = {normalized(m.mechanism) for m in base.mechanisms}
        if occupied & group:
            continue
        trial = base.model_copy(update={
            "mechanisms": (*base.mechanisms, *drafts),
            "inquiry_dispositions": tuple(
                disposition if d.hypothesis_id == disposition.hypothesis_id else d
                for d in base.inquiry_dispositions
            ),
        })
        if admissible(trial):
            base = trial
    # New discoveries must have their own valid provenance; no inferred links.
    for declaration in addition.new_mechanisms or ():
        key = normalized(declaration.mechanism)
        if any(normalized(m.mechanism) == key for m in base.mechanisms):
            continue
        drafts = tuple(m for m in addition.mechanisms if normalized(m.mechanism) == key)
        trial = base.model_copy(update={
            "mechanisms": (*base.mechanisms, *drafts),
            "new_mechanisms": (*(base.new_mechanisms or ()), declaration),
        })
        if admissible(trial):
            base = trial
    return base


def pending_formalizations(payload, handoff):
    """Describe missing submission only; never guess a causal/data failure reason."""
    dispositions = {d.hypothesis_id: d for d in payload.inquiry_dispositions}
    submitted = {normalized(m.mechanism) for m in payload.mechanisms}
    return [{"hypothesis_id": c["hypothesis_id"], "mechanism": c["mechanism"],
             "code": "NO_ORIGINAL_FORMAL_DRAFT", "reported_reason": (
                 dispositions[c["hypothesis_id"]].reason if c["hypothesis_id"] in dispositions
                 else "No disposition was reported."),
             "instruction": "Submit the exact original draft or explicitly defer with the actual limitation; do not substitute its alternative."}
            for c in handoff["hypotheses"] if normalized(c["mechanism"]) not in submitted]
