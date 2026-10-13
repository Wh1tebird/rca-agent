"""Host-bound candidate identity, separate from model-authored typed details."""
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rca_agent.core.dynamic import (
    AlternativeDraft,
    InquiryDisposition,
    MechanismClass,
    MechanismDiscoveryPayload,
    MechanismDraft,
)
from rca_agent.core.v3.models import MechanismFamily
from rca_agent.tools.fingerprint import sha256_digest

# Conservative lexical type hints, not root-cause or evidence rules. Multiple explicit
# layers remain ambiguous; absence of a match imposes no classification constraint.
_TYPE_CUES = {
    MechanismClass.DATABASE_EXECUTION: r"cross[- ]?join|slow sql|sql execution|query execution|database execution|复杂.*(?:sql|查询)|慢sql",
    MechanismClass.DATABASE_CONNECTION_POOL: r"connection pool|connection acquis|连接池|连接获取",
    MechanismClass.GC_PAUSE: r"garbage collection|\bgc\b|垃圾回收",
    MechanismClass.MEMORY_PRESSURE: r"memory (?:pressure|usage|exhaustion)|heap|内存",
    MechanismClass.THREAD_POOL_SATURATION: r"thread pool|worker (?:exhaustion|saturation)|线程池",
    MechanismClass.CPU_SATURATION: r"\bcpu\b",
    MechanismClass.NETWORK_LATENCY: r"network|\brtt\b|网络",
    MechanismClass.LOCK_CONTENTION: r"lock contention|锁竞争",
}


def compatible_classes(mechanism):
    return tuple(kind.value for kind, pattern in _TYPE_CUES.items()
                 if re.search(pattern, mechanism, re.IGNORECASE))


class CandidateDetails(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    causal_entity_ref: str = Field(pattern=r"^ent-[a-z0-9-]{3,40}$")
    mechanism_class: MechanismClass
    family: MechanismFamily
    causal_explanation: str = Field(min_length=10, max_length=2000)
    alternatives: tuple[AlternativeDraft, ...] = Field(min_length=1, max_length=8)


class BoundCandidateResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    status: Literal["formalized", "deferred"]
    details: CandidateDetails | None = None
    reason_code: Literal["FORMALIZED", "MISSING_FIELDS", "AMBIGUOUS_ENTITY", "UNSUPPORTED_CLASS"]
    reason: str = Field(min_length=1, max_length=1000)

    @model_validator(mode="after")
    def consistent(self):
        if ((self.status == "formalized") != (self.details is not None)
                or (self.status == "formalized") != (self.reason_code == "FORMALIZED")):
            raise ValueError("inconsistent candidate formalization")
        return self


class CauseDetails(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    causal_entity_ref: str = Field(pattern=r"^ent-[a-z0-9-]{3,40}$")
    mechanism_class: MechanismClass
    family: MechanismFamily
    causal_explanation: str = Field(min_length=10, max_length=2000)


class AlternativeDetails(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    alternative_id: str
    mechanism_class: MechanismClass


class BoundCandidateResponseV7(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    status: Literal["formalized", "deferred"]
    cause_details: CauseDetails | None = None
    alternative_details: tuple[AlternativeDetails, ...] = Field(default=(), max_length=20)
    reason_code: Literal["FORMALIZED", "MISSING_FIELDS", "AMBIGUOUS_ENTITY", "UNSUPPORTED_CLASS"]
    reason: str = Field(min_length=1, max_length=1000)

    @model_validator(mode="after")
    def consistent(self):
        if ((self.status == "formalized") != (self.cause_details is not None)
                or (self.status == "formalized") != (self.reason_code == "FORMALIZED")
                or self.status == "deferred" and self.alternative_details):
            raise ValueError("inconsistent candidate formalization")
        return self


def candidate_binding(candidate, source_hypothesis_id, *, version=6, handoff=None, revision=2):
    if type(version) is not int or version not in {5, 6, 7}:
        raise ValueError("unsupported bound handoff version")
    from rca_agent.investigation.contract_enforcement import validate_revision
    validate_revision(revision)
    result = {"hypothesis_id": candidate["hypothesis_id"], "mechanism": candidate["mechanism"],
            "alternative": candidate["alternative"], "source_hypothesis_id": source_hypothesis_id,
            "semantic_digest": sha256_digest(candidate)}
    if version >= 7:
        from rca_agent.investigation.inquiry_contract import normalized, original_alternative_id
        originals = (handoff or {}).get("hypotheses", [candidate])
        alternatives = []
        for original in originals:
            if normalized(original["mechanism"]) != normalized(candidate["mechanism"]):
                continue
            provenance = {"hypothesis_id": original["hypothesis_id"],
                          "source_hypothesis_id": source_hypothesis_id,
                          "semantic_digest": sha256_digest(original)}
            identifier = original.get("alternative_id") or original_alternative_id(original)
            if revision == 1 and any(a["alternative_id"] == identifier for a in alternatives):
                continue
            alternatives.append({"alternative_id": identifier,
                                 "mechanism": original["alternative"], "provenance": provenance})
        result.update(contract_version=7, alternative_bindings=alternatives)
        if revision == 2:
            result["validation_revision"] = 2
    return result


def bind_candidate(binding, response, issued_entities, *, check_class=False):
    """Fill host-owned identity only; never invent model-authored causal details."""
    version = binding.get("contract_version")
    if version is not None and (type(version) is not int or version != 7):
        raise ValueError("unsupported bound candidate contract")
    if binding.get("contract_version") == 7:
        from rca_agent.investigation.contract_enforcement import validate_revision
        validate_revision(binding.get("validation_revision", 1))
        from rca_agent.investigation.inquiry_contract import normalized
        if any(normalized(binding["mechanism"]) == normalized(a["mechanism"])
               for a in binding["alternative_bindings"]):
            raise ValueError("SELF_ALTERNATIVE")
        originals = binding["alternative_bindings"]
        if binding.get("validation_revision", 1) == 2:
            ids = [a["alternative_id"] for a in originals]
            if len(ids) != len(set(ids)):
                raise ValueError("ALTERNATIVE_ID_CONFLICT")
        response = BoundCandidateResponseV7.model_validate(response)
        if response.status == "deferred":
            return None, InquiryDisposition(hypothesis_id=binding["hypothesis_id"], disposition="deferred",
                                           reason=f"{response.reason_code}: {response.reason}")
        details = response.cause_details
        ids = [a.alternative_id for a in response.alternative_details]
        if (len(ids) != len(set(ids))
                or set(ids) != {a["alternative_id"] for a in originals}):
            raise ValueError("ALTERNATIVE_ID_MISMATCH")
        typed = {a.alternative_id: a for a in response.alternative_details}
        if details.causal_entity_ref not in issued_entities:
            raise ValueError("unissued candidate entity")
        allowed = compatible_classes(binding["mechanism"])
        if allowed and details.mechanism_class.value not in allowed:
            raise ValueError("candidate class conflicts with explicit causal terminology")
        alternatives = []
        for original in binding["alternative_bindings"]:
            kind = typed[original["alternative_id"]].mechanism_class
            allowed = compatible_classes(original["mechanism"])
            if kind != MechanismClass.UNKNOWN and allowed and kind.value not in allowed:
                raise ValueError("alternative class conflicts with explicit causal terminology")
            alternatives.append(AlternativeDraft(alternative_id=original["alternative_id"],
                mechanism=original["mechanism"], mechanism_class=kind,
                inquiry_provenance=original["provenance"]))
        draft = MechanismDraft(source_hypothesis_id=binding["source_hypothesis_id"],
            mechanism=binding["mechanism"], alternatives=tuple(alternatives),
            **details.model_dump(mode="python"))
        return draft, InquiryDisposition(hypothesis_id=binding["hypothesis_id"], disposition="kept",
                                        formal_mechanism=binding["mechanism"], reason=response.reason)
    response = BoundCandidateResponse.model_validate(response)
    if response.status == "deferred":
        return None, InquiryDisposition(hypothesis_id=binding["hypothesis_id"], disposition="deferred",
                                       reason=f"{response.reason_code}: {response.reason}")
    if response.details.causal_entity_ref not in issued_entities:
        raise ValueError("unissued candidate entity")
    allowed = compatible_classes(binding["mechanism"])
    if check_class and allowed and response.details.mechanism_class.value not in allowed:
        raise ValueError("candidate class conflicts with explicit causal terminology")
    draft = MechanismDraft(
        source_hypothesis_id=binding["source_hypothesis_id"], mechanism=binding["mechanism"],
        **response.details.model_dump(mode="python"),
    )
    return draft, InquiryDisposition(hypothesis_id=binding["hypothesis_id"], disposition="kept",
                                     formal_mechanism=binding["mechanism"],
                                     reason="Formalized original candidate; evidence and causal confirmation remain Core-owned.")


def bound_payload(handoff, records, issued_entities, *, check_class=False):
    drafts, dispositions = [], []
    for candidate in handoff["hypotheses"]:
        item = records[candidate["hypothesis_id"]]
        if item.get("response") is None:
            dispositions.append(InquiryDisposition(hypothesis_id=candidate["hypothesis_id"],
                disposition="deferred", reason=item["reason_code"]))
            continue
        draft, disposition = bind_candidate(item["binding"], item["response"], issued_entities,
                                            check_class=check_class)
        if draft is not None and (item["binding"].get("contract_version") != 7 or not any(
                " ".join(d.mechanism.casefold().split()) == " ".join(draft.mechanism.casefold().split())
                for d in drafts
            )):
            drafts.append(draft)
        dispositions.append(disposition)
    return MechanismDiscoveryPayload(mechanisms=tuple(drafts), inquiry_dispositions=tuple(dispositions))


def admit_novel(base, novel, handoff, observations, source_ids, issued_entities):
    from rca_agent.investigation.discovery_handoff import normalized, validate_handoff

    if novel is None:
        return base
    counts = {name: sum(normalized(m.mechanism) == name for m in novel.mechanisms)
              for name in {normalized(m.mechanism) for m in novel.mechanisms}}
    originals = {normalized(c["mechanism"]) for c in handoff["hypotheses"]}
    for declaration in novel.new_mechanisms or ():
        name = normalized(declaration.mechanism)
        if (name in originals or counts.get(name) != 1
                or any(normalized(m.mechanism) == name for m in base.mechanisms)):
            continue
        draft = next(m for m in novel.mechanisms if normalized(m.mechanism) == name)
        if draft.source_hypothesis_id not in source_ids or draft.causal_entity_ref not in issued_entities:
            continue
        try:
            trial = MechanismDiscoveryPayload(
                mechanisms=(*base.mechanisms, draft), inquiry_dispositions=base.inquiry_dispositions,
                new_mechanisms=(*(base.new_mechanisms or ()), declaration),
            )
        except ValueError:
            continue
        if not validate_handoff(trial, handoff, observations):
            base = trial
    return base
