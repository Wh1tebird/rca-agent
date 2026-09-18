"""Post-freeze coverage metrics; reuse Core eligibility with candidate-owned bindings."""
from __future__ import annotations

from rca_agent.core.dynamic import CompiledCandidate, assess_dynamic, verify_dynamic
from rca_agent.models import Observation
from rca_agent.runner import IssuedObservation


def evidence_coverage(result):
    ledger = {}
    responses = result["tool_responses"]
    for response in responses:
        for raw in response["observations"]:
            observation = Observation.model_validate(raw)
            ledger[observation.observation_id] = IssuedObservation(
                observation, response["query_ref"],
                response["status"] == "success" and not response["truncated"],
            )
    details = []
    for registered in result["registered_candidates"]:
        compiled = CompiledCandidate.model_validate(registered["compiled"])
        ids = {ref for p in result["policy_proposals"]
               if p["candidate_id"] == compiled.candidate.candidate_id
               for ref in p["observation_refs"]}
        records = tuple(ledger[ref] for ref in ids)
        assessments = tuple(assess_dynamic(compiled, records))
        verification = verify_dynamic(compiled, assessments, records)
        details.append({
            "candidate_id": compiled.candidate.candidate_id,
            "mechanism": compiled.candidate.fault_type,
            "required_checks": len(assessments),
            "checks_with_eligible_evidence": sum(a.observation is not None for a in assessments),
            "declared_alternatives": len(compiled.alternatives),
            "resolved_alternatives": len(verification.distinguished_alternatives),
            "verification": verification.model_dump(mode="json"),
            "checks": [
                {"rule": rule.rule_id, "polarity": condition.polarity.value,
                 "eligible": assessment.observation is not None,
                 "predicate_matched": assessment.matched,
                 "unavailable_reason": assessment.unavailable_reason}
                for rule, condition, assessment in zip(
                    compiled.rules, compiled.candidate.conditions, assessments, strict=True)
            ],
        })
    return {
        "queries": len(responses),
        "successful_queries": sum(r["status"] == "success" for r in responses),
        "queries_with_trusted_observations": sum(
            r["status"] == "success" and not r["truncated"] and any(
                o["data_status"] in {"observed", "derived"} and o["value"] is not None
                for o in r["observations"]
            ) for r in responses
        ),
        "required_checks": sum(c["required_checks"] for c in details),
        "checks_with_eligible_evidence": sum(c["checks_with_eligible_evidence"] for c in details),
        "declared_alternatives": sum(c["declared_alternatives"] for c in details),
        "resolved_alternatives": sum(c["resolved_alternatives"] for c in details),
        "candidates": details,
    }
